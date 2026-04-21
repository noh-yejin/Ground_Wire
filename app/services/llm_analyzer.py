from __future__ import annotations

from hashlib import sha1
import logging
import re
from pydantic import BaseModel, Field

from app.config import settings
from app.models import (
    AnalysisResult,
    Article,
    EvidenceSnippet,
    ImpactLabel,
    IssuePriority,
    ReliabilityScore,
    RiskLevel,
    SentimentLabel,
)
from app.services.clustering import _extract_keywords
from app.services.rag import EvidenceRetriever
from app.services.reliability import score_grounding
from app.services.source_normalizer import is_trusted_ready_source, normalize_source_name

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional in tests
    OpenAI = None

logger = logging.getLogger(__name__)

NEGATIVE_HINTS = ("위험", "후퇴", "둔화", "긴축", "변동성", "제재")
POSITIVE_HINTS = ("증가", "개선", "회복", "확대", "기대", "성장")
CONTRADICTION_HINTS = ("아니다", "부인", "반박", "정정", "철회", "논란", "상충", "사실무근", "불확실")


class IssueAnalysisSchema(BaseModel):
    summary: str = Field(description="Evidence-grounded issue summary")
    keywords: list[str] = Field(description="Issue keywords")
    key_signals: list[str] = Field(description="Important priority signals")
    key_points: list[str] = Field(description="Top short bullet points in Korean")
    trend_summary: str = Field(description="Short Korean explanation of whether issue attention is rising, flat, or falling")
    sentiment: SentimentLabel = Field(description="Overall sentiment")
    market_impact: ImpactLabel = Field(description="Estimated market impact direction")
    policy_risk: RiskLevel = Field(description="Estimated policy risk level")
    volatility_risk: RiskLevel = Field(description="Estimated volatility risk level")
    risk_points: list[str] = Field(description="Key risks or caveats")
    grounded: bool = Field(description="Whether the answer is grounded in evidence")
    priority: IssuePriority = Field(description="priority or general")
    hold_reason: str | None = Field(default=None, description="Why the issue should be held if not ready")


class ClaimVerificationSchema(BaseModel):
    verification_label: str = Field(description="support, partial_support, contradict, or insufficient")
    confidence: float = Field(description="0 to 1 confidence")
    matched_entities: list[str] = Field(description="Entities or terms directly matched between claim and evidence")
    rationale: str = Field(description="Short Korean explanation grounded in the evidence")


class ClaimExtractionSchema(BaseModel):
    claims: list[str] = Field(description="Atomic, evidence-checkable Korean claims")


class IssueAnalysisBundleSchema(IssueAnalysisSchema):
    claims: list[str] = Field(description="Atomic, evidence-checkable Korean claims used for grounding")


class HoldReviewSchema(BaseModel):
    promote_to_ready: bool = Field(description="Whether this borderline HOLD issue should be promoted to READY")
    rationale: str = Field(description="Short Korean rationale for the promotion decision")
    revised_summary: str | None = Field(default=None, description="Optional revised Korean summary if promoted")


class LLMAnalyzer:
    """Backend-only analysis interface.

    Uses OpenAI Responses API on the backend when an API key is configured.
    Falls back to deterministic local analysis when no API key is available.
    """

    def __init__(self) -> None:
        self.last_remote_error: str | None = None
        self.last_remote_success: bool = False
        self.retriever = EvidenceRetriever()
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds, max_retries=0) if (
            settings.openai_api_key and OpenAI is not None
        ) else None
        self._analysis_cache: dict[str, tuple[AnalysisResult, list[str]]] = {}
        self._claim_cache: dict[str, list[str]] = {}

    def analyze(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
        hold_reason: str | None,
        corpus_articles: list[Article] | None = None,
    ) -> AnalysisResult:
        joined_titles = " ".join(article.title for article in articles)
        joined_content = " ".join(article.content for article in articles)
        heuristic_keywords = _extract_keywords(f"{joined_titles} {joined_content}")[:6]
        heuristic_key_signals = derive_key_signals(topic, articles, heuristic_keywords)
        heuristic_trend_summary = derive_trend_summary(articles)
        heuristic_sentiment = detect_sentiment(joined_titles + " " + joined_content)
        heuristic_priority = derive_priority(topic, articles, reliability, heuristic_key_signals)
        heuristic_market_impact = derive_market_impact(topic, articles, heuristic_sentiment)
        heuristic_policy_risk = derive_policy_risk(topic, articles, heuristic_key_signals)
        heuristic_volatility_risk = derive_volatility_risk(topic, articles, heuristic_sentiment)

        should_use_remote_analysis = self.client is not None and _should_use_remote_analysis(articles, evidence, reliability)
        remote_bundle = self._analyze_bundle_with_openai(topic, articles, evidence, reliability) if should_use_remote_analysis else None
        remote_analysis = remote_bundle[0] if remote_bundle else None
        remote_claims = remote_bundle[1] if remote_bundle else None
        keywords = _merge_distinct_texts(
            (remote_analysis.keywords if remote_analysis else []),
            heuristic_keywords,
            limit=6,
        )
        key_signals = _merge_distinct_texts(
            (remote_analysis.key_signals if remote_analysis else []),
            derive_key_signals(topic, articles, keywords),
            limit=5,
        )
        trend_summary = _choose_preferred_text(
            remote_analysis.trend_summary if remote_analysis else "",
            heuristic_trend_summary,
        )
        sentiment = remote_analysis.sentiment if remote_analysis else heuristic_sentiment
        priority = remote_analysis.priority if remote_analysis else heuristic_priority
        market_impact = remote_analysis.market_impact if remote_analysis else heuristic_market_impact
        policy_risk = remote_analysis.policy_risk if remote_analysis else heuristic_policy_risk
        volatility_risk = remote_analysis.volatility_risk if remote_analysis else heuristic_volatility_risk
        claim_results, remote_claim_extraction_used = self._build_grounded_claims(
            topic,
            articles,
            evidence,
            keywords,
            corpus_articles or articles,
            allow_remote_extraction=should_use_remote_analysis,
            prefetched_remote_claims=remote_claims,
        )
        grounding = score_grounding(claim_results, reliability)
        grounded_summary = self._build_grounded_summary(topic, articles, claim_results, evidence, reliability)
        fallback_points = grounded_summary["key_points"] or derive_key_points(evidence, articles)
        remote_points = remote_analysis.key_points if remote_analysis else []
        key_points = _merge_distinct_texts(remote_points, fallback_points, limit=4)
        summary = _choose_preferred_text(
            remote_analysis.summary if remote_analysis else "",
            grounded_summary["summary"],
            article_titles=[article.title for article in articles],
        )
        effective_hold_reason = hold_reason or (remote_analysis.hold_reason if remote_analysis else None)
        if effective_hold_reason is None and (
            grounding["grounded_ratio"] < settings.min_grounded_ratio
            or grounding["issue_score"] < settings.min_grounding_issue_score
            or len(grounded_summary["grounded_claim_ids"]) < settings.min_grounded_claims
        ):
            effective_hold_reason = (
                f"grounding 검증 부족: grounded_ratio={grounding['grounded_ratio']}, issue_score={grounding['issue_score']}"
            )
        risk_points = _merge_distinct_texts(
            remote_analysis.risk_points if remote_analysis else [],
            derive_risk_points(articles, reliability, effective_hold_reason),
            limit=5,
        )
        for reason in grounding["reasons"]:
            if reason not in risk_points:
                risk_points.append(reason)
        grounded_flag = effective_hold_reason is None and bool(grounded_summary["grounded_claim_ids"])
        decision = _build_decision_details(
            effective_hold_reason=effective_hold_reason,
            grounding=grounding,
            grounded_summary=grounded_summary,
            claim_results=claim_results,
        )

        if effective_hold_reason:
            return AnalysisResult(
                summary=f"보류된 이슈입니다. 사유: {effective_hold_reason}. {summary}",
                keywords=keywords,
                key_signals=key_signals,
                key_points=key_points,
                trend_summary=trend_summary,
                sentiment=sentiment,
                market_impact=market_impact,
                policy_risk=policy_risk,
                volatility_risk=volatility_risk,
                risk_points=risk_points,
                grounded=grounded_flag,
                priority=priority,
                hold_reason=effective_hold_reason,
                grounding_details={
                    "claims": claim_results,
                    "grounding": grounding,
                    "grounded_summary": grounded_summary,
                    "decision": decision,
                    "llm": {
                        "analysis_used": remote_analysis is not None,
                        "claim_extraction_used": remote_claim_extraction_used,
                        "analysis_mode": "combined_remote" if remote_analysis is not None else "local",
                    },
                },
            )

        return AnalysisResult(
            summary=summary,
            keywords=keywords,
            key_signals=key_signals,
            key_points=key_points,
            trend_summary=trend_summary,
            sentiment=sentiment,
            market_impact=market_impact,
            policy_risk=policy_risk,
            volatility_risk=volatility_risk,
            risk_points=risk_points,
            grounded=grounded_flag,
            priority=priority,
            hold_reason=None,
            grounding_details={
                "claims": claim_results,
                "grounding": grounding,
                "grounded_summary": grounded_summary,
                "decision": decision,
                "llm": {
                    "analysis_used": remote_analysis is not None,
                    "claim_extraction_used": remote_claim_extraction_used,
                    "analysis_mode": "combined_remote" if remote_analysis is not None else "local",
                },
            },
        )

    def debug_status(self) -> dict:
        return {
            "openai_package_available": OpenAI is not None,
            "api_key_configured": bool(settings.openai_api_key),
            "client_initialized": self.client is not None,
            "model": settings.openai_model,
            "last_remote_success": self.last_remote_success,
            "last_remote_error": self.last_remote_error,
        }

    def analyze_lightweight(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
        hold_reason: str,
    ) -> AnalysisResult:
        joined_titles = " ".join(article.title for article in articles)
        joined_content = " ".join(article.content for article in articles)
        keywords = _extract_keywords(f"{joined_titles} {joined_content}")[:6]
        key_signals = derive_key_signals(topic, articles, keywords)
        summary = build_local_summary(topic, articles, evidence, reliability)
        key_points = derive_key_points(evidence, articles)[:4]
        sentiment = detect_sentiment(joined_titles + " " + joined_content)
        return AnalysisResult(
            summary=f"보류된 이슈입니다. 사유: {hold_reason}. {summary}",
            keywords=keywords,
            key_signals=key_signals,
            key_points=key_points,
            trend_summary=derive_trend_summary(articles),
            sentiment=sentiment,
            market_impact=derive_market_impact(topic, articles, sentiment),
            policy_risk=derive_policy_risk(topic, articles, key_signals),
            volatility_risk=derive_volatility_risk(topic, articles, sentiment),
            risk_points=derive_risk_points(articles, reliability, hold_reason),
            grounded=False,
            priority=derive_priority(topic, articles, reliability, key_signals),
            hold_reason=hold_reason,
            grounding_details={
                "claims": [],
                "grounding": {
                    "grounded_ratio": 0.0,
                    "issue_score": round(reliability.value, 3),
                    "reasons": [hold_reason],
                },
                "grounded_summary": {
                    "summary": summary,
                    "key_points": key_points,
                    "grounded_claim_ids": [],
                    "omitted_claims": [],
                },
                "decision": _build_decision_details(
                    effective_hold_reason=hold_reason,
                    grounding={"grounded_ratio": 0.0, "issue_score": round(reliability.value, 3), "reasons": [hold_reason]},
                    grounded_summary={"grounded_claim_ids": []},
                    claim_results=[],
                ),
                "llm": {
                    "analysis_used": False,
                    "claim_extraction_used": False,
                    "analysis_mode": "lightweight_hold",
                },
            },
        )

    def review_hold_for_promotion(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
        analysis: AnalysisResult,
    ) -> AnalysisResult | None:
        if self.client is None:
            return None
        details = analysis.grounding_details or {}
        grounding = details.get("grounding", {})
        decision = details.get("decision", {})
        claims = details.get("claims", [])
        ready_claims = [item for item in claims if item.get("ready")]
        if len(ready_claims) < 3:
            return None
        try:
            response = self.client.responses.parse(
                model=settings.openai_model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a trust-first escalation reviewer. "
                            "Only promote a HOLD issue to READY when the evidence is strong enough for production display. "
                            "Be conservative and answer in Korean."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_hold_review_prompt(
                            topic=topic,
                            articles=articles,
                            evidence=evidence,
                            reliability=reliability,
                            analysis=analysis,
                            grounding=grounding,
                            decision=decision,
                            ready_claims=ready_claims,
                        ),
                    },
                ],
                text_format=HoldReviewSchema,
            )
        except Exception as exc:
            self.last_remote_success = False
            self.last_remote_error = f"{type(exc).__name__}: {exc}"
            logger.warning("OpenAI hold review failed: %s", self.last_remote_error)
            return None

        parsed = getattr(response, "output_parsed", None)
        if parsed is None or not parsed.promote_to_ready:
            return None
        self.last_remote_success = True
        revised_summary = _choose_preferred_text(parsed.revised_summary or "", analysis.summary)
        grounding_details = dict(details)
        llm = dict(grounding_details.get("llm", {}))
        llm["second_pass_reviewed"] = True
        llm["second_pass_promoted"] = True
        grounding_details["llm"] = llm
        grounding_details["decision"] = {
            "status": "READY",
            "summary": "2차 검증에서 승격되었습니다.",
            "ready_claim_count": decision.get("ready_claim_count", len(ready_claims)),
            "total_claim_count": decision.get("total_claim_count", len(claims)),
            "grounded_claim_count": decision.get("grounded_claim_count", len(ready_claims)),
            "reasons": [
                {
                    "code": "ready_second_pass",
                    "label": "2차 검증 승격",
                    "detail": parsed.rationale,
                }
            ],
        }
        risk_points = list(analysis.risk_points)
        if parsed.rationale and parsed.rationale not in risk_points:
            risk_points.append(parsed.rationale)
        return AnalysisResult(
            summary=re.sub(r"^보류된 이슈입니다\.\s*사유:\s*[^.]+\.\s*", "", revised_summary).strip(),
            keywords=analysis.keywords,
            key_signals=analysis.key_signals,
            key_points=analysis.key_points,
            trend_summary=analysis.trend_summary,
            sentiment=analysis.sentiment,
            market_impact=analysis.market_impact,
            policy_risk=analysis.policy_risk,
            volatility_risk=analysis.volatility_risk,
            risk_points=risk_points,
            grounded=True,
            priority=analysis.priority,
            hold_reason=None,
            grounding_details=grounding_details,
        )

    def _analyze_with_openai(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
    ) -> AnalysisResult | None:
        bundle = self._analyze_bundle_with_openai(topic, articles, evidence, reliability)
        if bundle is None:
            return None
        return bundle[0]

    def _analyze_bundle_with_openai(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
    ) -> tuple[AnalysisResult, list[str]] | None:
        cache_key = _analysis_cache_key(topic, articles, evidence, reliability)
        if cache_key in self._analysis_cache:
            self.last_remote_success = True
            self.last_remote_error = None
            return self._analysis_cache[cache_key]
        self.last_remote_error = None
        try:
            response = self.client.responses.parse(
                model=settings.openai_model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reliability-first issue analysis model. "
                            "Only use the provided evidence. If evidence is insufficient, say so in hold_reason. "
                            "All outputs must be written in Korean, even when the source articles are in English. "
                            "Translate the source meaning into Korean before summarizing. "
                            "Return concise Korean JSON-compatible structured output."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_issue_prompt(topic, articles, evidence, reliability, include_claims=True),
                    },
                ],
                text_format=IssueAnalysisBundleSchema,
            )
        except Exception as exc:
            self.last_remote_success = False
            self.last_remote_error = f"{type(exc).__name__}: {exc}"
            logger.warning("OpenAI issue analysis failed: %s", self.last_remote_error)
            return None

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            self.last_remote_success = False
            self.last_remote_error = "No output_parsed returned from Responses API."
            logger.warning("OpenAI issue analysis failed: %s", self.last_remote_error)
            return None
        self.last_remote_success = True
        result = AnalysisResult(
            summary=parsed.summary,
            keywords=parsed.keywords[:6],
            key_signals=parsed.key_signals[:5],
            key_points=parsed.key_points[:4],
            trend_summary=parsed.trend_summary,
            sentiment=parsed.sentiment,
            market_impact=parsed.market_impact,
            policy_risk=parsed.policy_risk,
            volatility_risk=parsed.volatility_risk,
            risk_points=parsed.risk_points[:5],
            grounded=parsed.grounded,
            priority=parsed.priority,
            hold_reason=parsed.hold_reason,
            grounding_details={},
        )
        claims = self._normalize_remote_claims(getattr(parsed, "claims", []))
        bundle = (result, claims)
        self._analysis_cache[cache_key] = bundle
        return bundle

    def _build_grounded_claims(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        keywords: list[str],
        corpus_articles: list[Article],
        allow_remote_extraction: bool,
        prefetched_remote_claims: list[str] | None = None,
    ) -> tuple[list[dict], bool]:
        candidates, remote_used = self._extract_candidate_claims(
            topic,
            articles,
            evidence,
            keywords,
            allow_remote_extraction=allow_remote_extraction,
            prefetched_remote_claims=prefetched_remote_claims,
        )
        return [self._verify_claim(candidate, articles, corpus_articles) for candidate in candidates], remote_used

    def _extract_candidate_claims(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        keywords: list[str],
        allow_remote_extraction: bool,
        prefetched_remote_claims: list[str] | None = None,
    ) -> tuple[list[str], bool]:
        remote_claims = prefetched_remote_claims
        if remote_claims is None:
            remote_claims = (
                self._extract_candidate_claims_with_openai(topic, articles, evidence, keywords)
                if self.client is not None and allow_remote_extraction
                else None
            )
        heuristic_claims = self._extract_heuristic_claims(articles, evidence, keywords)
        if remote_claims:
            return _merge_distinct_texts(remote_claims, heuristic_claims, limit=6), True

        return heuristic_claims[:6], False

    def _extract_heuristic_claims(
        self,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        keywords: list[str],
    ) -> list[str]:
        candidates: list[str] = []
        seed_texts = [item.quote for item in evidence[:4]]
        seed_texts.extend(article.content.split(".")[0].strip() for article in articles[:4] if article.content)
        seed_texts.extend(article.title for article in articles[:4])
        for raw in seed_texts:
            cleaned = _clean_claim(_normalize_summary_text(raw))
            if not cleaned:
                continue
            if len(cleaned) < 8:
                continue
            if any(_is_similar_summary_text(cleaned, existing) for existing in candidates):
                continue
            if cleaned.lower() in {keyword.lower() for keyword in keywords[:3]}:
                continue
            if cleaned:
                candidates.append(cleaned)
        return candidates[:6]

    def _extract_candidate_claims_with_openai(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        keywords: list[str],
    ) -> list[str] | None:
        cache_key = _claim_cache_key(topic, articles, evidence, keywords)
        if cache_key in self._claim_cache:
            self.last_remote_success = True
            self.last_remote_error = None
            return self._claim_cache[cache_key]
        self.last_remote_error = None
        try:
            response = self.client.responses.parse(
                model=settings.openai_model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You extract atomic, evidence-checkable claims from news articles. "
                            "Return 2 to 6 short Korean claims. "
                            "Each claim must be factual, specific, and directly verifiable from the provided evidence. "
                            "Do not include opinions, market predictions, or duplicate paraphrases."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_claim_extraction_prompt(topic, articles, evidence, keywords),
                    },
                ],
                text_format=ClaimExtractionSchema,
            )
        except Exception as exc:
            self.last_remote_success = False
            self.last_remote_error = f"{type(exc).__name__}: {exc}"
            logger.warning("OpenAI claim extraction failed: %s", self.last_remote_error)
            return None

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            self.last_remote_success = False
            self.last_remote_error = "No output_parsed returned from claim extraction."
            return None
        self.last_remote_success = True
        result = self._normalize_remote_claims(parsed.claims)
        if result:
            self._claim_cache[cache_key] = result
        return result

    def _normalize_remote_claims(self, claims: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in claims:
            cleaned = _clean_claim(_normalize_summary_text(raw))
            if not cleaned:
                continue
            if any(_is_similar_summary_text(cleaned, existing) for existing in normalized):
                continue
            normalized.append(cleaned)
        return normalized[:6]

    def _verify_claim(self, claim: str, articles: list[Article], corpus_articles: list[Article]) -> dict:
        support_evidence = self.retriever.retrieve_for_claim(claim, articles, corpus_articles=corpus_articles)
        external_evidence = self.retriever.retrieve_external_for_claim(claim, articles, corpus_articles=corpus_articles)
        counter_evidence = self.retriever.retrieve_counter_evidence(claim, articles, corpus_articles=corpus_articles)
        evidence_rows: list[dict] = []
        support_count = 0
        contradiction_count = 0
        trusted_support_count = 0
        verification_scores: list[float] = []
        local_support_count = 0
        external_support_count = 0

        cluster_ids = {article.id for article in articles}
        support_pool = self._merge_evidence_lists(support_evidence, external_evidence)

        for item in support_pool:
            verification = self._classify_claim_with_grounding(claim, item)
            label = verification["label"]
            trusted = is_trusted_ready_source(item.source)
            if label != "insufficient":
                verification_scores.append(float(verification["score"]))
            if label in {"support", "partial_support"}:
                support_count += 1
                if item.article_id in cluster_ids:
                    local_support_count += 1
                else:
                    external_support_count += 1
                if trusted:
                    trusted_support_count += 1
            evidence_rows.append(
                {
                    "article_id": item.article_id,
                    "source": item.source,
                    "url": item.url,
                    "quote": item.quote[:220],
                    "retrieval_score": round(item.score, 3),
                    "verification_label": label,
                    "verification_score": round(float(verification["score"]), 3),
                    "trusted_source": trusted,
                    "entity_match_score": verification["entity_match_score"],
                    "number_match_score": verification["number_match_score"],
                    "matched_entities": verification["matched_entities"],
                    "scope": "local" if item.article_id in cluster_ids else "external",
                    "rationale": verification["rationale"],
                }
            )

        for item in counter_evidence:
            verification = self._classify_claim_with_grounding(claim, item, contradiction_mode=True)
            contradicted = verification["label"] == "contradict"
            if not contradicted:
                continue
            contradiction_count += 1
            evidence_rows.append(
                {
                    "article_id": item.article_id,
                    "source": item.source,
                    "url": item.url,
                    "quote": item.quote[:220],
                    "retrieval_score": round(item.score, 3),
                    "verification_label": "contradict",
                    "verification_score": round(float(verification["score"]), 3),
                    "trusted_source": is_trusted_ready_source(item.source),
                    "entity_match_score": verification["entity_match_score"],
                    "number_match_score": verification["number_match_score"],
                    "matched_entities": verification["matched_entities"],
                    "scope": "counter",
                    "rationale": verification["rationale"],
                }
            )

        avg_verification = sum(verification_scores) / max(len(verification_scores), 1)
        external_diversity_bonus = min(external_support_count / 2, 1.0)
        score = (
            min(support_count / 3, 1.0) * 0.36
            + min(trusted_support_count / 2, 1.0) * 0.30
            + avg_verification * 0.18
            + external_diversity_bonus * 0.10
            + min(len(claim.split()) / 12, 1.0) * 0.06
            - min(contradiction_count * 0.22, 0.44)
        )
        score = round(max(0.0, min(score, 1.0)), 3)
        return {
            "claim": claim,
            "support_count": support_count,
            "trusted_support_count": trusted_support_count,
            "contradiction_count": contradiction_count,
            "local_support_count": local_support_count,
            "external_support_count": external_support_count,
            "score": score,
            "ready": (
                support_count >= 2
                and trusted_support_count >= 1
                and contradiction_count == 0
                and score >= 0.72
                and (external_support_count >= 1 or local_support_count >= 2)
            ),
            "evidence": evidence_rows[:6],
        }

    def _build_grounded_summary(
        self,
        topic: str,
        articles: list[Article],
        claim_results: list[dict],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
    ) -> dict:
        grounded_claims = [item for item in claim_results if item.get("ready")]
        grounded_claims.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        grounded_claim_ids = [item["claim"] for item in grounded_claims[:3]]
        if grounded_claims:
            distinct_claims: list[str] = []
            for item in grounded_claims:
                claim = _normalize_summary_text(item["claim"])
                if not claim:
                    continue
                if any(_is_similar_summary_text(claim, existing) for existing in distinct_claims):
                    continue
                distinct_claims.append(claim)
            summary = " ".join(distinct_claims[:2])
            key_points = distinct_claims[:4]
            omitted_claims = [item["claim"] for item in claim_results if not item.get("ready")][:4]
            return {
                "summary": summary[:320],
                "key_points": key_points,
                "grounded_claim_ids": grounded_claim_ids,
                "omitted_claims": omitted_claims,
            }

        fallback = build_local_summary(topic, articles, evidence, reliability) if evidence else build_local_summary(topic, articles, [], reliability)
        return {
            "summary": fallback,
            "key_points": derive_key_points(evidence, articles)[:4],
            "grounded_claim_ids": [],
            "omitted_claims": [item["claim"] for item in claim_results[:4]],
        }

    def _classify_claim_with_grounding(
        self,
        claim: str,
        evidence: EvidenceSnippet,
        contradiction_mode: bool = False,
    ) -> dict:
        overlap = _token_overlap_score(claim, evidence.quote)
        entity_match = _entity_match_score(claim, evidence.quote)
        number_match = _number_match_score(claim, evidence.quote)
        heuristic_score = round(overlap * 0.45 + entity_match * 0.35 + number_match * 0.20, 3)
        heuristic_label = _heuristic_verification_label(
            heuristic_score=heuristic_score,
            overlap=overlap,
            contradiction_mode=contradiction_mode,
            quote=evidence.quote,
        )

        if self.client is not None and settings.enable_llm_claim_verification:
            llm_result = self._verify_with_openai(claim, evidence.quote, contradiction_mode=contradiction_mode)
            if llm_result is not None:
                merged_score = round(
                    max(0.0, min((heuristic_score * 0.45) + (llm_result["confidence"] * 0.55), 1.0)),
                    3,
                )
                final_label = llm_result["verification_label"]
                if contradiction_mode and final_label != "contradict":
                    final_label = heuristic_label
                return {
                    "label": final_label,
                    "score": merged_score,
                    "entity_match_score": entity_match,
                    "number_match_score": number_match,
                    "matched_entities": llm_result["matched_entities"] or _matched_entities(claim, evidence.quote),
                    "rationale": llm_result["rationale"],
                }

        return {
            "label": heuristic_label,
            "score": heuristic_score,
            "entity_match_score": entity_match,
            "number_match_score": number_match,
            "matched_entities": _matched_entities(claim, evidence.quote),
            "rationale": "휴리스틱 grounding 판정",
        }

    def _verify_with_openai(self, claim: str, quote: str, contradiction_mode: bool = False) -> dict | None:
        self.last_remote_error = None
        try:
            response = self.client.responses.parse(
                model=settings.openai_model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a claim verification model. "
                            "Only judge whether the evidence supports, partially supports, contradicts, or is insufficient for the claim. "
                            "Return concise Korean rationale."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Claim: {claim}\n"
                            f"Evidence: {quote}\n"
                            f"Mode: {'contradiction_check' if contradiction_mode else 'support_check'}\n"
                            "Output one verification label and confidence."
                        ),
                    },
                ],
                text_format=ClaimVerificationSchema,
            )
        except Exception as exc:
            self.last_remote_success = False
            self.last_remote_error = f"{type(exc).__name__}: {exc}"
            logger.warning("OpenAI claim verification failed: %s", self.last_remote_error)
            return None

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            return None
        self.last_remote_success = True
        label = str(parsed.verification_label).strip().lower()
        if label not in {"support", "partial_support", "contradict", "insufficient"}:
            label = "insufficient"
        return {
            "verification_label": label,
            "confidence": float(parsed.confidence),
            "matched_entities": parsed.matched_entities,
            "rationale": parsed.rationale,
        }

    def _merge_evidence_lists(self, *lists: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        merged: dict[tuple[str, str], EvidenceSnippet] = {}
        for items in lists:
            for item in items:
                key = (item.article_id, item.quote)
                if key not in merged or item.score > merged[key].score:
                    merged[key] = item
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def detect_sentiment(text: str) -> SentimentLabel:
    lowered = text.lower()
    positive = sum(1 for token in POSITIVE_HINTS if token in lowered)
    negative = sum(1 for token in NEGATIVE_HINTS if token in lowered)
    if positive and negative:
        return SentimentLabel.MIXED
    if positive > negative:
        return SentimentLabel.POSITIVE
    if negative > positive:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


def derive_risk_points(
    articles: list[Article],
    reliability: ReliabilityScore,
    hold_reason: str | None,
) -> list[str]:
    risks: list[str] = []
    if hold_reason:
        risks.append(hold_reason)
    if reliability.source_diversity < 0.6:
        risks.append("출처 다양성이 충분하지 않을 수 있습니다.")
    if reliability.cross_source_confirmation < 0.75:
        risks.append("복수 출처 교차 검증이 약합니다.")
    if any("변동성" in article.title for article in articles):
        risks.append("시장 변동성 확대 가능성에 유의해야 합니다.")
    return risks or ["현재 기준에서 두드러진 리스크는 제한적입니다."]


def derive_key_signals(topic: str, articles: list[Article], keywords: list[str]) -> list[str]:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    signals = [keyword for keyword in settings.priority_keywords if keyword.lower() in haystack.lower()]
    ordered: list[str] = []
    for signal in signals + keywords:
        if signal == "nbsp":
            continue
        if signal not in ordered:
            ordered.append(signal)
    return ordered[:5]


def derive_key_points(evidence: list[EvidenceSnippet], articles: list[Article]) -> list[str]:
    candidates: list[str] = []
    article_titles = [_normalize_summary_text(article.title) for article in articles[:6]]
    if evidence:
        for item in evidence[:6]:
            cleaned = _normalize_summary_text(item.quote)
            if not cleaned:
                continue
            if any(_is_similar_summary_text(cleaned, title) for title in article_titles if title):
                continue
            if any(_is_similar_summary_text(cleaned, existing) for existing in candidates):
                continue
            candidates.append(cleaned[:120])
            if len(candidates) >= 3:
                break
    if candidates:
        return candidates

    fallback: list[str] = []
    for article in articles[:4]:
        for raw in (article.content, article.title):
            sentence = _normalize_summary_text(raw)
            if not sentence:
                continue
            if any(_is_similar_summary_text(sentence, existing) for existing in fallback):
                continue
            fallback.append(sentence[:120])
            break
        if len(fallback) >= 3:
            break
    return fallback


def derive_market_impact(topic: str, articles: list[Article], sentiment: SentimentLabel) -> ImpactLabel:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    lowered = haystack.lower()
    if any(token in lowered for token in ("금리", "inflation", "tariff", "oil", "유가", "관세", "연준", "환율")):
        if sentiment in (SentimentLabel.NEGATIVE, SentimentLabel.MIXED):
            return ImpactLabel.NEGATIVE
        if sentiment == SentimentLabel.POSITIVE:
            return ImpactLabel.POSITIVE
    return ImpactLabel(sentiment.value) if sentiment != SentimentLabel.MIXED else ImpactLabel.MIXED


def derive_policy_risk(topic: str, articles: list[Article], key_signals: list[str]) -> RiskLevel:
    haystack = f"{topic} " + " ".join(article.title for article in articles) + " " + " ".join(key_signals)
    lowered = haystack.lower()
    high_tokens = ("정책", "regulation", "sanction", "제재", "관세", "금리", "연준", "정부", "국회")
    medium_tokens = ("guidance", "실적", "전망", "협상", "정상회담")
    if any(token in lowered for token in high_tokens):
        return RiskLevel.HIGH
    if any(token in lowered for token in medium_tokens):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def derive_volatility_risk(topic: str, articles: list[Article], sentiment: SentimentLabel) -> RiskLevel:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    lowered = haystack.lower()
    if any(token in lowered for token in ("분쟁", "conflict", "war", "attack", "tariff", "제재", "유가", "환율")):
        return RiskLevel.HIGH
    if sentiment == SentimentLabel.MIXED or any(token in lowered for token in ("실적", "guidance", "변동성", "긴축")):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def derive_trend_summary(articles: list[Article]) -> str:
    if len(articles) < 2:
        return "표본이 적어 추세를 판단하기 어렵습니다."
    ordered = sorted(articles, key=lambda article: article.published_at)
    midpoint = max(len(ordered) // 2, 1)
    early_count = len(ordered[:midpoint])
    late_count = len(ordered[midpoint:])
    if late_count >= early_count * 1.5:
        return "최근 시간대로 갈수록 기사 출현이 늘어나는 상승 추세입니다."
    if early_count >= late_count * 1.5:
        return "초반 대비 최근 기사 출현이 줄어드는 둔화 추세입니다."
    return "최근 시간대 기준 기사 출현 빈도가 비교적 안정적으로 유지되고 있습니다."


def derive_priority(
    topic: str,
    articles: list[Article],
    reliability: ReliabilityScore,
    key_signals: list[str],
) -> IssuePriority:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    signal_count = sum(1 for keyword in settings.priority_keywords if keyword.lower() in haystack.lower())
    if signal_count >= 2 or (signal_count >= 1 and reliability.value >= 0.75):
        return IssuePriority.PRIORITY
    return IssuePriority.GENERAL


def build_local_summary(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
) -> str:
    summary_candidates: list[str] = []
    for item in evidence[:6]:
        cleaned = _normalize_summary_text(item.quote)
        if not cleaned:
            continue
        if any(_is_similar_summary_text(cleaned, existing) for existing in summary_candidates):
            continue
        summary_candidates.append(cleaned)
    for article in articles[:6]:
        for raw in (article.content, article.title):
            cleaned = _normalize_summary_text(raw)
            if not cleaned:
                continue
            if any(_is_similar_summary_text(cleaned, existing) for existing in summary_candidates):
                continue
            summary_candidates.append(cleaned)
            break

    if summary_candidates:
        return summary_candidates[0][:170]

    return (
        f"{topic} 관련 보도가 {len(articles)}건 수집됐고 "
        f"{len({article.source for article in articles})}개 출처에서 확인됐습니다. "
        f"현재 신뢰도 점수는 {reliability.value}입니다."
    )


def _merge_distinct_texts(*groups: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for raw in group:
            cleaned = _normalize_summary_text(raw)
            if not cleaned:
                continue
            if any(_is_similar_summary_text(cleaned, existing) for existing in merged):
                continue
            merged.append(cleaned)
            if len(merged) >= limit:
                return merged
    return merged


def _choose_preferred_text(primary: str, fallback: str, article_titles: list[str] | None = None) -> str:
    primary_clean = _normalize_summary_text(primary)
    fallback_clean = _normalize_summary_text(fallback)
    titles = [_normalize_summary_text(title) for title in (article_titles or [])]
    if primary_clean:
        if len(primary_clean) >= 30:
            return primary_clean
        if titles and not any(_is_similar_summary_text(primary_clean, title) for title in titles):
            return primary_clean
    return fallback_clean or primary_clean


def _build_decision_details(
    effective_hold_reason: str | None,
    grounding: dict,
    grounded_summary: dict,
    claim_results: list[dict],
) -> dict:
    ready_claims = [item for item in claim_results if item.get("ready")]
    reasons: list[dict] = []
    if effective_hold_reason:
        reason_code = "hold_general"
        reason_label = "판정 보류"
        if "기사 수 부족" in effective_hold_reason:
            reason_code = "hold_article_count"
            reason_label = "기사 수 부족"
        elif "독립 출처 부족" in effective_hold_reason:
            reason_code = "hold_source_diversity"
            reason_label = "독립 출처 부족"
        elif "신뢰 가능한 핵심 출처" in effective_hold_reason:
            reason_code = "hold_trusted_source"
            reason_label = "핵심 출처 부족"
        elif "신뢰도 점수 부족" in effective_hold_reason:
            reason_code = "hold_reliability"
            reason_label = "신뢰도 부족"
        elif "grounding 검증 부족" in effective_hold_reason:
            reason_code = "hold_grounding"
            reason_label = "근거 결속 부족"
        reasons.append({"code": reason_code, "label": reason_label, "detail": effective_hold_reason})
        status = "HOLD"
        summary = f"{reason_label}로 인해 보류되었습니다."
    else:
        reasons.append(
            {
                "code": "ready_grounded",
                "label": "검증 통과",
                "detail": f"ready claim {len(ready_claims)}개, grounded_ratio={grounding.get('grounded_ratio', 0)}",
            }
        )
        status = "READY"
        summary = "근거 기반 검증을 통과했습니다."
    return {
        "status": status,
        "summary": summary,
        "ready_claim_count": len(ready_claims),
        "total_claim_count": len(claim_results),
        "grounded_claim_count": len(grounded_summary.get("grounded_claim_ids", [])),
        "reasons": reasons,
    }


def _should_use_remote_analysis(
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
) -> bool:
    if len(articles) >= 2 and len(evidence) >= 2:
        return True
    if len(articles) >= 3:
        return True
    return reliability.value >= 0.78 and len(evidence) >= 1


def _analysis_cache_key(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
) -> str:
    payload = "||".join(
        [
            topic,
            f"{reliability.value:.3f}",
            *[f"{article.id}:{article.title}:{article.published_at.isoformat()}" for article in articles[:8]],
            *[f"{item.article_id}:{item.source}:{item.quote[:120]}" for item in evidence[:6]],
        ]
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def _claim_cache_key(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    keywords: list[str],
) -> str:
    payload = "||".join(
        [
            topic,
            ",".join(keywords[:6]),
            *[f"{article.id}:{article.title}" for article in articles[:8]],
            *[f"{item.article_id}:{item.quote[:120]}" for item in evidence[:6]],
        ]
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def _normalize_summary_text(text: str | None) -> str:
    cleaned = re.sub(r"&nbsp;|\u00a0", " ", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s*-\s*(연합뉴스|한국경제|매일경제|조선일보|중앙일보|동아일보|Reuters|Bloomberg|CNBC|WSJ|v\.daum\.net|newsis\.com)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bView Full Coverage on Google News\b", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"([.!?。！？])\s+\1+", r"\1", cleaned)
    return cleaned[:220].strip()


def _normalize_compare_text(text: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]", "", _normalize_summary_text(text).lower())


def _is_similar_summary_text(left: str, right: str) -> bool:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    if shorter in longer:
        return (len(shorter) / max(len(longer), 1)) >= 0.88
    return False


def _build_issue_prompt(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
    include_claims: bool = False,
) -> str:
    article_lines = [
        f"- [{normalize_source_name(article.source)}] {article.title} | {article.published_at.isoformat()} | {article.url}"
        for article in articles[:8]
    ]
    evidence_lines = [
        f"- [{normalize_source_name(item.source)}] score={item.score}: {item.quote} ({item.url})"
        for item in evidence[:6]
    ]
    prompt = (
        f"이슈 주제: {topic}\n"
        f"신뢰도 점수: {reliability.value}\n"
        f"기사 목록:\n" + "\n".join(article_lines) + "\n"
        f"근거 스니펫:\n" + "\n".join(evidence_lines) + "\n"
        "요구사항:\n"
        "1. 근거 기반으로만 종합 요약\n"
        "2. 키워드 3~6개\n"
        "3. key_signals에는 속보성/시장성/정책성 핵심 시그널만 1~5개\n"
        "4. key_points에는 핵심 bullet 2~4개를 한국어로 제공\n"
        "5. trend_summary에는 추이를 한국어 한 문장으로 설명\n"
        "6. sentiment는 positive, neutral, negative, mixed 중 하나\n"
        "7. market_impact는 positive, neutral, negative, mixed 중 하나\n"
        "8. policy_risk와 volatility_risk는 low, medium, high 중 하나\n"
        "9. priority는 priority 또는 general\n"
        "10. 리스크 포인트 1~5개\n"
        "11. 근거 부족이면 grounded=false 또는 hold_reason 명시\n"
    )
    if include_claims:
        prompt += (
            "12. claims에는 서로 중복되지 않는 atomic claim 2~6개를 한국어로 제공\n"
            "13. 각 claim은 검증 가능한 사실 한 문장, 110자 이내, 제목 복붙 금지\n"
        )
    return prompt


def _build_hold_review_prompt(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
    analysis: AnalysisResult,
    grounding: dict,
    decision: dict,
    ready_claims: list[dict],
) -> str:
    article_lines = [
        f"- [{normalize_source_name(article.source)}] {article.title}"
        for article in articles[:5]
    ]
    evidence_lines = [
        f"- [{normalize_source_name(item.source)}] {item.quote[:180]}"
        for item in evidence[:5]
    ]
    claim_lines = [
        f"- claim={item.get('claim')} | support={item.get('support_count')} | trusted={item.get('trusted_support_count')} | external={item.get('external_support_count')} | score={item.get('score')}"
        for item in ready_claims[:4]
    ]
    return (
        f"이슈 주제: {topic}\n"
        f"현재 상태: HOLD\n"
        f"현재 hold_reason: {analysis.hold_reason}\n"
        f"현재 summary: {analysis.summary}\n"
        f"신뢰도 점수: {reliability.value}\n"
        f"grounded_ratio: {grounding.get('grounded_ratio')}\n"
        f"issue_score: {grounding.get('issue_score')}\n"
        f"decision: {decision}\n"
        f"기사 목록:\n" + "\n".join(article_lines) + "\n"
        f"근거 스니펫:\n" + "\n".join(evidence_lines) + "\n"
        f"검증 통과 claim:\n" + "\n".join(claim_lines) + "\n"
        "요구사항:\n"
        "1. production trust-first 기준으로 이 이슈를 READY로 승격할지 판단\n"
        "2. 근거가 아주 충분할 때만 promote_to_ready=true\n"
        "3. 애매하면 false\n"
        "4. rationale은 짧은 한국어 문장\n"
        "5. 승격 시 revised_summary는 보류 문구 없는 한국어 요약\n"
    )


def _build_claim_extraction_prompt(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    keywords: list[str],
) -> str:
    article_lines = [
        f"- [{normalize_source_name(article.source)}] {article.title} | {article.content[:220]}"
        for article in articles[:6]
    ]
    evidence_lines = [
        f"- [{normalize_source_name(item.source)}] {item.quote[:220]}"
        for item in evidence[:6]
    ]
    return (
        f"이슈 주제: {topic}\n"
        f"핵심 키워드: {', '.join(keywords[:6])}\n"
        "기사/근거:\n"
        + "\n".join(article_lines + evidence_lines)
        + "\n요구사항:\n"
        "1. 서로 중복되지 않는 atomic claim만 2~6개 추출\n"
        "2. 각 claim은 한 문장, 한국어, 110자 이내\n"
        "3. 기사 제목 복붙이 아니라 사실 단위로 정리\n"
        "4. 추정/전망/평가 대신 검증 가능한 사실만 작성\n"
    )


def _clean_claim(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^[\-\u2022]+\s*", "", cleaned)
    return cleaned[:180]


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", left)}
    right_tokens = {token.lower() for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", right)}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return round(overlap / max(len(left_tokens), 1), 3)


def _extract_entities(text: str) -> set[str]:
    tokens = re.findall(r"[A-Z][A-Za-z&.-]{1,}|[가-힣]{2,}|\d+(?:\.\d+)?%?", text)
    return {token.lower() for token in tokens if len(token) >= 2}


def _matched_entities(left: str, right: str) -> list[str]:
    matched = sorted(_extract_entities(left) & _extract_entities(right))
    return matched[:8]


def _entity_match_score(left: str, right: str) -> float:
    left_entities = _extract_entities(left)
    right_entities = _extract_entities(right)
    if not left_entities:
        return 0.0
    return round(len(left_entities & right_entities) / len(left_entities), 3)


def _extract_numbers(text: str) -> set[str]:
    return {token for token in re.findall(r"\d+(?:\.\d+)?%?", text)}


def _number_match_score(left: str, right: str) -> float:
    left_numbers = _extract_numbers(left)
    if not left_numbers:
        return 1.0
    right_numbers = _extract_numbers(right)
    return round(len(left_numbers & right_numbers) / len(left_numbers), 3)


def _heuristic_verification_label(
    *,
    heuristic_score: float,
    overlap: float,
    contradiction_mode: bool,
    quote: str,
) -> str:
    lowered = quote.lower()
    if contradiction_mode:
        if overlap >= 0.18 and any(token in lowered for token in CONTRADICTION_HINTS):
            return "contradict"
        return "insufficient"
    if heuristic_score >= 0.72:
        return "support"
    if heuristic_score >= 0.42 or overlap >= 0.2:
        return "partial_support"
    return "insufficient"
