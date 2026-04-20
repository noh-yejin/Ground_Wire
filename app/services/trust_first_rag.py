from __future__ import annotations

"""
Trust-first RAG scaffold.

This file is intentionally not wired into the runtime.
It is a concrete replacement blueprint for the following original files:

- app/services/rag.py
- app/services/reliability.py
- app/services/llm_analyzer.py
- app/services/pipeline.py

The goal is not "summarize fast", but "only summarize what can be justified".
"""

from dataclasses import dataclass, field
from enum import Enum

from app.models import Article, EvidenceSnippet, ReliabilityScore, RiskLevel
from app.services.rag import EvidenceRetriever
from app.services.source_normalizer import is_trusted_ready_source


class ClaimType(str, Enum):
    FACT = "fact"
    INTERPRETATION = "interpretation"
    MARKET_IMPACT = "market_impact"
    POLICY = "policy"


class VerificationLabel(str, Enum):
    SUPPORT = "support"
    PARTIAL_SUPPORT = "partial_support"
    CONTRADICT = "contradict"
    INSUFFICIENT = "insufficient"


@dataclass(slots=True)
class Claim:
    id: str
    text: str
    claim_type: ClaimType
    importance: float
    source_article_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VerifiedEvidence:
    claim_id: str
    article_id: str
    source: str
    url: str
    quote: str
    retrieval_score: float
    verification_label: VerificationLabel
    verification_score: float
    trusted_source: bool


@dataclass(slots=True)
class ClaimVerificationResult:
    claim: Claim
    evidence: list[VerifiedEvidence]
    support_count: int
    contradiction_count: int
    trusted_support_count: int
    grounded_score: float
    ready: bool


@dataclass(slots=True)
class GroundedSentence:
    text: str
    claim_ids: list[str]


@dataclass(slots=True)
class GroundedAnalysis:
    summary_sentences: list[GroundedSentence]
    key_points: list[str]
    omitted_claims: list[str]
    issue_score: float
    policy_risk: RiskLevel
    reasons: list[str]


class TrustFirstClaimExtractor:
    """Scaffold for replacing the current direct-summary flow.

    In production, this should call an LLM and require short, atomic,
    evidence-checkable claims. For now it is intentionally conservative.
    """

    def extract(self, topic: str, articles: list[Article]) -> list[Claim]:
        claims: list[Claim] = []
        for index, article in enumerate(articles[:5], start=1):
            sentence = article.content.split(".")[0].strip()
            if not sentence:
                continue
            claims.append(
                Claim(
                    id=f"claim-{index}",
                    text=sentence[:180],
                    claim_type=ClaimType.FACT,
                    importance=max(0.2, 1 - ((index - 1) * 0.12)),
                    source_article_ids=[article.id],
                )
            )
        return claims


class TrustFirstVerifier:
    """Claim verifier placeholder.

    In a full implementation this should:
    1. retrieve support evidence
    2. retrieve contradiction evidence
    3. run claim/evidence verification
    4. score grounding strength
    """

    def __init__(self, retriever: EvidenceRetriever | None = None) -> None:
        self.retriever = retriever or EvidenceRetriever()

    def verify(self, claim: Claim, articles: list[Article]) -> ClaimVerificationResult:
        retrieved = self.retriever.retrieve(articles)
        verified: list[VerifiedEvidence] = []
        support_count = 0
        contradiction_count = 0
        trusted_support_count = 0

        for item in retrieved:
            label, score = self._classify(claim.text, item)
            trusted = is_trusted_ready_source(item.source)
            verified_item = VerifiedEvidence(
                claim_id=claim.id,
                article_id=item.article_id,
                source=item.source,
                url=item.url,
                quote=item.quote,
                retrieval_score=item.score,
                verification_label=label,
                verification_score=score,
                trusted_source=trusted,
            )
            verified.append(verified_item)
            if label in {VerificationLabel.SUPPORT, VerificationLabel.PARTIAL_SUPPORT}:
                support_count += 1
                if trusted:
                    trusted_support_count += 1
            if label == VerificationLabel.CONTRADICT:
                contradiction_count += 1

        grounded_score = self._score_claim(
            support_count=support_count,
            trusted_support_count=trusted_support_count,
            contradiction_count=contradiction_count,
            evidence=verified,
            importance=claim.importance,
        )

        return ClaimVerificationResult(
            claim=claim,
            evidence=verified,
            support_count=support_count,
            contradiction_count=contradiction_count,
            trusted_support_count=trusted_support_count,
            grounded_score=grounded_score,
            ready=grounded_score >= 0.7,
        )

    def _classify(self, claim_text: str, evidence: EvidenceSnippet) -> tuple[VerificationLabel, float]:
        """Very conservative placeholder verifier.

        This is intentionally simple so it can live beside the current codebase
        without changing runtime behavior. A production version should replace
        this with entailment or LLM-based claim verification.
        """

        claim_tokens = {token.lower() for token in claim_text.split() if len(token) >= 3}
        quote_tokens = {token.lower() for token in evidence.quote.split() if len(token) >= 3}
        overlap = len(claim_tokens & quote_tokens)
        if overlap >= 4:
            return VerificationLabel.SUPPORT, min(0.99, 0.55 + overlap * 0.08)
        if overlap >= 2:
            return VerificationLabel.PARTIAL_SUPPORT, 0.55
        return VerificationLabel.INSUFFICIENT, 0.2

    def _score_claim(
        self,
        *,
        support_count: int,
        trusted_support_count: int,
        contradiction_count: int,
        evidence: list[VerifiedEvidence],
        importance: float,
    ) -> float:
        support_strength = min(support_count / 3, 1.0)
        trusted_strength = min(trusted_support_count / 2, 1.0)
        avg_verification = (
            sum(item.verification_score for item in evidence if item.verification_label != VerificationLabel.INSUFFICIENT)
            / max(len([item for item in evidence if item.verification_label != VerificationLabel.INSUFFICIENT]), 1)
        )
        contradiction_penalty = min(contradiction_count * 0.25, 0.6)
        score = (
            support_strength * 0.35
            + trusted_strength * 0.30
            + avg_verification * 0.20
            + importance * 0.15
            - contradiction_penalty
        )
        return round(max(0.0, min(score, 1.0)), 3)


class TrustFirstIssueScorer:
    """Replacement direction for app/services/reliability.py."""

    def score_issue(
        self,
        results: list[ClaimVerificationResult],
        base_reliability: ReliabilityScore,
    ) -> tuple[float, list[str]]:
        if not results:
            return 0.0, ["검증 가능한 claim이 없습니다."]

        ready_claims = [result for result in results if result.ready]
        contradiction_claims = [result for result in results if result.contradiction_count > 0]
        grounded_ratio = len(ready_claims) / max(len(results), 1)
        contradiction_penalty = min(len(contradiction_claims) * 0.12, 0.36)
        avg_claim_score = sum(result.grounded_score for result in results) / len(results)

        issue_score = (
            avg_claim_score * 0.55
            + grounded_ratio * 0.25
            + base_reliability.value * 0.20
            - contradiction_penalty
        )

        reasons: list[str] = []
        if grounded_ratio < 0.5:
            reasons.append("검증 통과 claim 비율이 낮습니다.")
        if contradiction_claims:
            reasons.append("상충 근거가 감지된 claim이 있습니다.")
        if base_reliability.value < 0.7:
            reasons.append("기본 출처/최신성 신뢰도도 충분히 높지 않습니다.")

        return round(max(0.0, min(issue_score, 1.0)), 3), reasons


class TrustFirstSummaryBuilder:
    """Summary builder that only uses verified claims."""

    def build(self, results: list[ClaimVerificationResult], issue_score: float, reasons: list[str]) -> GroundedAnalysis:
        ready_claims = [result for result in results if result.ready]
        ordered = sorted(ready_claims, key=lambda item: item.grounded_score, reverse=True)
        summary_sentences: list[GroundedSentence] = []
        key_points: list[str] = []

        for result in ordered[:3]:
            summary_sentences.append(
                GroundedSentence(
                    text=result.claim.text,
                    claim_ids=[result.claim.id],
                )
            )
            key_points.append(result.claim.text)

        omitted = [result.claim.text for result in results if not result.ready]
        policy_risk = RiskLevel.HIGH if any(result.contradiction_count > 0 for result in results) else RiskLevel.MEDIUM

        return GroundedAnalysis(
            summary_sentences=summary_sentences,
            key_points=key_points[:4],
            omitted_claims=omitted[:5],
            issue_score=issue_score,
            policy_risk=policy_risk,
            reasons=reasons,
        )


class TrustFirstIssueAnalyzer:
    """High-level scaffold for the replacement flow.

    Intended replacement direction for:
    - LLMAnalyzer.analyze()
    - NewsPipeline.analyze_only()
    """

    def __init__(self) -> None:
        self.claim_extractor = TrustFirstClaimExtractor()
        self.verifier = TrustFirstVerifier()
        self.issue_scorer = TrustFirstIssueScorer()
        self.summary_builder = TrustFirstSummaryBuilder()

    def analyze(
        self,
        topic: str,
        articles: list[Article],
        base_reliability: ReliabilityScore,
    ) -> tuple[GroundedAnalysis, list[ClaimVerificationResult]]:
        claims = self.claim_extractor.extract(topic, articles)
        verification_results = [self.verifier.verify(claim, articles) for claim in claims]
        issue_score, reasons = self.issue_scorer.score_issue(verification_results, base_reliability)
        analysis = self.summary_builder.build(verification_results, issue_score, reasons)
        return analysis, verification_results
