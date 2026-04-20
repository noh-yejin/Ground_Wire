# Trust-First RAG Guide

이 문서는 현재 프로젝트의 RAG/LLM 분석 구조를 "신뢰 최우선" 방향으로 바꾸기 위한 설계 문서다.
원본 런타임 코드는 수정하지 않고, 어떤 파일을 어떻게 바꿔야 하는지와 왜 바꿔야 하는지를 정리한다.

## 대응되는 원본 코드

- `app/services/rag.py`
  - 현재: 입력 기사 내부에서 유사 스니펫을 재검색
  - 개선: claim 단위 retrieval + 다출처 근거 묶음 생성 + 반박 근거 탐색
- `app/services/reliability.py`
  - 현재: 출처 수/최신성/근거 수 중심 휴리스틱 점수
  - 개선: claim grounding 비율, contradiction 비율, 원문 출처 신뢰, evidence freshness 기반 점수
- `app/services/llm_analyzer.py`
  - 현재: structured output은 만들지만 claim-level 검증 없음
  - 개선: "주장 추출 -> 근거 검증 -> 검증 통과 claim만 요약" 구조로 변경
- `app/services/pipeline.py`
  - 현재: cluster -> evidence -> reliability -> analysis
  - 개선: cluster -> claim set -> external/internal retrieval -> verifier -> grounded summary -> trust gate
- `app/repository.py`
  - 현재: 최종 analysis payload만 저장
  - 개선: claim, evidence bundle, verification result, contradiction result를 같이 저장

## 현재 구조의 핵심 문제

1. RAG가 외부 검증보다 "입력 기사 내부 재검색"에 가깝다.
2. 요약문이 evidence에 의해 사후 검증되지 않는다.
3. 서로 다른 출처가 같은 사실을 말하는지 보지 않고, 출처 개수만으로 교차 검증을 추정한다.
4. contradiction(상충 보도) 처리가 없다.
5. summary/key_points가 "근거 있음"인지 "그럴듯한 생성"인지 구분되지 않는다.

## 목표 상태

"이 이슈 요약은 믿을 수 있는가?"라는 질문에 대해 아래를 남기는 구조가 되어야 한다.

- 어떤 claim이 핵심 주장인지
- 각 claim을 뒷받침하는 근거가 몇 개인지
- 근거가 서로 독립적인 출처인지
- 반박 또는 상충 근거가 있는지
- 최종 요약문 각 문장이 어떤 claim/evidence에서 왔는지
- 검증이 약한 문장은 요약에서 제외되었는지

## 신뢰 최우선 파이프라인

### 1. Cluster Stage

입력 기사들을 현재처럼 이슈 단위로 묶는다.

- 유지 가능: `app/services/pipeline.py`
- 유지 가능: `app/services/clustering.py`

이 단계는 크게 문제가 없다.

### 2. Claim Extraction Stage

이슈 전체를 바로 요약하지 말고, 먼저 "검증 가능한 짧은 주장(claim)"들로 분해한다.

예시:

- "미국의 대중 반도체 규제가 확대됐다."
- "삼성전자와 SK하이닉스가 HBM 수혜 기대를 받고 있다."
- "이 사안은 한국 반도체 수출에 하방 리스크로 작용할 수 있다."

좋은 claim 조건:

- 하나의 사실 또는 하나의 해석만 담는다.
- 길이가 짧고, evidence로 뒷받침 가능해야 한다.
- 과장된 인과관계는 분리한다.

추천 구현:

- LLM으로 3~8개 claim 추출
- claim마다 `claim_type` 부여
  - `fact`
  - `interpretation`
  - `market_impact`
  - `policy`

### 3. Retrieval Stage

claim마다 근거를 다시 찾는다. 이 단계는 "이슈 전체 제목"이 아니라 "claim 문장"을 질의로 써야 한다.

retrieval 대상은 두 층으로 분리하는 것이 좋다.

- Layer A: 현재 이슈에 포함된 기사 본문
- Layer B: 외부 검증용 원문 소스
  - 공식 발표문
  - 정부/기관 문서
  - 공시/IR
  - wire/신뢰 가능한 원문 매체

최소 요구:

- claim마다 supporting evidence 2개 이상
- 가능하면 source normalization 후 독립 출처 2개 이상
- claim마다 contradiction evidence도 별도로 탐색

### 4. Verification Stage

retrieved evidence를 그냥 점수순으로 쓰지 말고, claim과의 관계를 판정한다.

관계 타입 예시:

- `support`
- `partial_support`
- `contradict`
- `insufficient`

검증은 두 단계가 좋다.

1. lexical / heuristic filter
   - 날짜/주체/숫자/지역 키워드 일치 여부
2. LLM verifier 또는 entailment model
   - "이 evidence가 claim을 지지하는가?"
   - "claim의 핵심 요소가 evidence 안에 실제로 존재하는가?"

### 5. Claim Scoring Stage

claim별 신뢰 점수를 별도로 계산한다.

추천 요소:

- source diversity
- trusted source ratio
- evidence freshness
- support strength
- contradiction penalty
- numeric/entity match score
- direct quote ratio

예시:

`claim_score = 0.25 * diversity + 0.20 * trust + 0.20 * freshness + 0.20 * support_strength + 0.15 * entity_match - contradiction_penalty`

### 6. Summary Synthesis Stage

최종 summary/key_points는 전체 기사에서 바로 생성하지 말고, "검증 통과한 claim만" 재료로 만들어야 한다.

규칙:

- `claim_score`가 기준 미만이면 summary에서 제외
- contradiction이 강한 claim은 "논쟁 중"으로만 표기
- impact/policy/risk 문장도 claim 검증 결과를 기반으로 생성

좋은 출력 형식:

- 요약문 2~4문장
- 각 문장마다 `claim_ids` 연결
- 핵심 bullet 2~4개
- "확실한 사실"과 "추정/해석"을 구분

### 7. Trust Gate Stage

이슈 전체를 READY/HOLD로 나누는 기준도 claim 기반으로 바꿔야 한다.

예시:

- READY
  - verified claim 2개 이상
  - trusted source 2개 이상
  - contradiction ratio 낮음
  - summary 문장의 100%가 verified claim 기반
- HOLD
  - 주요 claim이 1개 이하만 verified
  - contradiction 강함
  - impact claim만 있고 사실 claim이 약함
  - 공식/독립 출처 부족

## 추천 데이터 구조

기존 `AnalysisResult`만으로는 부족하므로 아래 저장 구조를 추가하는 것이 좋다.

### Claim

- `id`
- `text`
- `claim_type`
- `importance`
- `source_article_ids`

### VerifiedEvidence

- `claim_id`
- `article_id`
- `source`
- `url`
- `quote`
- `retrieval_score`
- `verification_label`
- `verification_score`

### ClaimVerificationResult

- `claim_id`
- `support_count`
- `trusted_support_count`
- `contradiction_count`
- `freshest_evidence_at`
- `claim_score`
- `ready`

### GroundedSummary

- `summary_sentences`
- `sentence_claim_links`
- `key_points`
- `omitted_claims`

## 구현 우선순위

### 1단계

- claim extraction 추가
- claim별 evidence retrieval 추가
- contradiction 탐색 추가

### 2단계

- LLM verifier 추가
- summary는 verified claim만으로 생성
- READY/HOLD를 claim score 기반으로 변경

### 3단계

- repository에 claim/evidence/verification 결과 저장
- UI에 "왜 신뢰 가능한지" 설명 가능한 메타데이터 노출

## 실제 코드 변경 방향

### `app/services/rag.py` → `app/services/trust_first_rag.py`

필수 변경:

- retrieve query를 `이슈 제목`이 아니라 `claim text` 기준으로 변경
- support evidence와 contradiction evidence를 분리
- evidence object에 verification label 저장

### `app/services/reliability.py`

필수 변경:

- 이슈 전체 1개 점수 대신
  - claim별 점수
  - 이슈 종합 점수
  - contradiction penalty
  - grounded sentence ratio
  로 확장

### `app/services/llm_analyzer.py`

필수 변경:

- `summary`를 바로 생성하지 말고
  - claim extraction
  - claim verification
  - grounded summary synthesis
  3단계로 분리

### `app/services/pipeline.py`

필수 변경:

- 기존 `evidence = retriever.retrieve(group)` 한 줄 구조를 해체
- claim loop를 도입
- trust gate를 마지막에 수행

## 추천 운영 원칙

- 고신뢰가 목적이면 "문장을 덜 보여주더라도 검증된 문장만 보여준다"
- unsupported claim은 요약에 쓰지 않는다
- contradiction이 있는 이슈는 "상충 보도 존재"를 명시한다
- market impact/policy risk 같은 해석형 라벨은 fact claim보다 낮은 신뢰 계층으로 취급한다

## 이 문서와 함께 만든 코드 파일

- `app/services/trust_first_rag.py`

이 파일은 원본 런타임에 연결되지 않은 스캐폴드다.
실제로 바꾸려면 위 원본 파일들을 이 설계 기준으로 단계적으로 교체하면 된다.
