# System Architecture

## 1. Scope

이 시스템은 뉴스 기사 자체를 대량 요약하는 서비스가 아니라, 복수의 기사들을 하나의 "이슈"로 묶고 신뢰성 기준을 통과한 경우에만 결과를 제공하는 AI Agent를 목표로 합니다.

MVP 범위:

- 로컬 실행 가능
- RSS 및 공개 API 우선
- SQLite 기반 저장
- FastAPI 기반 웹/백엔드
- APScheduler 기반 자동화
- Slack webhook 기반 알림

## 2. End-to-End Flow

```text
Collect
  -> Preprocess
  -> Store raw article metadata
  -> Cluster by issue
  -> Retrieve evidence candidates
  -> Rerank with source reliability
  -> Verify multi-source support
  -> LLM JSON analysis
  -> Save issue snapshot
  -> Dashboard / Slack
```

## 3. Module Design

### Collector

책임:

- RSS 및 공개 API에서 기사 후보 수집
- 제목, URL, 발행 시각, 출처, 요약 본문 확보
- 중요한 기사만 선택적으로 본문 크롤링

정책:

- 기본은 feed summary 기반 수집
- `정책 키워드`, `시장 영향`, `국제 분쟁`, `긴급 속보` 같은 high-priority 조건일 때만 본문 크롤링
- 전체 기사 full crawl은 하지 않음

### Preprocessor

책임:

- URL/title hash 기반 중복 제거
- HTML 제거 및 본문 정리
- 짧은 기사, 광고성 기사, 메타데이터 부족 기사 제거
- 정규화된 기사 레코드를 SQLite에 저장

저장 필드:

- `source`
- `published_at`
- `url`
- `title`
- `content`
- `language`
- `collected_at`

### Issue Clustering

책임:

- 기사 단위가 아니라 이슈 단위로 묶기
- 제목/본문 기반 유사도 활용

MVP 구현:

- 토큰 기반 유사도와 overlap heuristic

확장 포인트:

- sentence-transformers 임베딩
- FAISS 또는 Chroma 인덱스 기반 nearest-neighbor clustering

### RAG Reliability Layer

책임:

- Retrieval: 이슈 대표 기사 기준 evidence 후보 탐색
- Reranking: 출처 신뢰도와 텍스트 유사도를 함께 반영
- Source filtering: 저신뢰 출처만 존재하는 경우 보류
- Grounding: evidence snippet 없이 요약 금지
- Multi-source verification: 최소 2개 독립 출처 확인

핵심 정책:

- evidence가 없는 주장 생성 금지
- 근거 부족 시 `HOLD`
- 상충 출처가 큰 경우 리스크로 명시

### LLM Analyzer

책임:

- 이슈 종합 요약
- 키워드 추출
- 감정 분석
- 리스크 포인트 정리
- JSON 구조 출력

백엔드 제약:

- LLM API는 반드시 백엔드에서만 호출
- 프론트엔드는 저장된 JSON 결과만 조회

추천 출력 스키마:

```json
{
  "summary": "string",
  "keywords": ["string"],
  "sentiment": "positive|neutral|negative|mixed",
  "risk_points": ["string"],
  "grounded": true,
  "hold_reason": null
}
```

### Dashboard

표시 요소:

- 주요 이슈 리스트
- 요약
- 출처 수
- 신뢰도 점수
- 감정 분석 결과
- 원문 링크

### Scheduler

작업 분리:

- `collect_news_job`: 10분 주기
- `analyze_issues_job`: 1시간 주기

이유:

- 수집은 자주, 분석은 상대적으로 덜 자주 실행해 비용과 안정성을 관리

## 4. SQLite Schema

### `articles`

- `id`
- `title`
- `source`
- `published_at`
- `url`
- `content`
- `language`
- `collected_at`
- `content_quality`

### `issues`

- `id`
- `topic`
- `status`
- `reliability`
- `payload`
- `updated_at`

### `job_runs`

- `id`
- `job_name`
- `status`
- `details`
- `created_at`

## 5. Why This Design Fits MVP

- 로컬 SQLite만으로 실행 가능
- 수집과 분석이 분리되어 장애 영향이 작음
- RAG 계층이 명시되어 hallucination 억제 논리를 설명하기 쉬움
- 프론트가 LLM을 직접 호출하지 않아 보안/비용/정책상 안전
- 이후 Chroma/FAISS, OpenAI API, 실제 뉴스 API를 자연스럽게 붙일 수 있음
