# Reliability-First News AI Agent

국내외 뉴스를 수집하고, 유사 기사들을 하나의 이슈로 묶은 뒤, RAG 기반 근거 검증을 거쳐 웹 대시보드와 Slack으로 보고하는 신뢰성 중심 AI Agent MVP입니다.

## Design Goal

이 프로젝트의 핵심은 "많이 요약하는 것"이 아니라 "근거가 있을 때만 말하는 것"입니다.

시스템은 아래 원칙을 강제합니다.

- 단일 기사 요약보다 이슈 단위 분석 우선
- 다중 출처 검증 우선
- 출처 신뢰도 기반 필터링 적용
- 근거 부족 시 응답 거절 또는 보류
- LLM 호출은 백엔드에서만 수행

## Target Stack

- Backend: `FastAPI`
- Scheduler: `APScheduler`
- Collection: `feedparser`, `requests`, `BeautifulSoup`
- Storage: `SQLite`
- Retrieval layer: `Chroma-compatible abstraction`
- Delivery: `Slack webhook`

## Architecture

```text
[RSS / Public APIs]
        |
        v
[Collector]
  - RSS polling
  - API fetch
  - selective crawling
        |
        v
[Preprocessor]
  - dedupe
  - text cleanup
  - low-quality filtering
  - metadata persistence
        |
        v
[Issue Clustering]
  - title/body similarity
  - issue grouping
        |
        v
[RAG Reliability Layer]
  - retrieval
  - reranking
  - source reliability filter
  - grounding
  - multi-source verification
        |
        +--> low evidence => HOLD / FALLBACK
        |
        v
[LLM Analyzer]
  - summary
  - keywords
  - sentiment
  - risk points
  - strict JSON
        |
        v
[SQLite]
        |
        +--> [Dashboard]
        |
        +--> [Slack Reporter]
```

상세 설계 문서는 [docs/ARCHITECTURE.md](/Users/yejin/Documents/New%20project/docs/ARCHITECTURE.md:1)에 있습니다.

## Reliability Policy

- 최소 2개 이상의 독립 출처가 있어야 함
- 중요한 주장마다 evidence snippet이 있어야 함
- 출처 신뢰도 가중치가 낮은 기사만으로는 `READY` 불가
- 근거 부족 시 요약 대신 `HOLD` 사유 반환
- 프론트엔드는 저장된 결과만 조회하고, 직접 LLM API를 호출하지 않음

## Scheduler Policy

- 수집 작업: 10분 주기
- 분석 작업: 1시간 주기
- 필요 시 수동 실행 API 제공

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

브라우저:

- Dashboard: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`

## Main APIs

- `POST /api/jobs/collect`: 수집 + 전처리 실행
- `POST /api/jobs/analyze`: 이슈 분석 실행
- `POST /api/pipeline/run`: 수집부터 분석까지 전체 실행
- `GET /api/issues`: 이슈 목록 조회
- `POST /api/issues/{issue_id}/report`: Slack 전송 시도

## Environment Variables

- `SLACK_WEBHOOK_URL`
- `ENABLE_SCHEDULER=true|false`
- `COLLECT_INTERVAL_MINUTES`
- `ANALYZE_INTERVAL_MINUTES`

## Notes

- 현재는 샘플 데이터와 로컬 휴리스틱 기반으로 동작합니다.
- RAG 계층은 Chroma/FAISS로 교체 가능한 인터페이스로 설계했습니다.
- OpenAI 같은 LLM API 연동은 `app/services/llm_analyzer.py` 백엔드 계층에서만 추가하면 됩니다.
