# 📑 GroundWire
### **Reliability-based News Issue Analysis Dashboard**

GroundWire는 실시간 뉴스 스트림과 외부 reference corpus를 함께 활용해, 검증 가능한 이슈만 READY로 승격하는 **신뢰도 중심 이슈 모니터링 시스템**입니다.

뉴스 기사 간 교차 확인, 외부 문서 기반 근거 검색, 상충 근거 탐지, reference-aware scoring을 통해 사람이 검토할 가치가 있는 이슈만 선별합니다.

## 📊 Demo

### Main Dashboard Overview
실시간 뉴스 수집 상태와 핵심 이슈 요약을 한눈에 확인

![main](./assets/main.png)



### Core Issue & Market Summary
이슈 단위로 요약된 뉴스와 시장 영향 분석 제공

<p align="center">
  <img src="./assets/summary.png" width="75%">
</p>


### Issue Board & Detailed Analysis
이슈별 신뢰도, 근거, 검증 상태를 기반으로 분석 결과 제공

![detail](./assets/detail.png)



### Keyword Trends & Signals
실시간 키워드 흐름과 뉴스 신호 변화 분석

<p align="center">
  <img src="./assets/keywords.png" width="40%">
</p>  


## 🛠 기술 스택

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--5.4--mini-412991?logo=openai&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Storage-003B57?logo=sqlite&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-Templates-B41717?logo=jinja&logoColor=white)
![APScheduler](https://img.shields.io/badge/APScheduler-Scheduler-4B5563)  


## 🚀 핵심 기능

- 실시간 RSS 기반 뉴스 수집 및 전처리
- 유사 기사 클러스터링을 통한 이슈 분석
- 뉴스 코퍼스와 reference corpus를 분리한 하이브리드 retrieval
- 외부 문서 기반 근거 검색 및 evidence provenance 추적
- reference-aware scoring을 통한 신뢰도 계산
- 정정, 반박, 철회, 해명성 문서를 반영하는 contradiction handling
- `READY` / `HOLD` 판정
  

## 🔄 데이터 플로우

뉴스 수집 → 기사 정제 및 클러스터링 → reference source 동기화 → claim 추출 

→ 뉴스 근거 검색 → reference 근거 검색 → 반박/정정 근거 검색 

→ reference-aware scoring → READY / HOLD 판정 

→ 대시보드 반영


## ⚙️ 실행 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd "New project"
```
### 2. 가상환경 생성
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3. 의존성 설치
```bash
pip install -r requirements.txt
```
### 4. 환경 변수 설정
기본 실행 설정은 `.env`에 작성합니다.
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=model_here
EMBEDDING_MODEL=embedding_model_here
```
### 5. threshold 변수 설정
```python
PRESENTATION_MODE=true
HOLD_THRESHOLD=0.55
MIN_ARTICLES_PER_ISSUE=1
MIN_UNIQUE_SOURCES=1
REQUIRE_TRUSTED_READY_SOURCE=false
MIN_GROUNDED_RATIO=0
MIN_GROUNDING_ISSUE_SCORE=0.4
MIN_GROUNDED_CLAIMS=0
```
- `.env.local` 파일은 선택 사항으로, 시연 또는 테스트 시 threshold와 판정 기준을 로컬 환경에서 임시로 조정할 때 사용합니다.

### 6. 실행
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
