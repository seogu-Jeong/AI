# Phase 3 — AI 예측 + 시그널 + 패턴 인식 설계

**작성일:** 2026-06-03
**담당:** hygrenn (백엔드 + ML)
**레퍼런스:** TRD 섹션 6, 14 / PRD Phase 3

---

## 1. 결정 사항 요약

| 항목 | 결정 |
|---|---|
| LSTM 학습 방식 | 오프라인 사전 학습 (코스피 상위 100종목) |
| 미학습 종목 처리 | 기술적 지표 100% fallback (`lstm_available: false`) |
| DB 마이그레이션 | Phase 3+4 전체 테이블 한 번에 (Alembic 버전 3~6) |
| 아키텍처 | ML 레이어 완전 분리 (`ml/` 독립 모듈) |

---

## 2. ML 파이프라인 (`backend/ml/`)

### 2.1 디렉토리 구조

```
backend/ml/
├── __init__.py
├── features.py         — OHLCV → 13개 피처 (pandas-ta)
├── model.py            — StockLSTM 클래스
├── train.py            — 오프라인 학습 스크립트
├── predict.py          — 추론 인터페이스
├── pattern_matcher.py  — 유사 패턴 Top 5 히스토리 검색
└── weights/            — {code}.pth 저장소
```

### 2.2 데이터 흐름

```
pykrx OHLCV (2년치 일봉, ~500 거래일)
  → features.py
      - RSI(14), MACD/Signal/Hist, BB(Upper/Mid/Lower)
      - MA5, MA20, MA60, Vol_MA5, Stoch_K
      - 총 13개 피처 (OHLCV 포함)
      - MinMaxScaler (피처별 독립 스케일)
      - dropna() 후 학습 가능 샘플만
  → train.py
      - Train/Val/Test: 70% / 15% / 15%
      - Huber Loss + AdamW (lr=0.001, weight_decay=1e-5)
      - CosineAnnealingLR
      - Early Stopping: Val Loss 10 epoch 개선 없으면 중단
      - 저장: ml/weights/{code}.pth (모델 state_dict + scaler)
```

### 2.3 StockLSTM 모델 (TRD 스펙)

```
입력: (batch, seq_len=60, features=13)
출력: (batch, 5) — 다음 5 거래일 종가 변화율

구조:
  LSTM(input=13, hidden=128, layers=2, dropout=0.2)
  → MultiheadAttention(embed=128, heads=4) on last 10 steps
  → Linear(128→64) + ReLU + Dropout(0.2)
  → Linear(64→5)
```

### 2.4 추론 (Monte Carlo Dropout)

```
model.train() 모드로 50회 순전파
→ predictions: (50, 5) 변화율 배열
→ base:    median (50th percentile)
→ bullish: 75th percentile
→ bearish: 25th percentile
→ 변화율 → 실제 가격 역변환 (current_price * (1 + rate))
→ confidence: 최근 20회 방향 정확도 (%)
```

### 2.5 train.py 실행 방법

```bash
# 단일 종목
python -m ml.train --codes 005930

# 다수 종목 (상위 100종목 리스트 파일)
python -m ml.train --codes-file ml/top100_codes.txt

# Google Colab에서도 동일 스크립트 사용 가능
```

### 2.6 유사 패턴 매칭 (pattern_matcher.py)

- 최근 20일 종가 정규화 벡터를 히스토리와 cosine similarity 비교
- Top 5 유사 구간 반환: 날짜, 유사도, 이후 5일 실제 수익률

---

## 3. 서비스 레이어

### 3.1 ai_service.py

**역할:** ML 추론 + 기술적 지표를 조율해 최종 시그널 산출

```
입력: stock_code
처리:
  1. pykrx/캐시에서 OHLCV 로드
  2. features.py로 지표 계산
  3. weights/{code}.pth 존재 확인
     - 있음: predict.py 호출 → lstm_direction (-1~1)
     - 없음: lstm_available=False, 기술적 지표 100% 사용
  4. 시그널 계산:
     tech_score  = RSI(40%) + MACD(35%) + BB(25%)
     lstm_score  = 50 + lstm_direction * 50
     final_score = tech_score * 0.4 + lstm_score * 0.6  (or 100% tech if no model)
     BUY ≥ 65, SELL ≤ 35, else HOLD
  5. ai_signals_history 테이블에 저장
  6. Redis 캐시 (TTL: 장중 300초, 장외 24시간)
출력: {signal, score, tech_score, lstm_score, lstm_available, indicators}
```

### 3.2 pattern_service.py

**역할:** pandas-ta로 캔들 패턴 감지

```
감지 대상 14종:
  hammer, invertedhammer, doji, engulfing,
  morningstar, eveningstar, shootingstar, hangingman,
  threewhitesoldiers, threeblackcrows, piercingpattern,
  darkcloudcover, harami, haramicross

반환: [{name, display_name, direction, value}]
  direction: "bullish" | "bearish"
  value: 100 | -100 (pandas-ta 표준)
```

### 3.3 Celery AI 태스크 (`backend/tasks/ai_tasks.py`)

```python
# 기존 backend/tasks.py stub → backend/tasks/ 폴더로 구조 변경
tasks/
├── __init__.py   — Celery 앱 초기화
├── ai_tasks.py   — AI 시그널 갱신
└── email_tasks.py (Phase 4)

# ai_tasks.py
@celery.task
def refresh_ai_signals():
    """장 종료 후(15:35) watchlist + portfolio 종목 시그널 갱신"""
    → 유저별 관심/보유 종목 수집 (현재는 watchlist 없으므로 학습된 100종목 전체)
    → ai_service.calculate_signal(code) 호출
    → 시그널 변경 시 ai_signals_history 저장
```

**APScheduler (main.py에 추가):**
```python
scheduler.add_job(refresh_ai_signals, 'cron', hour=15, minute=35, day_of_week='mon-fri')
```

---

## 4. API 엔드포인트 (`backend/api/routes/ai.py`)

Rate limit: 20/min (slowapi)

| Method | Path | 설명 | 캐시 TTL |
|---|---|---|---|
| GET | /ai/{code}/signal | 종합 시그널 (BUY/HOLD/SELL + 점수 분해) | 장중 300초, 장외 24h |
| GET | /ai/{code}/predict | LSTM 5일 예측 (bullish/base/bearish) | 장중 300초, 장외 24h |
| GET | /ai/{code}/indicators | RSI, MACD, BB, MA 원시값 | 장중 300초, 장외 24h |
| GET | /ai/{code}/patterns | 감지된 캔들 패턴 목록 | 장중 300초, 장외 24h |
| GET | /ai/{code}/similar | 유사 패턴 Top 5 히스토리 | 장중 300초, 장외 24h |
| GET | /ai/{code}/multiframe | 일봉/주봉/월봉 멀티타임프레임 시그널 | 장중 300초, 장외 24h |
| GET | /ai/top-picks | BUY 시그널 상위 종목 (학습 100종목 중) | 300초 |
| GET | /ai/signals/history/{code} | 시그널 이력 최근 30일 | 300초 |

**`/ai/{code}/signal` 응답 스키마:**
```json
{
  "code": "005930",
  "name": "삼성전자",
  "signal": "BUY",
  "signal_score": 72.4,
  "signal_breakdown": {
    "technical_score": 68.1,
    "lstm_score": 75.2,
    "technical_weight": 0.4,
    "lstm_weight": 0.6
  },
  "lstm_available": true,
  "confidence": 67.3,
  "as_of": "2026-06-03T15:30:00+09:00"
}
```

**`/ai/{code}/predict` 응답 스키마:**
```json
{
  "code": "005930",
  "current_price": 73400,
  "prediction": {
    "bullish": [74200, 75100, 75800, 76200, 77000],
    "base":    [73800, 74200, 74100, 74500, 74900],
    "bearish": [73100, 72800, 72500, 72000, 71800]
  },
  "confidence": 67.3,
  "lstm_available": true
}
```

---

## 5. DB 마이그레이션 (Alembic, Phase 3+4 전체)

| 버전 | 테이블 | Phase |
|---|---|---|
| 버전 3 | `ai_signals_history` | 3 |
| 버전 4 | `watchlist_groups`, `watchlist_items` | 4 |
| 버전 5 | `portfolios`, `trades` | 4 |
| 버전 6 | `risk_settings`, `alert_settings`, `backtest_results` | 4 |

각 테이블 스키마는 TRD 섹션 4 (4.2~4.8) 그대로 사용.

---

## 6. 테스트 전략

| 파일 | 테스트 수 (목표) | 내용 |
|---|---|---|
| `tests/test_ml.py` | 8 | features.py 피처 수/NaN, predict.py MC Dropout 출력 shape, model forward pass |
| `tests/test_patterns.py` | 4 | pattern_service.py 패턴 감지 유닛 테스트 |
| `tests/test_ai.py` | 12 | /ai/ 엔드포인트 8개 mock 통합 + fallback 동작 + 캐시 hit/miss |
| **합계** | **24** | 기존 52 + 24 = 76 passing 목표 |

---

## 7. 구현 순서

1. DB 마이그레이션 (버전 3~6) — 테이블 생성
2. `ml/features.py` — 피처 엔지니어링
3. `ml/model.py` — StockLSTM 클래스
4. `ml/train.py` — 학습 스크립트
5. `ml/predict.py` — 추론 인터페이스
6. `ml/pattern_matcher.py` — 유사 패턴 검색
7. `services/pattern_service.py` — 캔들 패턴
8. `services/ai_service.py` — 시그널 조율
9. `tasks/` 폴더 구조 변경 + `ai_tasks.py`
10. `api/routes/ai.py` — 8개 엔드포인트
11. `main.py` — ai 라우터 + APScheduler 등록
12. 테스트 작성
13. `docs/progress.md` 업데이트
