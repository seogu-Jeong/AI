# Phase 4-B — 백테스팅 엔진 설계

**작성일:** 2026-06-04
**담당:** hygrenn (백엔드)
**레퍼런스:** TRD 섹션 10, docs/progress.md Phase 4-B

---

## 1. 결정 사항

| 항목 | 결정 |
|---|---|
| 시그널 소스 | OHLCV 기반 즉석 계산 (기술적 지표 100%) |
| 실행 방식 | 동기 (POST /backtest/run에서 직접 계산 후 응답) |
| 아키텍처 | 단순 함수형 서비스 (`run_backtest(config)`) |
| 기존 재사용 | `ml/features.build_features()`, `services/ai_service._calc_tech_score()` |

---

## 2. 파일 목록

### 신규 생성
| 파일 | 역할 |
|---|---|
| `backend/services/backtest_service.py` | 백테스팅 엔진 |
| `backend/api/routes/backtest.py` | POST /backtest/run, GET /backtest/{id} |
| `tests/test_backtest.py` | 통합 테스트 4개 |

### 수정
| 파일 | 변경 내용 |
|---|---|
| `backend/main.py` | backtest 라우터 등록 |
| `docs/progress.md` | Phase 4-B 완료 업데이트 |

---

## 3. BacktestConfig 입력 스키마

```python
class BacktestConfig(BaseModel):
    code: str                              # 종목코드
    start_date: date                       # 시작일
    end_date: date                         # 종료일
    initial_cash: int = 10_000_000        # 초기 자금 (원)
    entry_signal_score: float = 65.0      # 매수 기준 점수
    exit_signal_score: float = 35.0       # 매도 기준 점수
    stop_loss_pct: float = 0.05           # 손절 비율 (5%)
    take_profit_pct: float = 0.15         # 익절 비율 (15%)
    commission_rate: float = 0.00015      # 수수료율 (0.015%)

    @validator("end_date")
    def end_after_start(cls, v, values):
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v
```

---

## 4. backtest_service.py 설계

### 4.1 실행 흐름

```
run_backtest(config, user_id, db)
  ① pykrx OHLCV 다운로드 (start_date ~ end_date)
      → get_market_ohlcv_by_date() 직접 호출 (Redis 캐시 없이 — 임의 기간이라 캐시 키 복잡)
  ② build_features(df) → 날짜별 13개 피처 계산
  ③ 날짜별 기술적 지표 지표 계산 → _calc_tech_score(indicators)
  ④ 매매 시뮬레이션 (반복문):
      cash = initial_cash, position = 0, entry_price = 0
      for each day:
        score = tech_score[day]
        if score >= entry_signal_score and position == 0:
            shares = int(cash * 0.95 / price)   # 95% 투자
            cost = shares * price * (1 + commission_rate)
            cash -= cost; position = shares; entry_price = price
        elif position > 0:
            change = (price - entry_price) / entry_price
            if (score <= exit_signal_score
                    or change <= -stop_loss_pct
                    or change >= take_profit_pct):
                revenue = position * price * (1 - commission_rate)
                pnl = revenue - position * entry_price
                cash += revenue
                trades_log.append({date, entry_price, exit_price, pnl})
                position = 0
        equity_curve.append(cash + position * price)
  ⑤ 성과 지표 계산
  ⑥ BacktestResult DB 저장
  ⑦ 결과 반환
```

### 4.2 성과 지표 계산

```python
# MDD
peak = equity_curve[0]
mdd = 0.0
for v in equity_curve:
    if v > peak: peak = v
    dd = (peak - v) / peak
    if dd > mdd: mdd = dd

# 샤프비율 (무위험이자율 0, 연간화)
daily_returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                 for i in range(1, len(equity_curve))]
mean_r = sum(daily_returns) / len(daily_returns) if daily_returns else 0
std_r = math.sqrt(sum((r - mean_r)**2 for r in daily_returns) / len(daily_returns))
sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0

# 승률
wins = sum(1 for t in trades_log if t["pnl"] > 0)
win_rate = wins / len(trades_log) * 100 if trades_log else 0

# 총 수익률
total_return_pct = (equity_curve[-1] - config.initial_cash) / config.initial_cash * 100
```

### 4.3 DB 저장

```python
result = BacktestResult(
    user_id=user_id,
    stock_code=config.code,
    strategy_config=config.model_dump(),   # JSON 스냅샷
    period_start=config.start_date,
    period_end=config.end_date,
    total_return_pct=total_return_pct,
    mdd_pct=mdd * 100,
    sharpe_ratio=sharpe,
    win_rate_pct=win_rate,
    total_trades=len(trades_log),
    result_detail={
        "trades": trades_log,
        "equity_curve": equity_curve[::5],   # 5일 간격으로 저장 (용량 절약)
    },
)
db.add(result); await db.commit()
```

---

## 5. API 엔드포인트

### POST /backtest/run

```
인증: 필요 (JWT)
Rate limit: 5/minute (무거운 계산)
Request: BacktestConfig JSON
Response: BacktestResult 전체
```

응답 예시:
```json
{
  "id": "uuid",
  "stock_code": "005930",
  "period_start": "2024-01-01",
  "period_end": "2025-12-31",
  "total_return_pct": 12.34,
  "mdd_pct": 8.21,
  "sharpe_ratio": 1.45,
  "win_rate_pct": 55.0,
  "total_trades": 8,
  "strategy_config": {...},
  "result_detail": {"trades": [...], "equity_curve": [...]}
}
```

### GET /backtest/{id}

```
인증: 필요 (JWT)
본인 결과만 조회 가능 (user_id 검증)
없으면 404
```

---

## 6. 테스트 전략

```
tests/test_backtest.py (4개)
  test_run_backtest_returns_result   — backtest_service mock, 200 응답 + 필수 필드
  test_run_backtest_invalid_dates    — end_date <= start_date → 422
  test_get_backtest_not_found        — 존재하지 않는 id → 404
  test_get_backtest_success          — DB에 직접 삽입 후 조회
```

KIS API, pykrx 모두 mock. `backtest_service.run_backtest`를 mock해서 라우터만 테스트.

---

## 7. 구현 순서

1. `backtest_service.py` 구현
2. `api/routes/backtest.py` 구현 (TDD)
3. `main.py` 라우터 등록
4. `docs/progress.md` 업데이트
