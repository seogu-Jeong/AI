# Phase 4-B Review

검토일: 2026-06-04

## 결론

Phase 4-B의 백테스팅 엔진 파일, API 라우터, DB 저장 흐름, 테스트 파일은 계획에 맞춰 생성되어 있습니다. `POST /backtest/run`, `GET /backtest/{id}` 라우터도 등록되어 있고, pykrx 조회는 `asyncio.to_thread()`로 감싸져 있습니다.

하지만 백테스트 결과를 의사결정에 쓰기에는 아직 검증이 부족합니다. 현재 테스트는 실제 엔진 로직보다 API wrapper와 mock 결과에 가까우며, 입력값 제약, 데이터 조회 실패 처리, 손익/수수료 계산, 결과 시계열 구조에 개선이 필요합니다.

## 수정 필요

1. 실제 백테스트 엔진 로직 테스트가 부족합니다.
   - 위치: `tests/test_backtest.py:1`, `backend/services/backtest_service.py:70`
   - 라우터 테스트는 `run_backtest()`를 mock하는 흐름이 중심이라 `_simulate()`, `_compute_metrics()`, `_compute_daily_scores()`의 정확성을 검증하지 못합니다.
   - 권고: 고정 OHLCV fixture로 매수, 매도, 손절, 익절, 무거래, 수수료 반영, MDD/샤프 계산을 직접 검증하세요.

2. 요청 입력값에 범위 검증이 없습니다.
   - 위치: `backend/api/routes/backtest.py:20`
   - `initial_cash <= 0`, 음수 수수료, 100% 초과 수수료, 음수 손절/익절, 진입 점수보다 높은 청산 점수 같은 비정상 설정이 들어올 수 있습니다.
   - 권고: `Field(gt=0)`, 점수 0~100, 수수료 0~0.01, 손절/익절 0~1 등 현실적인 범위를 강제하세요.

3. pykrx 조회 실패가 통제된 API 오류로 변환되지 않습니다.
   - 위치: `backend/services/backtest_service.py:31`, `backend/services/backtest_service.py:205`
   - `_fetch_ohlcv()`가 pykrx 예외를 그대로 올립니다. 네트워크 문제, 잘못된 종목코드, pykrx 응답 오류가 500으로 터질 수 있습니다.
   - 권고: pykrx 예외를 잡아 400/502 계열 HTTPException으로 변환하고, 종목코드 형식도 사전 검증하세요.

4. 거래 로그의 PnL이 매수 수수료를 반영하지 않습니다.
   - 위치: `backend/services/backtest_service.py:96`, `backend/services/backtest_service.py:114`, `backend/services/backtest_service.py:115`
   - 매수 시 cash에서는 수수료 포함 비용을 차감하지만, 매도 로그의 `pnl`은 `revenue - position * entry_price`라 매수 수수료가 빠집니다. 총수익률과 거래별 손익이 서로 다른 기준이 됩니다.
   - 권고: `entry_cost = shares * entry_price * (1 + commission_rate)`를 저장하고, PnL은 `revenue - entry_cost`로 계산하세요.

5. 결과의 `equity_curve`에 날짜가 없습니다.
   - 위치: `backend/services/backtest_service.py:238`
   - `equity_curve[::5]` 숫자 배열만 저장되어 차트에서 어느 날짜의 평가금액인지 알 수 없습니다.
   - 권고: `{date, equity}` 형태로 저장하고, downsample을 하더라도 날짜를 함께 보존하세요.

6. 마지막 날 미청산 포지션 처리가 명확하지 않습니다.
   - 위치: `backend/services/backtest_service.py:129`, `backend/services/backtest_service.py:176`
   - equity에는 미청산 포지션 평가금액이 포함되지만 `total_trades`, `win_rate_pct`, 거래 로그에는 반영되지 않습니다. 백테스트 해석이 애매합니다.
   - 권고: 종료일 강제 청산 옵션을 두거나, 결과에 `open_position`을 별도로 저장하고 지표 설명을 명확히 하세요.

7. 긴 기간/대량 요청 제한이 약합니다.
   - 위치: `backend/api/routes/backtest.py:57`, `backend/services/backtest_service.py:210`
   - rate limit은 5/min이지만 기간 제한이 없어 긴 기간 요청이 pykrx와 CPU 계산을 오래 점유할 수 있습니다.
   - 권고: 최대 기간, 최소 데이터 길이, 동시 실행 제한을 추가하세요. 사용자가 혼자 쓰더라도 자동화가 붙으면 API가 쉽게 밀릴 수 있습니다.

8. 저장되는 전략 설정에 종목코드와 기간이 분리되어 있어 재현 정보가 불완전하게 보일 수 있습니다.
   - 위치: `backend/services/backtest_service.py:221`
   - `stock_code`, `period_start`, `period_end`는 모델 컬럼에 저장되지만 `strategy_config` 자체에는 없습니다. 결과 JSON을 별도 시스템으로 넘기면 설정만으로 재현하기 어렵습니다.
   - 권고: `strategy_config` 안에도 `code`, `start_date`, `end_date`를 포함하거나 응답 문서에서 컬럼 조합이 재현 단위임을 명시하세요.

## 권고 사항

1. Phase 3의 기술적 점수 공식 검증이 먼저 필요합니다.
   - 위치: `backend/services/backtest_service.py:64`, `backend/services/ai_service.py:34`
   - 백테스트는 `_calc_tech_score()` 결과에 전적으로 의존합니다. Phase 3 리뷰에서 지적한 RSI/점수 해석 문제가 남아 있으면 백테스트 매수/매도 판단도 그대로 왜곡됩니다.

2. “AI 백테스트”가 아니라 “기술적 지표 점수 백테스트”로 문구를 분리하세요.
   - 위치: `backend/services/backtest_service.py:13`, `backend/services/backtest_service.py:15`
   - LSTM 예측값이나 학습 모델 confidence는 백테스트에 사용되지 않습니다. 현재 구조는 OHLCV 기반 기술 지표 점수 전략입니다.

3. 백테스트 결과 저장 전에 입력 파라미터와 데이터 길이를 로그로 남기세요.
   - 위치: `backend/services/backtest_service.py:215`
   - 결과가 이상할 때 데이터 부족, 기간, 점수 threshold, 거래횟수를 추적하기 어렵습니다.

4. API 응답 스키마 문서 또는 Pydantic response model을 추가하세요.
   - 위치: `backend/api/routes/backtest.py:40`
   - 현재는 dict serializer라 프론트가 기대하는 `result_detail` 구조가 코드 외부에서 고정되지 않습니다.

5. pykrx 호출을 캐시하거나 같은 요청 중복 실행을 막으세요.
   - 위치: `backend/services/backtest_service.py:205`
   - 같은 종목/기간/설정을 반복 요청하면 매번 외부 데이터 조회와 계산을 수행합니다.

## 확인한 점

- `pytest -q tests/test_trades.py tests/test_portfolio.py tests/test_risk.py tests/test_backtest.py`: 19 passed, 1 skipped
- `pytest -q`: 2 failed, 49 passed, 57 errors
- 전체 테스트 실패 원인: 테스트 DB의 `ai_signals_history` 타입/테이블 중복 생성 문제와 현재 Python/bcrypt/passlib 조합의 인증 테스트 실패가 섞여 있습니다.
- Phase 4-B 선택 테스트는 API wrapper 중심으로 통과했지만, 엔진 수학/거래 시뮬레이션의 정합성은 아직 충분히 검증되지 않았습니다.
