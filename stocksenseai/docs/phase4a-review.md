# Phase 4-A Review

검토일: 2026-06-04

## 결론

Phase 4-A의 주요 파일, 라우터, 서비스, Celery 태스크, 마이그레이션은 계획에 맞춰 생성되어 있습니다. 주문 실행, 체결 폴링, 포트폴리오 DB 반영, 리스크 설정/체크, 알림 설정 API까지 표면 기능은 들어가 있습니다.

다만 현재 구현은 실거래에 올리기에는 위험합니다. 특히 주문 입력 검증, 시장가 주문 리스크 계산, 체결 시각 저장, 매도/손익 계산, 계좌 모드 전환 후 취소 처리에 문제가 있어 모의투자로만 충분히 검증해야 합니다.

## 수정 필요

1. 주문 입력 검증이 부족해서 잘못된 주문이 KIS로 나갈 수 있습니다.
   - 위치: `backend/api/routes/trades.py:18`, `backend/services/kis_service.py:78`, `backend/services/kis_service.py:79`
   - `order_type`, `price_type`이 단순 문자열이고 `quantity`, `price`, `stock_code` 제약이 없습니다. `order_type != "BUY"`는 전부 매도로 처리되고, `price_type != "MARKET"`은 전부 지정가로 처리됩니다.
   - 권고: `Literal["BUY", "SELL"]`, `Literal["MARKET", "LIMIT"]`, `quantity > 0`, `LIMIT price > 0`, 종목코드 6자리 검증을 Pydantic에서 강제하세요.

2. 시장가 주문이 리스크 체크를 사실상 우회합니다.
   - 위치: `backend/api/routes/trades.py:37`, `backend/api/routes/trades.py:50`, `backend/services/risk_service.py:99`
   - 시장가 주문은 `price=0` 기본값으로 `check_order()`에 들어가며 `new_order_value = quantity * price`가 0이 됩니다. 큰 시장가 매수도 종목별 한도와 손실 한도 계산에 반영되지 않습니다.
   - 권고: 시장가 주문은 현재가 또는 호가 기준 추정 체결가를 조회해서 리스크 계산에 넣고, 조회 실패 시 주문을 막으세요.

3. 주문 취소가 원 주문의 `mode`가 아니라 현재 사용자 `mode`로 KIS를 호출합니다.
   - 위치: `backend/api/routes/trades.py:125`, `backend/services/kis_service.py:113`
   - 사용자가 모의투자 주문을 낸 뒤 실거래 모드로 바꾸고 취소하면, 취소 요청이 실거래 키/계좌로 나갈 수 있습니다.
   - 권고: 취소 시 `trade.mode`와 현재 `user.mode`가 다르면 거부하거나, KIS 호출용 user context를 원 주문 mode로 고정하세요. `kis_order_no` 없음도 방어해야 합니다.

4. 체결 폴링 결과의 `filled_at` 타입이 DB 모델과 맞지 않을 가능성이 큽니다.
   - 위치: `backend/services/kis_service.py:193`, `backend/tasks/order_tasks.py:43`
   - `poll_fill()`은 KIS의 `ord_tmd` 문자열을 그대로 반환하고, 태스크는 이를 `Trade.filled_at` DateTime 컬럼에 저장합니다. 실제 체결 시 DB 저장 오류가 날 수 있습니다.
   - 권고: KIS 체결일/체결시각을 KST aware `datetime`으로 변환해서 저장하세요. 최소한 저장 전 타입 검증 테스트를 추가하세요.

5. KIS 체결 조회 실패가 미체결로 처리되어 장애가 숨겨집니다.
   - 위치: `backend/services/kis_service.py:184`, `backend/tasks/order_tasks.py:31`
   - `poll_fill()`이 모든 예외를 잡고 `None`을 반환합니다. 네트워크 장애, 인증 오류, KIS 응답 스키마 변경도 미체결처럼 재시도되다가 끝납니다.
   - 권고: KIS 장애와 미체결을 분리하세요. 장애는 로그/알림/실패 상태로 남기고, 미체결만 재시도해야 합니다.

6. 매도 체결 시 보유수량 검증과 실현손익 처리가 없습니다.
   - 위치: `backend/tasks/order_tasks.py:87`, `backend/tasks/order_tasks.py:90`
   - 보유 종목이 없어도 매도 주문이 FILLED 처리될 수 있고, 보유수량보다 많이 팔면 포트폴리오가 삭제됩니다. 실현손익도 별도 기록되지 않습니다.
   - 권고: 주문 전 SELL 가능 수량을 검증하고, 체결 후에는 평균단가 기준 실현손익을 저장하세요. 초과 매도는 실패 상태로 남겨야 합니다.

7. 일일 손실, 성과, 지표 계산이 실제 손익을 계산하지 못합니다.
   - 위치: `backend/services/risk_service.py:78`, `backend/api/routes/portfolio.py:94`, `backend/api/routes/portfolio.py:122`
   - SELL의 `executed_price - order_price`로 손익을 계산합니다. `order_price`는 매수가가 아니라 매도 주문가이고, 시장가 매도는 `None`이라 손익 계산에서 빠집니다.
   - 권고: 매도 체결 시 매수 평균단가 또는 lot 기반 cost basis를 사용하세요. 포트폴리오 성과/리스크는 그 실현손익 데이터를 기준으로 계산해야 합니다.

8. `trading_blocked`, `stop_loss_enabled`, 일일 손실 스케줄이 실제로 동작하지 않습니다.
   - 위치: `backend/services/risk_service.py:94`, `backend/tasks/email_tasks.py:104`, `backend/main.py:37`
   - 리스크 설정에 `trading_blocked`가 있지만 주문 전 체크에서 사용하지 않습니다. `check_daily_loss`는 APScheduler에 등록되어 있지만 본문이 `pass`입니다.
   - 권고: `trading_blocked`면 주문을 즉시 차단하고, 일일 손실 초과 시 해당 플래그를 자동 설정/알림하도록 구현하세요.

9. Phase 3 AI 시그널 스냅샷이 주문에 저장되지 않습니다.
   - 위치: `backend/api/routes/trades.py:44`
   - 계획/모델 흐름상 주문 시점의 AI 판단을 `ai_signal_at_order`로 남겨야 하지만 현재 Trade 생성에 포함되지 않습니다.
   - 권고: 주문 직전 `calculate_signal()` 또는 최신 `ai_signals_history`를 조회해 JSON으로 저장하세요.

10. 예약된 가격 알림은 아직 구현되지 않았고 테스트도 없습니다.
    - 위치: `backend/tasks/email_tasks.py:98`, `backend/main.py:36`
    - `check_price_alerts`가 5분마다 예약되어 있지만 아무 작업도 하지 않습니다. `alerts` API에 대한 테스트 파일도 없습니다.
    - 권고: Phase 4-A 완료 표기에는 “알림 설정 API 구현, 가격/손실 자동 알림 미구현”으로 분리해서 적으세요.

## 권고 사항

1. 실거래 전용 방어선을 별도로 추가하세요.
   - 위치: `backend/api/routes/auth.py:299`, `backend/api/routes/trades.py:26`
   - `PUT /auth/mode`는 키 존재만 확인하고 모드를 바꿉니다. 실거래 전환은 2단계 확인, 최근 키 검증, 소액 주문 제한, kill switch가 필요합니다.

2. KIS 잔고와 DB 포트폴리오의 차이를 감지하는 reconciler가 필요합니다.
   - 위치: `backend/services/kis_service.py:198`, `backend/api/routes/portfolio.py:35`
   - 현재 포트폴리오는 DB 체결 태스크 결과에 의존합니다. 수동매매, 외부 체결, 폴링 실패가 있으면 실제 계좌와 다를 수 있습니다.

3. 리스크 설정 입력값 범위를 제한하세요.
   - 위치: `backend/api/routes/risk.py:14`
   - `max_per_stock_pct`, `daily_loss_limit_pct`가 음수나 100 초과로 들어와도 모델 레벨에서 막지 않습니다.

4. 알림 이메일 입력은 `EmailStr` 등으로 검증하세요.
   - 위치: `backend/api/routes/alerts.py:30`
   - 잘못된 이메일이 저장되면 SendGrid 실패가 Celery retry로만 반복될 수 있습니다.

5. KIS API 응답 필드 매핑은 실제 응답 fixture로 고정 테스트가 필요합니다.
   - 위치: `backend/services/kis_service.py:189`
   - `ccld_dvsn`, `avg_prvs`, `tot_ccld_qty`, `ord_tmd` 필드가 실제 응답과 맞는지 테스트 데이터로 검증해야 합니다.

## 확인한 점

- `pytest -q tests/test_trades.py tests/test_portfolio.py tests/test_risk.py tests/test_backtest.py`: 19 passed, 1 skipped
- `pytest -q`: 2 failed, 49 passed, 57 errors
- 전체 테스트 실패 원인: 테스트 DB의 `ai_signals_history` 타입/테이블 중복 생성 문제와 현재 Python/bcrypt/passlib 조합의 인증 테스트 실패가 섞여 있습니다.
- Phase 4-A 관련 선택 테스트는 통과했지만, 실제 KIS 연동/체결/손익 계산 경로는 대부분 mock 기반이라 운영 안정성을 보장하지 못합니다.
