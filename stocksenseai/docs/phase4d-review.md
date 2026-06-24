# Phase 4-D Review

검토일: 2026-06-04

## 결론

Phase 4-D의 관심종목 CRUD 라우터 8개, 목표가/손절가 알림 태스크, 일일 손실 체크 태스크, watchlist 라우터 등록은 구현되어 있습니다. Phase 4-D 선택 테스트도 현재 환경에서 `11 passed`로 재현됩니다.

다만 “목표가 알림 + 일일 손실 자동 차단”을 운영 기능으로 보려면 아직 위험한 부분이 있습니다. 특히 일일 손실 계산은 Phase 4-A의 손익 계산 문제를 그대로 사용하고, `trading_blocked`를 설정해도 주문 차단 로직에서 아직 반영하지 않기 때문에 실제 자동 차단 효과가 없습니다.

## 수정 필요

1. 일일 손실 자동 차단이 실제 주문 차단으로 이어지지 않습니다.
   - 위치: `backend/tasks/email_tasks.py:242`, `backend/services/risk_service.py:94`
   - `_check_daily_loss_async()`는 `settings.trading_blocked = True`로 바꾸지만, `risk_service.check_order()`는 `trading_blocked`를 확인하지 않습니다. 따라서 자동 차단 태스크가 실행되어도 이후 주문이 계속 나갈 수 있습니다.
   - 권고: `check_order()` 초반에 `settings.trading_blocked`면 즉시 주문을 거부하세요. 실거래 모드에서는 별도 kill switch처럼 취급해야 합니다.

2. 일일 손실 계산 자체가 정확하지 않습니다.
   - 위치: `backend/tasks/email_tasks.py:234`, `backend/services/risk_service.py:57`, `backend/services/risk_service.py:78`
   - `_check_daily_loss_async()`는 `_get_today_loss()`를 사용합니다. 이 함수는 SELL의 `executed_price - order_price`로 손익을 계산하는데, `order_price`는 매수 평균단가가 아니라 매도 주문가입니다. 시장가 매도는 `order_price`가 `None`이라 손실 계산에서 빠집니다.
   - 권고: 매도 체결 시 평균단가 또는 lot 기준 실현손익을 Trade나 별도 ledger에 저장하고, 일일 손실은 그 값을 기준으로 계산하세요.

3. 리스크 차단과 알림 쿨다운이 paper/real 모드를 구분하지 않습니다.
   - 위치: `backend/models/risk.py:13`, `backend/tasks/email_tasks.py:223`
   - `RiskSettings`는 user 단위이고 mode 컬럼이 없습니다. `daily_loss_alert:{user_id}` Redis key도 mode를 포함하지 않습니다. 모의투자 손실이 실거래 차단 상태와 섞일 수 있습니다.
   - 권고: risk settings와 cooldown key를 mode별로 분리하거나, 최소한 paper 손실이 real 차단에 영향을 주지 않도록 명확히 분기하세요.

4. 일일 손실 기준 시간이 UTC `date.today()`라 KST 거래일과 어긋납니다.
   - 위치: `backend/tasks/email_tasks.py:197`
   - 한국 주식 거래일 기준이어야 하는데 UTC 자정 기준으로 `today_start`를 만듭니다. 서버 시간대와 실행 시점에 따라 당일 손실 범위가 틀어질 수 있습니다.
   - 권고: `Asia/Seoul` 기준 거래일 시작을 aware datetime으로 만들고 DB 저장 시간대와 일관되게 비교하세요.

5. 목표가/손절가를 한번 설정하면 API로 제거할 수 없습니다.
   - 위치: `backend/api/routes/watchlist.py:44`, `backend/api/routes/watchlist.py:247`
   - `ItemUpdate.target_price_high`와 `target_price_low`는 `None` 기본값이고, update 로직은 `is not None`일 때만 반영합니다. 클라이언트가 `null`을 보내도 기존 목표가가 유지됩니다.
   - 권고: Pydantic의 `model_fields_set`을 사용해 “필드 미전송”과 “명시적 null”을 구분하고, null이면 목표가를 제거하세요.

6. 관심종목 중복 추가가 동시 요청에서 500으로 터질 수 있습니다.
   - 위치: `backend/api/routes/watchlist.py:203`, `backend/models/watchlist.py:32`
   - API는 insert 전 중복 조회를 하지만, 동시에 같은 종목을 추가하면 DB UniqueConstraint에서 IntegrityError가 날 수 있습니다. 현재는 이를 409로 변환하지 않습니다.
   - 권고: commit 시 `IntegrityError`를 잡아 rollback 후 409를 반환하세요.

7. 알림 태스크의 모듈 레벨 `except ImportError: pass`가 장애를 숨길 수 있습니다.
   - 위치: `backend/tasks/email_tasks.py:6`
   - import 실패를 조용히 넘기면 이후 `_check_price_alerts_async()`에서 `AsyncSessionLocal`, `WatchlistItem`, `get_redis` 등이 NameError로 터질 수 있습니다.
   - 권고: 테스트 patch 편의를 위해 broad import guard를 두지 말고, 필요한 함수 내부에서 명시적으로 import하거나 실패를 로그로 남기세요.

8. 목표가 알림 실패/가격 조회 실패가 기록되지 않습니다.
   - 위치: `backend/tasks/email_tasks.py:141`, `backend/tasks/email_tasks.py:145`
   - 특정 종목 현재가 조회 실패를 `pass`로 무시합니다. 알림이 빠져도 사용자는 모릅니다.
   - 권고: 최소한 stock_code, user_count, 실패 사유를 로그로 남기고, 반복 실패를 확인할 수 있게 하세요.

## 권고 사항

1. 목표가 알림 쿨다운 정책을 더 명확히 하세요.
   - 위치: `backend/tasks/email_tasks.py:161`, `backend/tasks/email_tasks.py:174`
   - 현재는 조건이 계속 유지되면 24시간마다 다시 발송됩니다. 의도한 동작이라면 문서화하고, “한 번 돌파 후 가격이 다시 내려갔다가 재돌파할 때만 알림”이 목표라면 상태 저장 방식이 필요합니다.

2. watchlist API 테스트가 mock DB 중심이라 실제 DB 제약을 검증하지 않습니다.
   - 위치: `tests/test_watchlist.py:30`
   - 현재 테스트는 cascade delete, unique constraint, IntegrityError 처리, 실제 정렬, 실제 `/watchlist/items` 조회를 검증하지 않습니다.
   - 권고: test DB를 사용하는 통합 테스트를 추가하세요.

3. 알림 태스크 테스트도 핵심 실패 케이스가 빠져 있습니다.
   - 위치: `tests/test_alert_tasks.py:35`
   - low target 알림, Redis setex 호출, 가격 조회 실패, `daily_loss_limit=False`, mode별 분리, 이미 trading_blocked인 경우를 더 검증하세요.

4. 그룹/아이템 update 응답에 최신 객체를 반환하는 편이 프론트 구현에 유리합니다.
   - 위치: `backend/api/routes/watchlist.py:140`, `backend/api/routes/watchlist.py:266`
   - 현재는 `{updated: true}`만 반환합니다. 프론트는 다시 GET을 호출해야 최신 목표가/정렬을 반영할 수 있습니다.

5. 알림 이메일 주소 검증은 Phase 4-A의 `/alerts/settings`와 함께 보완하세요.
   - 위치: `backend/api/routes/alerts.py:30`, `backend/tasks/email_tasks.py:262`
   - 잘못된 `notification_email`이 저장되면 SendGrid 실패가 Celery retry로 반복될 수 있습니다.

## 확인한 점

- `pytest -q tests/test_watchlist.py tests/test_alert_tasks.py`: 11 passed
- `/watchlist` 라우터가 `backend/main.py`에 등록되어 있음
- `check_price_alerts.delay` 5분, `check_daily_loss.delay` 10분 스케줄 등록 확인
- 관심종목 모델과 watchlist 마이그레이션의 user/stock unique constraint 확인
- Phase 4-D 관련 코드 수정은 하지 않았고 리뷰 파일만 추가했습니다.
