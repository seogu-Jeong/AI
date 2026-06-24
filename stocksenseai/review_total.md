# StockSenseAI 전체 개발 검토

검토일: 2026-06-08  
대상: `dev` 브랜치 (`8f8dd6b`)  
중점: 계획 이행률보다 프로그램 오류, 데이터 무결성, 통합 실패, 운영 위험

## 2026-06-08 hwang 브랜치 백엔드 수정 결과

아래 백엔드 문제를 수정하고 Docker 통합 환경에서 검증했다.

- P1-1: 리스크 차단 및 종목 한도 검사를 BUY에만 적용해 SELL 청산 허용
- P1-2: 체결 폴링이 주문 당시 `Trade.mode`를 사용하도록 변경
- P1-3/P2-9: `filled_quantity` 컬럼 및 migration 추가, 누적 체결 증가분만 장부 반영, Trade row lock으로 중복 반영 방지
- P1-6: 런타임/개발/ML 의존성 분리, pytest 충돌 수정, `pandas-ta-classic` 전환, PyTorch 기본 이미지 제외
- P2-8: Portfolio row lock과 미체결 매도수량 차감으로 동시 초과 매도 방지
- P2-11/P2-12: 최근 실제 거래일 가격 역방향 탐색 및 pykrx 호출 `asyncio.to_thread()` 이동
- P2-14: KIS 서비스 구현 복구 및 체결/잔고 응답 `rt_cd` 오류 검사
- P2-15: FastAPI 프로세스별 APScheduler 제거, 단일 Celery Beat 서비스 추가
- P2-16: `trade_filled` 알림 설정 확인 후 체결 이메일 발송
- merge 회귀: KIS 주문/잔고 구현, 주요 백엔드 라우터 7개, ORM의 `realized_pnl`/`enforce_hard_stop`/`notification_email` 복구
- 리스크 수동 차단 해제 API `POST /risk/unblock` 추가

검증:

- `docker compose build backend frontend`: 성공
- `docker compose up -d`: PostgreSQL, Redis, migration, backend, Celery worker/beat, frontend 정상 기동
- Alembic: 단일 head `c9d0e1f2a3b4`, 신규 DB 전체 migration 성공
- 백엔드 Docker 기본 테스트: `124 passed, 1 skipped`
- 거래/리스크 회귀 테스트: `14 passed`
- `/health`, frontend HTTP, 복구된 주요 API OpenAPI 노출 확인

아직 남은 백엔드 항목:

- P2-7: 일일 기준 평가액 snapshot과 미실현 손익을 포함한 손실률 계산
- P2-10: `/simulate/download` 인증 SSE를 fetch streaming 또는 단기 토큰 방식으로 변경
- P3-19: 백테스트 최대 2년 제한 및 장시간 작업 Celery job 전환
- P3-20: 모델 artifact 배포/버전/검증 절차 정의

## 결론

백엔드는 Phase 1~4의 주요 API와 모델이 상당 부분 구현되어 있고, Alembic 마이그레이션도 단일 head로 연결되어 있다. 프론트 단위 테스트도 모두 통과한다.

하지만 현재 상태를 실제 사용 가능한 통합 서비스 또는 실거래 가능한 서비스로 보기는 어렵다. 프론트는 로그인, 실시간 시세, 주문, 포트폴리오, AI, 백테스트, 시뮬레이터 등 대부분이 mock 또는 백엔드 계약과 불일치한 상태다. 백엔드 거래 경로에는 매도 차단, 부분 체결 오처리, 모드 전환 후 체결 추적 실패처럼 실제 자산과 내부 장부를 어긋나게 만들 수 있는 문제가 있다.

실거래 기능은 아래 P1 문제를 수정하고 실제 KIS 모의투자 환경에서 주문-부분체결-취소-모드전환-재시작 시나리오를 검증하기 전까지 활성화하지 않는 것을 권고한다.

## P1 - 즉시 수정 필요

### 1. 리스크 차단이 매도까지 막아 포지션을 청산할 수 없다

- 위치: `backend/api/routes/trades.py:84`, `backend/services/risk_service.py:100-130`
- `place_order()`는 BUY/SELL 구분 없이 `check_order()`를 호출한다.
- `check_order()`는 주문금액을 기존 보유금액에 더해 종목 비중을 계산하고, `trading_blocked`이면 모든 주문을 거부한다.
- 그 결과 종목 비중 초과 또는 일일 손실 차단 상태에서 위험을 줄이기 위한 매도도 거부될 수 있다.
- PRD의 일일 손실 차단 요구사항은 "추가 매수 자동 차단"이며, 매도 차단이 아니다.

권고:

- `check_order()`에 `order_type`을 전달한다.
- 종목 비중 및 일일 손실 hard stop은 BUY에만 적용한다.
- SELL은 보유수량 검증만 수행하고 항상 청산 가능하게 둔다.

### 2. 주문 후 사용자가 mode를 바꾸면 체결 폴링이 잘못된 KIS 계정을 조회한다

- 위치: `backend/tasks/order_tasks.py:8-31`, `backend/services/kis_service.py:157-164`
- Celery 태스크는 주문 당시 `mode`를 인자로 받지만 사용하지 않는다.
- 태스크가 DB에서 최신 User를 읽고 `kis_service.poll_fill(user, ...)`을 호출하므로, 주문 후 사용자가 paper/real mode를 전환하면 다른 KIS 서버와 계정으로 체결을 조회한다.
- 정상 체결 주문이 `UNKNOWN`으로 끝나고 포트폴리오가 갱신되지 않을 수 있다.

권고:

- 체결 조회 함수가 명시적인 주문 mode와 해당 mode의 키를 사용하도록 변경한다.
- 주문 당시 계정 식별 정보도 Trade에 보존하거나 변경 불가능한 주문 컨텍스트를 별도 저장한다.

### 3. 부분 체결을 전체 체결로 처리하여 포트폴리오 수량과 손익이 틀어진다

- 위치: `backend/services/kis_service.py:197-203`, `backend/tasks/order_tasks.py:44-59`, `backend/tasks/order_tasks.py:79-99`
- KIS 응답에서 `filled_qty`를 읽지만, 포트폴리오 갱신은 항상 `trade.quantity` 전체를 사용한다.
- 일부만 체결되어도 Trade를 `FILLED`로 표시하고 전체 주문 수량을 매수/매도 처리한다.
- 부분 매도 시 실현손익도 실제 체결수량이 아닌 전체 주문수량으로 계산된다.

권고:

- `filled_quantity`, `remaining_quantity`, 부분 체결 상태를 Trade에 저장한다.
- 누적 체결량의 증가분만 포트폴리오에 반영한다.
- 동일 Celery 태스크가 중복 실행되어도 재반영되지 않도록 idempotency를 보장한다.

### 4. 로그인 성공 후에도 프론트엔드 로그인 상태가 설정되지 않는다

- 위치: `frontend/src/store/authStore.ts:22-24`, `backend/api/routes/auth.py:65-68`, `backend/api/routes/auth.py:147`
- 프론트는 로그인 응답의 `data.user`를 저장하지만 백엔드는 access token만 반환한다.
- `user`가 `undefined`가 되어 로그인 모달이 닫혀도 앱은 정상 로그인 상태가 되지 않는다.
- 새로고침 시 access token 복원 또는 `/auth/me` 호출 흐름도 없다.
- 이 문제는 `docs/frontend-api-integration-guide.md`에도 이미 기록되어 있으나 구현에는 반영되지 않았다.

권고:

- 로그인 후 `/auth/me`를 호출해 사용자 상태를 저장한다.
- 앱 초기화 시 refresh cookie를 이용해 access token과 사용자 정보를 복원한다.
- 동시 401 요청에 대한 refresh mutex를 추가한다.

### 5. 실시간 시세 클라이언트와 서버 프로토콜이 달라 연결이 실패한다

- 위치: `frontend/src/hooks/useStockWebSocket.ts:12-18`, `backend/api/routes/realtime.py:14-34`
- 프론트는 브라우저 WebSocket을 생성하지만 서버 엔드포인트는 SSE `EventSourceResponse`다.
- `VITE_WS_BASE`를 설정하면 WebSocket handshake 자체가 실패한다.
- 미설정 시에는 실제 서버 대신 MockWebSocket만 사용한다.

권고:

- 프론트를 EventSource 또는 fetch 기반 SSE로 변경한다.
- 이름도 `useStockStream` 등 실제 프로토콜에 맞게 변경한다.
- Redis Pub/Sub에서 전달되는 실제 메시지 형식으로 통합 테스트를 추가한다.

### 6. 새 환경에서 백엔드 의존성 설치 및 Docker 이미지 빌드가 실패한다

- 위치: `backend/requirements.txt:22-23`, `backend/Dockerfile:6`
- `pytest==8.2.0`과 `pytest-asyncio>=1.4.0`은 현재 resolver에서 충돌한다. `pytest-asyncio 1.4.0`은 `pytest>=8.4`를 요구한다.
- 실제 `pip install -r backend/requirements.txt` 실행은 `ResolutionImpossible`로 실패했다.
- Dockerfile도 같은 requirements를 설치하므로 깨끗한 Docker 빌드가 동일하게 실패한다.

권고:

- 호환되는 pytest/pytest-asyncio 버전을 고정한다.
- 런타임 의존성과 개발/테스트 의존성을 분리한다.
- CI에서 깨끗한 Docker build를 필수 검사로 추가한다.

## P2 - 높은 우선순위

### 7. 일일 손실 계산과 차단 해제 흐름이 PRD 요구사항을 충족하지 못한다

- 위치: `backend/services/risk_service.py:29-37`, `backend/services/risk_service.py:57-89`, `backend/tasks/email_tasks.py:235-247`, `backend/api/routes/risk.py:14-55`
- 현재 손실은 당일 SELL의 실현손실만 합산한다. PRD는 당일 실현손익과 평가손익 합산을 요구한다.
- 분모는 "어제 포트폴리오 총액"이 아니라 현재 남은 보유종목의 매수원가다.
- 전량 손절 후 보유종목이 0이면 분모가 0이 되어 큰 손실도 차단되지 않는다.
- `trading_blocked=True`가 된 후 사용자가 수동 해제할 API가 없다. PRD는 설정 화면에서 수동 해제를 요구한다.

권고:

- 일별 기준 평가액 snapshot 또는 브로커 잔고 기준으로 손실률을 계산한다.
- 미실현 손익을 포함한다.
- 인증된 명시적 unblock API와 감사 로그를 추가한다.

### 8. 동시 매도 주문으로 보유수량보다 많이 매도할 수 있다

- 위치: `backend/api/routes/trades.py:54-68`, `backend/api/routes/trades.py:86-106`
- 매도 가능 수량 확인과 KIS 주문 사이에 잠금이나 pending sell 예약이 없다.
- 같은 보유수량을 대상으로 요청 두 개가 동시에 들어오면 둘 다 검증을 통과할 수 있다.

권고:

- DB row lock과 pending sell 수량 예약을 적용한다.
- `(보유수량 - 미체결 매도수량) >= 신규 매도수량`을 원자적으로 검증한다.

### 9. 체결 처리 태스크가 중복 실행될 때 포트폴리오를 중복 갱신할 수 있다

- 위치: `backend/tasks/order_tasks.py:44-60`
- 처리 전에 Trade가 이미 `FILLED`인지 확인하지 않는다.
- Celery redelivery, 작업자 재시작, 수동 재실행 시 같은 체결을 다시 포트폴리오에 반영할 수 있다.

권고:

- Trade row를 잠그고 처리 완료 상태 및 마지막 반영 누적 체결량을 확인한다.
- 장부 갱신과 Trade 상태 변경을 하나의 idempotent transaction으로 구성한다.

### 10. `/simulate/download` SSE는 브라우저 EventSource로 인증할 수 없다

- 위치: `backend/api/routes/simulate.py:118-136`, `backend/api/deps.py:12-21`
- 엔드포인트는 Bearer access token을 요구한다.
- 브라우저의 기본 EventSource는 Authorization 헤더를 설정할 수 없다.
- refresh cookie만 보내도 `HTTPBearer`가 access token을 요구하므로 403이 된다.
- 문서의 `new EventSource(..., {withCredentials: true})` 예시는 현재 백엔드 인증 방식에서 동작하지 않는다.

권고:

- fetch streaming을 사용하거나 SSE 전용 단기 토큰 방식을 설계한다.
- 실제 브라우저 기반 인증 SSE 통합 테스트를 추가한다.

### 11. 공휴일에는 현재가 조회가 빈 값이 되어 포트폴리오가 0원으로 보일 수 있다

- 위치: `backend/services/market_service.py:21-28`, `backend/services/market_service.py:85-104`
- `_last_trading_day()`는 주말만 건너뛰고 한국 거래소 휴장일은 처리하지 않는다.
- 평일 공휴일에는 당일 데이터가 비어 `{code}`만 반환된다.
- 포트폴리오는 `close`가 없으면 0으로 계산하고, 시장가 주문 리스크 체크는 현재가 조회 실패로 주문을 거부한다.

권고:

- 실제 데이터가 존재하는 최근 거래일까지 역방향 탐색한다.
- 빈 가격을 0원 평가로 조용히 처리하지 말고 명시적 stale/error 상태를 반환한다.

### 12. 동기 pykrx 호출이 FastAPI event loop를 막는다

- 위치: `backend/services/market_service.py:31-45`, `backend/services/market_service.py:77-89`, `backend/services/market_service.py:108-120`, `backend/services/market_service.py:160-181`
- async 라우트에서 네트워크/파싱을 수행하는 pykrx 동기 함수를 직접 호출한다.
- 캐시 miss 시 한 요청이 처리되는 동안 같은 worker의 다른 요청이 정지할 수 있다.
- 종목 목록 생성은 ticker 이름을 반복 조회하여 특히 오래 걸릴 수 있다.

권고:

- `asyncio.to_thread()` 또는 별도 수집 작업으로 이동한다.
- API 요청 시점이 아니라 주기적 수집으로 캐시를 미리 채운다.

### 13. 프론트 핵심 기능 대부분이 실제 API가 아닌 mock 성공 화면이다

- 위치:
  - `frontend/src/components/Trade/OrderModal.tsx:24-30`
  - `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx:1-6`
  - `frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx:4-15`
  - `frontend/src/components/MainPanel/SimulatorTab/SimulatorTab.tsx:4-22`
  - `frontend/src/components/MainPanel/AITab/AITab.tsx:1-7`
  - `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx:3-25`
  - `frontend/src/store/stockStore.ts:4-20`
- 주문 모달은 API 호출 없이 1.5초 후 "체결됨"을 표시한다.
- 이 UI는 사용자가 실제 주문이 실행됐다고 오해하게 만들 수 있어 특히 위험하다.

권고:

- 실제 연동 전에는 화면에 명확한 DEMO/MOCK 배너를 표시한다.
- 주문 버튼은 실제 API 연동 완료 전 비활성화한다.
- `docs/frontend-api-integration-guide.md`의 계약 오류도 함께 수정하며 순차 연동한다.

### 14. KIS 응답 오류를 정상 빈 결과로 처리하는 경로가 있다

- 위치: `backend/services/kis_service.py:183-204`, `backend/services/kis_service.py:239-255`, `backend/services/kis_service.py:278-327`
- `poll_fill`, `get_balance`, `get_balance_full`은 HTTP 200 응답 후 KIS의 `rt_cd` 오류를 검사하지 않는다.
- 인증 오류나 요청 오류가 미체결 또는 0원 잔고처럼 처리될 수 있다.

권고:

- 모든 KIS 응답에 공통 `_ensure_kis_ok()` 검사를 적용한다.
- 외부 API 오류와 정상 빈 결과를 구분한다.

### 15. 프로세스마다 APScheduler가 실행되어 예약 작업이 중복될 수 있다

- 위치: `backend/main.py:28-44`
- FastAPI lifespan에서 각 프로세스가 스케줄러를 시작한다.
- multi-worker 배포나 여러 backend replica에서는 AI 갱신과 알림 점검이 중복 실행된다.

권고:

- 예약 실행은 단일 Celery Beat 서비스로 분리한다.
- 태스크 자체도 분산 lock/idempotency를 갖도록 한다.

### 16. 알림 설정 일부가 실제 동작에 반영되지 않는다

- 위치: `backend/models/risk.py:32-36`, `backend/tasks/email_tasks.py:44-85`
- `trade_filled=False`여도 체결 이메일을 항상 보낸다.
- `signal_change`, `weekly_report`, `stop_loss_enabled`는 설정 필드는 있으나 실행 로직이 없다.

권고:

- 지원하지 않는 설정은 UI/API에서 숨기거나 구현 상태를 명시한다.
- 모든 알림 발송 전에 해당 설정을 검사한다.

## P3 - 개선 필요

### 17. 프론트 프로덕션 빌드와 lint가 실패한다

- `npm run build` 실패:
  - `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx:37` Recharts formatter 타입 오류
  - `frontend/src/components/Trade/OrderBook.tsx:2` 미사용 import
- `npm run lint` 실패: 8 errors, 1 warning
- 테스트 통과만으로 배포 가능 상태라고 판단할 수 없다.

### 18. 프론트-백엔드 API 타입과 연동 가이드에도 불일치가 남아 있다

- 위치: `frontend/src/types/index.ts`, `docs/frontend-api-integration-guide.md`
- `User.access_allowed`, `Holding.ai_signal`, `OrderRequest.mode`, Portfolio/Risk 타입이 실제 백엔드와 다르다.
- 가이드는 종목 목록을 `{items, total, page}`로 설명하지만 백엔드는 배열만 반환한다.
- 가이드의 차트 interval 예시(`1d`, `1w`, `1mo`)와 백엔드 허용값(`day`, `week`, `month`)도 다르다.

권고:

- OpenAPI에서 TypeScript 타입과 API client를 생성해 계약 드리프트를 줄인다.
- 프론트-백엔드 contract test를 추가한다.

### 19. 백테스트 입력 기간 제한과 비동기 실행이 없다

- 위치: `backend/api/routes/backtest.py:20-43`, `backend/api/routes/backtest.py:63-83`
- PRD는 최대 2년을 요구하지만 기간 제한이 없다.
- 요청 처리 중 전체 백테스트를 동기 대기하므로 긴 기간 요청이 worker 자원을 오래 점유한다.

권고:

- 최대 기간을 검증한다.
- 긴 작업은 Celery job으로 실행하고 상태 조회 API를 제공한다.

### 20. 핵심 AI 기능은 가중치가 없어 기본적으로 비활성 상태다

- 위치: `backend/ml/weights/.gitkeep`, `backend/services/ai_service.py:96-105`
- 저장소에는 학습 가중치가 없으므로 LSTM 예측은 기본적으로 빈 배열이고 시그널은 기술 지표만 사용한다.
- 기능 결함이라기보다 배포 준비 상태의 문제지만, 제품의 핵심 차별점과 직접 관련된다.

권고:

- 모델 artifact 배포/버전/검증 절차를 정의한다.
- 가중치가 없을 때 UI에서 "AI 예측"처럼 표시하지 않도록 상태를 명확히 한다.

## 테스트 및 검증 결과

### 성공

- 프론트 테스트: `25 files`, `103 tests` 통과
- Python AST parse: `82 files`, 문법 오류 0
- Alembic: 단일 head `b8c9d0e1f2a3`
- Git 상태: 기존 미추적 `AGENTS.md` 외 코드 변경 없음

### 실패 또는 미완료

- `pip install -r backend/requirements.txt`: pytest 의존성 충돌로 실패
- 백엔드 전체 pytest:
  - `apscheduler`, `pandas_ta`, `sklearn`, `torch`가 현재 환경에 없어 수집 단계에서 순차적으로 실패
  - requirements 전체 설치는 의존성 충돌로 실패하여 통합 테스트 단계까지 진행하지 못함
- `npm run build`: TypeScript 오류 2건으로 실패
- `npm run lint`: 8 errors, 1 warning으로 실패
- Docker 검증: 로컬에 `docker` 명령이 없어 실행하지 못했으며, requirements 충돌상 Dockerfile의 pip install 단계는 현재 실패할 것으로 판단됨

## 단계별 수정 권장 순서

1. 거래 안전성: SELL 허용 정책, mode 고정, 부분 체결, idempotency, 동시 매도 예약
2. 배포 기준선: requirements 충돌 해결, Docker build, backend 전체 pytest, CI
3. 인증/실시간 통합: 로그인 계약, 세션 복원, SSE 전환, 인증 SSE 방식
4. 프론트 핵심 연동: 주문 mock 제거 또는 비활성화, 종목/차트/포트폴리오 순서로 실제 API 연결
5. 리스크 정확성: 일일 손실 기준액, 평가손익, 수동 unblock
6. 운영 안정성: pykrx 비동기 격리, Celery Beat, KIS 공통 오류 처리
7. 제품 완성도: AI 모델 artifact, 스크리너, 알림 설정 실제 구현

## 최종 판단

코드베이스는 기능별 백엔드 골격과 테스트 자산이 잘 나뉘어 있어 계속 개발하기 좋은 기반은 갖췄다. 다만 현재 테스트는 mock과 단위 경로를 주로 검증하며, 실제 통합 실패와 거래 장부 위험을 잡지 못한다.

현재 단계는 "Phase별 기능 코드가 대부분 존재하는 개발 버전"에 가깝고, "계획대로 완성되어 안전하게 동작하는 통합 버전"은 아니다. 특히 실거래 활성화 전 P1 및 거래 관련 P2 항목을 반드시 해결해야 한다.
