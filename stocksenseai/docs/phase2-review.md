# Phase 2 Review

검토일: 2026-06-03

## 결론

Phase 2의 REST 기반 호가·체결·분봉, SSE 엔드포인트, KIS access_token/approval_key 캐시, 기본 WebSocket 체결 메시지 파싱과 Redis publish는 구현되어 있습니다. `pytest -q` 결과도 `52 passed`로 확인했습니다.

다만 `docs/superpowers/plans/2026-06-03-phase2-realtime.md`의 목표를 엄밀히 보면 “KIS WebSocket Pool로 실시간 체결/호가를 Redis Pub/Sub에 발행”하는 범위까지는 아직 완성이라고 보기 어렵습니다. 현재 WebSocket은 체결(`H0STCNT0`)만 구독·파싱하고, 호가(`H0STASP0`) 실시간 처리는 없습니다.

## 수정 필요

1. WebSocket 실시간 호가(`H0STASP0`)가 구현되어 있지 않습니다.
   - 위치: `backend/services/websocket_service.py:25`, `backend/services/websocket_service.py:85`
   - 계획서의 Phase 2 목표와 TR ID 표에는 WebSocket 체결(`H0STCNT0`)과 호가(`H0STASP0`)가 모두 포함되어 있습니다.
   - 현재 `_parse_execution_msg`는 `H0STCNT0`만 받고, subscribe 메시지도 `tr_id: H0STCNT0`만 전송합니다.
   - 권고: `_parse_orderbook_msg`를 추가하고 `stock:{code}` 또는 `stock:{code}:orderbook` 채널 발행 정책을 정하세요. SSE에서 체결/호가 이벤트를 분리할지도 명확히 해야 합니다.

2. WebSocket 세션 배치 계산이 41종목 제한과 정확히 맞지 않습니다.
   - 위치: `backend/services/websocket_service.py:91`
   - `needed = (subscribed_count // MAX_PER_SESSION) + 1`라서 41번째 종목에서 세션이 2개 필요하다고 계산됩니다. 41개까지는 1개 세션이어야 합니다.
   - 권고: `needed = max(1, math.ceil(subscribed_count / MAX_PER_SESSION))` 형태로 바꾸세요.

3. 구독 해제 메시지가 원래 구독한 세션으로 가지 않을 수 있습니다.
   - 위치: `backend/services/websocket_service.py:51`, `backend/services/websocket_service.py:70`, `backend/services/websocket_service.py:87`
   - `_symbol_session` 딕셔너리가 있지만 사용되지 않습니다. `_send_subscribe`는 항상 `_get_session()`의 마지막 active session으로 메시지를 보냅니다.
   - 여러 세션이 생긴 뒤 오래된 종목을 unsubscribe하면 다른 세션으로 해제 메시지를 보낼 수 있습니다.
   - 권고: 최초 subscribe 시 code → session index/object를 저장하고, unsubscribe는 해당 세션에 보내세요.

4. WebSocket Pool에 `start()`가 없습니다.
   - 위치: `backend/services/websocket_service.py:126`, `backend/main.py:19`
   - 계획서에는 `start()` / `stop()`을 lifespan에서 호출한다고 되어 있지만 현재 lifespan은 shutdown 시 `stop()`만 호출합니다.
   - lazy subscribe 방식 자체는 가능하지만, 문서와 구현이 다릅니다.
   - 권고: lazy 방식으로 갈 거면 계획/progress를 수정하세요. 아니면 `start()`에서 approval_key 사전 준비나 연결 상태 초기화를 수행하세요.

5. SSE 연결 종료 시 pubsub close가 없습니다.
   - 위치: `backend/api/routes/realtime.py:25`
   - `unsubscribe`만 호출하고 pubsub connection close/reset 처리가 없습니다. Redis 클라이언트 구현에 따라 장기 운영 시 리소스가 남을 수 있습니다.
   - 권고: `finally`에서 `await pubsub.close()` 또는 사용 중인 redis 버전에 맞는 `aclose()`를 호출하세요.

6. SSE 엔드포인트에 rate limit 또는 code validation이 없습니다.
   - 위치: `backend/api/routes/realtime.py:13`
   - `/stocks` 엔드포인트는 slowapi 제한이 있지만 `/ws/stocks/{code}`는 제한이 없습니다. 임의 code로 무제한 SSE 연결을 열 수 있습니다.
   - 권고: code 형식 검증(`^[0-9A-Z]{6,12}$` 등)과 연결 수/rate limit 정책을 추가하세요.

7. KIS token cache key가 app key prefix 8자만 사용합니다.
   - 위치: `backend/services/kis_token_service.py:17`, `backend/services/kis_token_service.py:43`
   - 서로 다른 키가 같은 prefix를 공유하면 토큰/approval_key 캐시 충돌 가능성이 있습니다.
   - 권고: 전체 키를 저장하지 말고 `sha256(app_key).hexdigest()[:16]` 같은 digest를 사용하세요.

8. KIS REST 응답의 API-level 실패를 확인하지 않습니다.
   - 위치: `backend/services/kis_market_service.py:53`, `backend/services/kis_market_service.py:93`, `backend/services/kis_market_service.py:147`
   - HTTP 200이어도 KIS 응답의 `rt_cd`, `msg_cd`, `msg1`에서 실패가 올 수 있습니다. 현재는 `output1/output2`가 없으면 빈 값처럼 파싱됩니다.
   - 권고: 공통 `_ensure_kis_ok(resp.json())`를 두고 `rt_cd != "0"`이면 502로 매핑하세요.

## 권고 사항

1. 분봉 API의 시간 파라미터 정책을 명확히 하세요.
   - 위치: `backend/services/kis_market_service.py:142`
   - `FID_INPUT_HOUR_1`이 항상 `"090000"`으로 고정되어 있습니다.
   - 권고: KIS API가 이 값을 “조회 시작 시각”으로 해석하는지 “기준 시각”으로 해석하는지 레퍼런스와 실키로 확인하세요. 필요하면 현재 시각 기준으로 바꾸거나 query parameter로 받으세요.

2. 분봉 캐시 TTL이 장외 1시간입니다.
   - 위치: `backend/services/kis_market_service.py:164`
   - 계획의 일봉 차트 장외 24시간과 달리 분봉은 장외 3600초입니다. 의도된 차이면 괜찮지만 progress에는 캐싱 정책이 구분되어 있지 않습니다.
   - 권고: progress 또는 설계 문서에 REST orderbook/trades/intraday TTL을 명시하세요.

3. SSE 테스트는 실제 Redis Pub/Sub 장시간 동작을 보장하지 않습니다.
   - 위치: `tests/test_phase2.py:343`
   - 현재 테스트는 mock `listen()`이 한 메시지를 yield하는 수준입니다.
   - 권고: Redis가 떠 있는 환경에서 subscribe → publish → stream 수신을 확인하는 smoke test를 별도 marker로 추가하세요.

4. WebSocket Pool 테스트가 41개 초과 배치와 unsubscribe 라우팅을 검증하지 않습니다.
   - 위치: `tests/test_phase2.py:280`
   - 현재는 단일 종목 count만 봅니다. 그래서 위 2, 3번 문제가 테스트로 잡히지 않습니다.
   - 권고: 41개, 42개, 82개 구독 시 세션 수와 각 code의 session mapping을 검증하는 테스트를 추가하세요.

5. progress의 “KIS WebSocket Pool (41종목 배치, Redis Pub/Sub)” 완료 표기는 과합니다.
   - 위치: `docs/progress.md:121`
   - 체결 메시지 publish는 구현됐지만 41종목 배치 정확성, 호가 WebSocket, 세션별 unsubscribe가 빠져 있습니다.
   - 권고: 현재 상태를 “체결 WebSocket 기본 파싱 + Redis publish”로 낮춰 쓰거나 위 수정 후 완료로 유지하세요.

6. Docker/운영 문서에 SYSTEM_KIS 설정과 실시간 확인 절차를 추가하세요.
   - 위치: `.env.example:25`, `README.md:1`
   - `.env.example`에는 변수가 있지만 README가 여전히 실행 절차를 설명하지 않습니다.
   - 권고: KIS 키 없는 모드, KIS 키 있는 모드, `curl -N /ws/stocks/{code}` 확인 절차를 README에 넣으세요.

## 확인한 점

- `pytest -q`: 52 passed, 1 warning
- `backend/services/kis_token_service.py`: access_token, approval_key Redis cache 구현 확인
- `backend/services/kis_market_service.py`: REST orderbook, recent trades, intraday OHLCV 구현 확인
- `backend/api/routes/stocks.py`: `orderbook`, `trades`, intraday chart 라우팅 확인
- `backend/api/routes/realtime.py`: Redis Pub/Sub 기반 SSE 스트림 확인
- `backend/main.py`: realtime router 등록 및 shutdown 시 `kis_pool.stop()` 호출 확인
- `.env.example`: `SYSTEM_KIS_*` 변수 추가 확인

