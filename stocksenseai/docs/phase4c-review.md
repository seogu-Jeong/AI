# Phase 4-C Review

검토일: 2026-06-04

## 결론

Phase 4-C의 핵심 구성인 `price_cache` 모델/마이그레이션, `simulator_service`, `/simulate` 라우터 4개, 테스트 파일, main 라우터 등록은 구현되어 있습니다. 계획의 TDD 테스트도 현재 환경에서 `8 passed`로 재현됩니다.

다만 “KOSPI 대형주 80개 가격 캐시 + 일시불/적립식 수익률 계산 API”라는 목표 기준으로는 몇 가지 정확성 문제가 남아 있습니다. 특히 고정 종목 수가 80개가 아니고, 캐시가 일부만 있을 때 누락 구간을 다운로드하지 않으며, 적립식 차트가 월별 매수일만 보여줘 누적 투자액 대비 평가액 시계열로 보기 어렵습니다.

## 수정 필요

1. `SIMULATOR_TICKERS`가 계획의 80개가 아니라 79개입니다.
   - 위치: `backend/services/simulator_service.py:15`
   - 계획과 progress는 KOSPI 대형주 80개를 전제로 하지만 실제 리스트를 AST로 세어보면 79개입니다. `/simulate/download`의 total과 `/simulate/data-status` ready 기준도 79개 기준으로 동작합니다.
   - 권고: 의도한 80번째 종목을 추가하거나 문서/progress를 79개로 수정하세요.

2. 가격 캐시가 일부만 존재하면 누락 기간을 보충하지 않습니다.
   - 위치: `backend/services/simulator_service.py:116`, `backend/services/simulator_service.py:129`
   - `get_prices()`는 조회 결과가 하나라도 있으면 pykrx 다운로드를 건너뜁니다. 예를 들어 2024년 데이터만 캐시에 있고 2020~2026 기간을 요청하면, 캐시된 일부 날짜만으로 시뮬레이션합니다.
   - 권고: 요청 기간의 시작/종료 거래일 커버리지를 확인하고, 부족한 구간은 pykrx로 보충하세요. 단순히 `not cached`가 아니라 “범위가 충분한지”를 봐야 합니다.

3. 캐시 저장이 UPSERT가 아니라 중복 저장 시 IntegrityError가 날 수 있습니다.
   - 위치: `backend/services/simulator_service.py:144`, `backend/services/simulator_service.py:154`
   - 현재는 `PriceCache` 객체를 `add_all()` 후 commit합니다. 동시 `/download` 요청이나 일부 중복 데이터 다운로드가 있으면 복합 PK 충돌 가능성이 있습니다.
   - 권고: PostgreSQL `ON CONFLICT DO UPDATE/NOTHING`을 사용하거나, 날짜별 기존 row를 제외하고 insert하세요.

4. `/download`에서 한 종목 실패 시 전체 SSE 다운로드가 중단됩니다.
   - 위치: `backend/services/simulator_service.py:357`, `backend/api/routes/simulate.py:121`
   - `download_tickers()`가 `get_prices()` 예외를 잡지 않습니다. pykrx가 특정 종목만 실패해도 generator 전체가 끊기고 complete 이벤트도 나가지 않습니다.
   - 권고: 종목별 try/except로 `status: "error"` progress를 보내고 다음 종목으로 진행하세요. 마지막 complete에는 성공/실패 개수를 포함하세요.

5. 적립식 `chart_data`가 “누적 투자액 vs 평가액 시리즈” 역할을 충분히 하지 못합니다.
   - 위치: `backend/services/simulator_service.py:281`
   - 차트 데이터는 매월 첫 매수일에만 추가됩니다. 계획은 누적 투자액 대비 평가액 시리즈인데, 현재는 월별 매수 직후 가격 기준 value만 있고 종료일까지의 변동이나 최종 평가 시점이 포함되지 않을 수 있습니다.
   - 권고: 전체 거래일 기준으로 `{date, invested, value, return_pct}`를 만들거나 최소한 마지막 거래일의 최종 평가 포인트를 추가하세요.

6. 적립식 요청의 실제 종료 거래일이 응답에 없습니다.
   - 위치: `backend/services/simulator_service.py:290`, `backend/services/simulator_service.py:296`
   - 계산은 `_find_nearest(..., "backward")`로 실제 종료 거래일을 찾지만 응답에는 `end_date_actual`이 없습니다. 일시불은 실제 매수/매도일을 반환하므로 API 일관성이 떨어집니다.
   - 권고: `start_date_actual`, `end_date_actual`을 recurring 응답에도 포함하세요.

7. 입력 종목이 `SIMULATOR_TICKERS`에 포함되는지 검증하지 않습니다.
   - 위치: `backend/api/routes/simulate.py:26`, `backend/services/simulator_service.py:98`
   - 라우터는 6자리 숫자만 확인합니다. 계획은 고정 80개 대형주 캐시인데, 임의 6자리 코드를 넣으면 pykrx 조회를 시도하고 이름은 ticker로 fallback됩니다.
   - 권고: “고정 종목만 허용”이 목표라면 라우터에서 whitelist 검증을 하세요. 임의 종목 허용이 목표라면 문서를 수정하고 data-status 의미를 바꾸세요.

8. 시뮬레이터 계산에 수수료, 세금, 배당, 현금 잔액 정책이 없습니다.
   - 위치: `backend/services/simulator_service.py:223`, `backend/services/simulator_service.py:276`
   - 현재는 정수 주식만 사고 남은 현금은 수익률 계산에서 제외합니다. 계획 예시와 테스트에는 맞지만, 실제 “내가 그때 샀다면” UX에서는 남은 현금 포함 여부가 중요합니다.
   - 권고: 최소한 응답에 `cash_left_krw`를 포함하고, 수익률 기준이 “투입되어 실제 매수된 금액 기준”인지 “입금 원금 전체 기준”인지 명확히 하세요.

## 권고 사항

1. `data-status`는 단순 종목 수만 보지 말고 기간 커버리지를 포함하세요.
   - 위치: `backend/services/simulator_service.py:333`
   - 현재는 종목별 데이터가 하루씩만 있어도 ticker_count 기준으로 ready가 true가 될 수 있습니다.
   - 권고: `min_trade_date`, `max_trade_date`, `missing_tickers`, `required_count`를 반환하세요.

2. 날짜 검증에 미래 날짜 제한을 추가하세요.
   - 위치: `backend/api/routes/simulate.py:20`, `backend/api/routes/simulate.py:43`
   - 미래 날짜 요청은 pykrx 조회 실패 또는 빈 데이터로 이어집니다. 사용자 경험상 422로 먼저 막는 편이 낫습니다.

3. 응답 스키마를 Pydantic response model로 고정하세요.
   - 위치: `backend/api/routes/simulate.py:66`
   - 현재는 dict를 그대로 반환합니다. 프론트가 chart_data 필드 구조를 안정적으로 쓰려면 response model이 있는 편이 안전합니다.

4. 통합 테스트가 서비스 mock 중심이라 캐시/pykrx 저장 경로 검증이 부족합니다.
   - 위치: `tests/test_simulate.py:137`, `tests/test_simulate.py:187`, `tests/test_simulate.py:216`
   - 라우터 테스트는 정상 응답 모양을 확인하지만, `get_prices()`가 DB 캐시를 읽고 저장하는 경로는 직접 검증하지 않습니다.
   - 권고: pykrx를 mock하고 실제 test DB에 `price_cache` row가 저장되는 테스트, 일부 캐시 누락 시 보충되는 테스트를 추가하세요.

5. 다운로드 작업은 장기 실행 작업이므로 재진입/중복 실행 방어가 필요합니다.
   - 위치: `backend/api/routes/simulate.py:114`
   - rate limit 3/min만으로는 같은 사용자가 여러 SSE 다운로드를 동시에 열 수 있습니다.
   - 권고: Redis lock 또는 DB job 상태로 한 번에 하나의 다운로드만 허용하세요.

## 확인한 점

- `pytest -q tests/test_simulate.py`: 8 passed
- `backend/main.py`에 `/simulate` 라우터 등록 확인
- `price_cache` 모델과 Alembic 마이그레이션의 복합 PK 구조 확인
- `SIMULATOR_TICKERS` 실제 개수: 79개
- Phase 3 리뷰 파일은 이번 요청 전환 후 수정하지 않았습니다.
