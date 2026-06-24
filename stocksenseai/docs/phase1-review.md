# Phase 1 Review

검토일: 2026-06-02

## 결론

Phase 1 범위인 인증, 기본 시세 API, Docker/PostgreSQL/Redis/Celery, Alembic, Rate Limit, KIS 키 암호화는 대부분 구현되어 있습니다. 승인된 로컬 환경에서 `pytest -q` 결과도 `30 passed`로 확인했습니다.

다만 `progress.md`의 완료 표시는 전반적으로 맞지만, 아래 항목들은 Phase 1 완료 전에 정리하는 것이 좋습니다.

## 수정 필요

1. Google OAuth 로그인은 Refresh Token을 발급하지 않습니다.
   - 위치: `backend/api/routes/auth.py:191`
   - TRD 5.8은 OAuth 콜백 후 Access + Refresh 발급을 요구하지만, 현재 구현은 access token만 프론트로 redirect합니다.
   - `progress.md`의 알려진 제약에는 적혀 있으나, Phase 1 인증 완성도를 기준으로는 OAuth 사용자 세션 유지가 이메일 로그인과 다릅니다.
   - 권고: OAuth 콜백에서도 `RefreshToken` DB row를 만들고 HttpOnly cookie를 설정하세요. 프론트 리다이렉트 구조를 유지하더라도 쿠키는 콜백 응답에서 설정할 수 있습니다.

2. 마이그레이션에 DB server default가 부족합니다.
   - 위치: `backend/models/user.py:16`, `backend/models/user.py:18`, `backend/models/user.py:27`, `backend/models/user.py:49`
   - 모델은 Python-side `default`를 사용하지만, 마이그레이션은 `is_verified`, `mode`, `dark_mode`, `revoked`에 DB 기본값을 만들지 않습니다.
   - ORM 경유 삽입은 괜찮지만 SQL 직접 삽입, 관리 도구, 다른 서비스 연동 시 NOT NULL 컬럼 누락으로 실패할 수 있습니다.
   - 권고: 새 Alembic 리비전으로 `server_default=false`, `server_default='demo'`, `server_default=true` 등을 명시하세요.

3. Refresh Token selector가 nullable입니다.
   - 위치: `backend/models/user.py:47`, `db/migrations/versions/b465250cc7d4_add_refresh_token_selector.py:24`
   - 현재 코드가 새 토큰에는 selector를 넣지만 DB 제약상 null을 허용합니다. Phase 1 문서의 O(1) 조회 구조와 맞추려면 selector는 필수값이어야 합니다.
   - 권고: 기존 데이터 backfill 전략을 둔 뒤 `nullable=False`로 변경하세요.

4. 테스트 재현성이 로컬 PostgreSQL에 강하게 의존합니다.
   - 위치: `tests/conftest.py:8`
   - 샌드박스 기본 실행에서는 로컬 5432 접속 제한으로 DB 의존 테스트 20개가 실패했습니다. 승인된 환경에서는 통과했습니다.
   - 권고: README 또는 docs에 `docker-compose up -d postgres redis`, `stocksense_test` DB 생성, 테스트 실행 순서를 명시하세요. 가능하면 Testcontainers나 SQLite 대역은 피하더라도, test DB 자동 생성 스크립트를 두는 편이 좋습니다.

5. Git 작업트리에 문서 삭제가 남아 있습니다.
   - 현재 상태: 루트 `2026-05-31-stocksenseai-PRD.md`, `2026-05-31-stocksenseai-TRD.md`가 삭제 상태입니다.
   - 동일 문서가 `docs/superpowers/specs/`에 있어 의도된 이동일 수 있지만, 삭제가 커밋될 경우 참조 링크나 팀원 워크플로가 깨질 수 있습니다.
   - 권고: 루트 문서 삭제가 의도인지 확인하고, 의도된 이동이면 README/CLAUDE의 참조를 docs 경로로 통일하세요.

## 권고 사항

1. 비밀번호 정책을 추가하세요.
   - 위치: `backend/api/routes/auth.py:41`
   - 현재 `password: str`만 있어 짧거나 약한 비밀번호도 가입 가능합니다.
   - 권고: 최소 길이 8~12자, 공백 제한 정도는 Pydantic `Field` 또는 validator로 처리하세요.

2. KIS 계좌번호 형식 검증을 추가하세요.
   - 위치: `backend/api/routes/auth.py:60`
   - 현재 `account_no: str`만 검증합니다.
   - 권고: 한국투자증권 계좌번호 입력 형식이 정해져 있다면 정규식과 명확한 에러 메시지를 넣으세요.

3. 시세 API 에러 처리가 부족합니다.
   - 위치: `backend/services/market_service.py:23`, `backend/services/market_service.py:68`, `backend/services/market_service.py:95`
   - pykrx 호출 실패, KRX 휴장일, 빈 데이터가 API 응답에서 구분되지 않습니다.
   - 권고: 외부 호출 실패는 502/503 계열로 매핑하고, 빈 데이터는 명시적인 빈 결과 또는 404 정책을 정하세요.

4. 현재가 조회가 휴장일/장 시작 전에는 빈 응답이 될 수 있습니다.
   - 위치: `backend/services/market_service.py:67`
   - 오늘 날짜만 조회하므로 주말, 휴장일, 장 시작 전에는 `{"code": code}`만 반환될 가능성이 있습니다.
   - 권고: 최근 영업일 fallback을 추가하세요.

5. `get_market_ticker_name` 반복 호출 캐싱이 필요합니다.
   - 위치: `backend/services/market_service.py:100`, `backend/services/market_service.py:120`
   - 종목 목록과 검색에서 ticker name 조회가 반복됩니다.
   - 권고: ticker list 캐시에 code만 저장하지 말고 `{code, name}` 형태로 저장하세요.

6. 시장 시간 계산에 timezone을 명시하세요.
   - 위치: `backend/services/market_service.py:9`
   - 서버 timezone에 의존합니다. 현재 사용자는 Asia/Seoul 환경이지만 Docker/서버 배포 시 달라질 수 있습니다.
   - 권고: `zoneinfo.ZoneInfo("Asia/Seoul")` 기준으로 장중 여부를 계산하세요.

7. AES 키 설정 검증을 명확히 하세요.
   - 위치: `backend/core/security.py:15`
   - 잘못된 base64 또는 32바이트가 아닌 키는 import 시점에 앱이 죽습니다. 실패 자체는 맞지만 에러 메시지가 설정 문제를 설명하지 못합니다.
   - 권고: `Settings` validator에서 32바이트 base64인지 검증하고 명확한 메시지를 내세요.

8. Docker Compose만으로 DB 마이그레이션이 자동 실행되지 않습니다.
   - 위치: `docker-compose.yml:24`
   - backend 서비스는 uvicorn만 실행합니다. 새 DB에서는 테이블이 없으므로 API가 바로 실패할 수 있습니다.
   - 권고: 개발 문서에 `alembic upgrade head`를 명시하거나 entrypoint에서 마이그레이션 실행 여부를 선택하게 하세요.

9. README가 실행 문서 역할을 하지 못합니다.
   - 위치: `README.md:1`
   - 현재 README는 제목만 있습니다.
   - 권고: `.env` 생성, Docker 실행, 마이그레이션, 테스트, API 문서 URL을 최소한으로 추가하세요.

10. 테스트 커버리지 표기는 "통합 전체"로 보기에는 과합니다.
    - 위치: `docs/progress.md:102`
    - stocks 테스트는 pykrx/Redis 실제 연동보다 라우터와 mock 중심입니다.
    - 권고: progress에는 "라우터/서비스 mock 기반 통합"처럼 정확히 적고, 별도 실제 Redis/pykrx smoke test를 추가하세요.

## 확인한 점

- `pytest -q`: 30 passed, 1 warning
- `/auth` 기본 플로우: register, verify, login, refresh, logout, me 구현 확인
- `/stocks` 기본 플로우: list, search, indices, chart, detail 구현 확인
- Alembic async env 구성 확인
- Redis lazy singleton 및 lifespan close 확인
- Rate limit middleware와 login 5/min 제한 확인

