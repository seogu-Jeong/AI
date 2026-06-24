# 260619 hwang DB revise

## 수정 주제

Docker 개발/데모 DB와 테스트 DB 초기화 흐름을 안정화한다.

### 배경

백엔드 전체 테스트가 Docker PostgreSQL 인증 오류로 fixture setup 단계에서 중단되었다. 원인은 코드 assertion 실패가 아니라, 테스트 DB 준비 과정이 수동 절차에 의존했고 `tests/conftest.py`가 SQLAlchemy URL을 문자열화하면서 비밀번호를 `***`로 마스킹한 값을 실제 접속 URL로 사용했기 때문이다.

### 수정 내용

- `docker-compose.yml`에 `db-init` 서비스를 추가했다.
  - PostgreSQL healthcheck 이후 실행된다.
  - `POSTGRES_TEST_DB`를 자동으로 생성한다.
  - 기본 테스트 DB 이름은 `stocksense_test`다.
  - 이미 존재하는 경우 성공으로 처리한다.
- `migrate` 서비스가 `db-init` 성공 이후 실행되도록 순서를 조정했다.
- backend 서비스에 `TEST_DATABASE_URL`을 명시해 외부 shell 환경변수 오염으로 테스트 DB 비밀번호가 잘못 잡히는 문제를 막았다.
- `.env.example`에 `POSTGRES_TEST_DB=stocksense_test`를 추가했다.
- README의 테스트 DB 수동 생성 안내를 제거하고, Docker DB 운영/리셋/테스트 실행법을 새로 정리했다.
- `tests/test_docker_compose_db.py`를 추가해 compose에 테스트 DB 자동 생성 구조가 유지되는지 확인한다.
- `tests/conftest.py`에서 `url.render_as_string(hide_password=False)`를 사용해 테스트 DB URL의 실제 비밀번호를 보존한다.
- `tests/test_test_database_url.py`를 추가해 테스트 DB URL이 `***`로 마스킹되지 않도록 회귀 테스트를 추가했다.

### 검증

먼저 신규 테스트가 실패하는 것을 확인했다.

```bash
docker compose run --rm -v /Users/hwang/Gwang/Class/aiclass/FinalProject:/project -w /project -e PYTHONPATH=/project/backend backend sh -c 'pip install --no-cache-dir pytest==8.4.2 pytest-asyncio==1.4.0 >/tmp/test-deps.log && pytest -q tests/test_docker_compose_db.py'
```

초기 결과: `db-init` 서비스가 없어 실패

수정 후 통과:

```bash
docker compose config --quiet
```

결과: 성공

```bash
docker compose run --rm db-init
```

결과: `Database stocksense_test already exists` 또는 `Created database stocksense_test`

```bash
docker compose run --rm -v /Users/hwang/Gwang/Class/aiclass/FinalProject:/project -w /project -e PYTHONPATH=/project/backend backend sh -c 'pip install --no-cache-dir pytest==8.4.2 pytest-asyncio==1.4.0 >/tmp/test-deps.log && pytest -q tests/test_docker_compose_db.py'
```

결과: `1 passed, 2 warnings`

추가 진단에서 pytest 내부 `TEST_DB_URL`의 비밀번호가 실제 값이 아니라 `***`로 바뀌는 것을 확인했다. SQLAlchemy `URL` 객체의 `str(url)`이 비밀번호를 숨기는 동작이 원인이었고, `render_as_string(hide_password=False)`로 수정했다. backend 서비스에도 `TEST_DATABASE_URL`을 명시해 Docker 테스트가 compose의 `stocksense_test`를 사용하도록 고정했다.

최종 관련 테스트:

```bash
docker compose run --rm -v /Users/hwang/Gwang/Class/aiclass/FinalProject:/project -w /project -e PYTHONPATH=/project/backend backend sh -c 'pip install --no-cache-dir pytest==8.4.2 pytest-asyncio==1.4.0 >/tmp/test-deps.log && pytest -q tests/test_test_database_url.py tests/test_docker_compose_db.py tests/test_system_status.py tests/test_trades.py tests/test_portfolio.py --tb=short'
```

결과: `25 passed, 2 warnings`

### 운영 판단

개발/데모에서는 Docker PostgreSQL volume을 공식 DB로 유지한다. DB 계정/비밀번호를 변경해야 하는 경우 기존 volume에는 자동 반영되지 않으므로, 데이터를 보존할지 초기화할지 먼저 결정해야 한다. 초기화가 가능하면 `docker compose down -v` 후 다시 올리는 방식이 가장 확실하다.
