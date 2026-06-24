# StockSenseAI

AI 기반 주식 분석 플랫폼 (FastAPI + React + PostgreSQL + Redis)

저장소를 처음 clone한 경우 한국투자증권 공식 API 참조 저장소도 초기화합니다.

```bash
git submodule update --init --recursive
```

## 통합 개발 환경 실행

### 1. 환경 변수 설정 (선택)

```bash
cp .env.example .env
# .env 편집 — SENDGRID_API_KEY, GOOGLE_CLIENT_ID/SECRET, ENCRYPTION_KEY 등
```

`.env`가 없어도 개발 기본값으로 실행되며, 외부 연동 기능을 사용할 때 생성하면 됩니다.

ENCRYPTION_KEY 생성:
```bash
python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"
```

### 2. 전체 서비스 실행

```bash
docker compose up --build
```

Compose가 PostgreSQL, Redis, DB migration, FastAPI, Celery worker, Vite frontend를 함께 실행합니다.

- 프론트엔드: http://localhost:5173
- API 문서: http://localhost:8000/docs
- API 상태 확인: http://localhost:8000/health

종료:

```bash
docker compose down
```

데이터 볼륨까지 초기화:

```bash
docker compose down -v
```

## 로컬 개별 실행

Docker 없이 백엔드와 프론트엔드를 따로 실행할 수도 있습니다.

```bash
cd backend
pip install -r requirements-dev.txt
uvicorn main:app --reload

cd frontend
npm install
npm run dev
```

LSTM 모델 학습/예측까지 사용할 때는 백엔드 의존성을 다음처럼 설치합니다.

```bash
pip install -r backend/requirements-ml.txt
```

### 로컬 예측 생성 후 배포 서버 업로드

배포 백엔드는 PyTorch 없이 DB에 저장된 예측을 제공합니다. 모델 학습과 예측 생성은 로컬에서 실행한 뒤 인증 키로 결과만 업로드합니다.

```bash
cd backend
ML_UPLOAD_KEY=<서버와 동일한 키> python -m ml.generate_predictions \
  --codes 005930,000660 \
  --api-url http://localhost:8000
```

서버의 `.env`에도 동일한 `ML_UPLOAD_KEY`를 설정해야 합니다.

## DB 운영

개발/데모 DB는 Docker PostgreSQL을 기준으로 사용합니다. 백엔드나 프론트엔드를 재시작해도 `pgdata` volume이 유지되는 동안 계정, 포트폴리오, 거래 기록은 유지됩니다.

```bash
# 인프라 실행
docker compose up -d postgres redis

# 앱 DB migration + 테스트 DB 자동 생성까지 포함한 전체 실행
docker compose up -d backend frontend
```

`docker-compose.yml`의 `db-init` 서비스가 `POSTGRES_TEST_DB`를 자동 생성합니다. 기본값은 `stocksense_test`입니다. backend 서비스에는 `TEST_DATABASE_URL`도 명시되어 있어, 로컬 shell에 잘못된 `TEST_DATABASE_URL`이 잡혀 있어도 Docker 테스트는 compose의 테스트 DB를 사용합니다. 따라서 테스트용 DB를 수동으로 `CREATE DATABASE` 할 필요가 없습니다.

주의: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`는 PostgreSQL volume이 처음 만들어질 때만 반영됩니다. 이미 생성된 volume이 있는데 `.env`의 DB 계정/비밀번호를 바꾸면 인증 실패가 날 수 있습니다. 이 경우 기존 데이터를 보존하려면 `.env`를 volume 생성 당시 값으로 되돌리고, 초기화해도 괜찮으면 아래 리셋 명령을 사용합니다.

```bash
# 개발 DB 완전 초기화. 계정/거래 기록/포트폴리오 데이터가 모두 삭제됩니다.
docker compose down -v
docker compose up -d postgres redis
docker compose run --rm db-init
docker compose run --rm migrate
```

## 테스트

테스트는 앱 DB가 아니라 `_test` suffix가 붙은 DB만 사용합니다. `TEST_DATABASE_URL`을 직접 지정하지 않으면 `DATABASE_URL`의 DB 이름 뒤에 `_test`를 붙여 사용합니다.

```bash
docker compose run --rm -v "$PWD:/project" -w /project -e PYTHONPATH=/project/backend backend \
  sh -c 'pip install --no-cache-dir pytest==8.4.2 pytest-asyncio==1.4.0 >/tmp/test-deps.log && pytest -q tests'
```

## Celery 워커 (선택)

```bash
cd backend
celery -A tasks worker --loglevel=info
```
