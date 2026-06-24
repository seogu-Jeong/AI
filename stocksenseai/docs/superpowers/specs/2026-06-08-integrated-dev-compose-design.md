# 통합 개발용 Docker Compose 설계

**작성일:** 2026-06-08
**담당:** hygrenn

## 목표

프로젝트 루트에서 `docker compose up --build` 한 번으로 PostgreSQL, Redis,
DB migration, FastAPI, Celery worker, Vite frontend를 함께 실행한다.

## 결정 사항

- 개발 환경용 Compose로 구성하고 백엔드와 프론트엔드 소스를 bind mount한다.
- 브라우저가 접근하는 API 주소는 `http://localhost:8000`을 사용한다.
- 컨테이너 내부 백엔드와 Celery는 `postgres`, `redis` 서비스명을 사용한다.
- 프론트엔드 의존성은 named volume에 보관해 호스트의 `node_modules`와 분리한다.
- 프론트엔드는 `Dockerfile.dev`에서 `npm ci`로 재현 가능하게 설치한다.
- `.env`는 선택 사항이며, 없으면 애플리케이션의 개발 기본값을 사용한다.
- 일회성 `migrate` 서비스가 Alembic migration을 적용한 뒤 애플리케이션 서비스를 시작한다.
- PyTorch는 LSTM 모델용 선택 의존성으로 분리해 기본 개발 이미지에서는 설치하지 않는다.

## 실행 흐름

1. PostgreSQL healthcheck 통과 및 Redis 시작
2. Alembic migration 적용
3. FastAPI와 Celery worker 시작
4. Vite 개발 서버 시작 및 호스트의 5173 포트에 노출

## 범위 밖

- 운영 배포용 Nginx/정적 빌드 이미지
- Docker Desktop 설치
