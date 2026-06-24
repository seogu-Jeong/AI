# 통합 개발용 Docker Compose 구현 계획

**Goal:** `docker compose up --build`로 전체 개발 스택을 실행한다.

## Task 1: Compose 네트워크 환경 정리

- [x] 백엔드와 Celery의 DB/Redis 주소를 Compose 서비스명으로 덮어쓴다.
- [x] Celery에 백엔드 소스 bind mount를 추가한다.
- [x] 일회성 Alembic migration 서비스와 시작 순서를 추가한다.

## Task 2: 프론트엔드 컨테이너 추가

- [x] `frontend/Dockerfile.dev`와 `.dockerignore`를 추가한다.
- [x] Compose에 Vite frontend 서비스와 node_modules named volume을 추가한다.

## Task 3: 빌드 가능성 및 사용법 정리

- [x] 충돌하는 pytest/pytest-asyncio 버전을 호환 가능한 조합으로 고정한다.
- [x] PyTorch를 선택적 ML 의존성으로 분리한다.
- [x] 루트 환경 변수 예시와 README의 통합 실행 절차를 갱신한다.

## Task 4: 검증

- [x] Compose YAML 구조를 정적으로 검증한다.
- [x] 프론트엔드 테스트와 빌드를 실행한다.
- [x] 변경 diff를 점검한다.

> `docker compose up --build` 실제 기동, migration, health/API/frontend 응답을 검증했다.
