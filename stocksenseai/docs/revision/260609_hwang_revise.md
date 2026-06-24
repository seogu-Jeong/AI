# 2026-06-09 수정 내역 — hygrenn (백엔드/인프라)

---

## 1. 신규 추가

### `backend/services/kis_account_service.py`
KIS 잔고 조회 서비스 신규 작성.
- `.env`의 `SYSTEM_KIS_APP_KEY` / `SYSTEM_KIS_APP_SECRET` / `SYSTEM_KIS_ACCOUNT_NO` 사용
- `.env`의 `SYSTEM_KIS_MODE`에 설정된 단일 계좌 모드로 잔고 조회
- KIS `inquire-balance` API 호출 → `summary` + `holdings` 반환
- 계좌번호 마스킹 (`1234****-01`)
- `output2`: 총자산·예수금·평가금·매입금·평가손익·수익률
- `output1`: 종목별 수량·평균단가·현재가·평가금액·손익·수익률

### `backend/api/routes/account.py`
계좌 조회 라우터 신규 작성.
- `GET /account/balance` — rate limit 30/minute, 로그인 필수
- `GET /account/config` — 현재 시스템 모드와 마스킹된 계좌번호 반환
- 요청에서 모드를 선택하지 않고 서버에 설정된 계좌만 조회

### `frontend/src/components/Account/AccountPanel.tsx`
우측 슬라이드 계좌 패널 신규 작성.
- 헤더 유저 버튼 클릭 시 오픈, 배경 클릭 or X 버튼으로 닫기
- 총자산 / 예수금·평가금·매입금 3분할 요약
- 보유 종목 목록 (수익+/손실- 색상 구분, TrendingUp/Down 아이콘)
- 새로고침 버튼, 로딩/에러/미보유 상태 처리
- 현재 시스템 계좌가 실계좌인지 모의계좌인지 안내
- 조회 탭에서 주문 모드를 변경하지 않도록 구성

---

## 2. 수정

### `backend/main.py`
- `account_router` import 및 `/account` prefix로 등록

### `backend/core/config.py`
- `SYSTEM_KIS_ACCOUNT_NO: str = ""` 설정 추가

### `frontend/src/components/Layout/Header.tsx`
- 로그인 상태에서 이메일 `<span>` → 클릭 가능한 `<Button variant="ghost">` 로 변경
- `onAccountClick?: () => void` prop 추가

### `frontend/src/components/Layout/MainLayout.tsx`
- `accountOpen` state 추가
- `AccountPanel` import 및 조건부 렌더링 (backdrop + panel)
- `Header`에 `onAccountClick` prop 전달

### `.env.example`
- `SYSTEM_KIS_ACCOUNT_NO` 항목 추가 (형식: `12345678-01`)

### `CLAUDE.md`
- KIS API 레퍼런스 섹션 전면 재작성
  - `reference/` → `references/open-trading-api/examples_llm/` 로 경로 수정
  - 작업별 읽어야 할 파일 표로 정리
  - 실제 개발 중 발견한 gotcha 5개 추가 (모의투자 서버 제한, rate limit, 페이지네이션, hashkey, 잔고조회 TR ID)

---

## 3. 설정 필요 (`.env`)

잔고 조회 기능 사용을 위해 `.env`에 아래 항목 추가 필요:

```
SYSTEM_KIS_ACCOUNT_NO=12345678-01   # 본인 KIS 계좌번호
```

`SYSTEM_KIS_APP_KEY`, `SYSTEM_KIS_APP_SECRET`, `SYSTEM_KIS_MODE` 는 기존에 설정되어 있어야 함.

---

## 4. 단일 계좌 운용 원칙

- 로컬 프로그램에서 KIS 계좌 한 개만 `.env`에 등록하여 사용
- `SYSTEM_KIS_MODE=paper|real`이 잔고 조회, 주문, 주문 기록, 포트폴리오의 단일 기준
- 화면이나 API 요청 파라미터로 주문 모드를 임의 전환하지 않음
- 실계좌/모의계좌 전환이 필요하면 `.env`의 `SYSTEM_KIS_MODE`를 변경한 뒤 백엔드 재시작

---

## 5. 매수/매도 시스템 키 통일 (추가 수정)

### 문제
매수/매도 버튼 클릭 시 "KIS 키를 먼저 등록하세요" 오류 발생.
`kis_service.py`의 모든 함수가 DB에 저장된 퍼유저 키를 조회하고 있었음.
`trades.py`는 `user.mode == "demo"` 이면 무조건 차단.

### 수정 파일

**`backend/services/kis_service.py`**
- `_get_keys()`: DB 퍼유저 키 조회 → `.env` `SYSTEM_KIS_*` 키 반환으로 전환
- `_effective_mode()` 추가: 항상 `settings.SYSTEM_KIS_MODE` 반환
- `_headers()`: 시스템 키 + 시스템 모드 사용
- `place_order`, `cancel_order`, `poll_fill`, `get_balance`, `get_balance_full`: `user.mode` → `_effective_mode()` 사용
- `from core.security import decrypt_aes` 제거 → `from core.config import settings` 로 교체

**`backend/api/routes/trades.py`**
- `user.mode == "demo"` 차단 제거
- → `.env` 시스템 키 미설정 시 503 반환으로 교체
- `from core.config import settings` import 추가

---

## 6. 백엔드 재시작 명령

```bash
docker compose up -d backend
```

---

## 7. 단일 로컬 계좌 모드 정합성 수정

- `.env`에는 KIS 계좌 한 개만 등록하며 `SYSTEM_KIS_MODE`를 실제 주문·잔고 모드의 단일 기준으로 사용
- `GET /account/config` 추가: 현재 시스템 모드와 마스킹 계좌번호 반환
- `/account/balance`의 `mode` 선택 제거: 등록된 시스템 계좌만 조회
- 주문 기록, 체결 폴링, 기본 주문 목록, 취소, 리스크 검사, 포트폴리오 조회를 시스템 모드 기준으로 통일
- 계좌 패널과 주문창에 현재 모드 및 마스킹 계좌번호 표시
- 실계좌 주문은 첫 클릭에서 경고를 표시하고 두 번째 확인 후 실행
- KIS 시스템 키가 없으면 주문 API가 명확한 503 오류 반환

## 8. 분봉 로딩 UX 및 프론트 빌드 수정

- 분봉 조회를 백그라운드 task로 실행하여 첫 응답을 오래 대기하지 않도록 수정
- 분봉 준비 중 일봉 데이터를 임시 표시하고 `loading_intraday` 상태 반환
- 프론트가 3초 간격으로 재조회하여 준비 완료 시 분봉으로 자동 교체
- 분봉 로딩 중임을 차트 상단에 표시
- KIS 키가 없으면 `fallback_only` 상태와 일봉만 반환하여 무한 재조회 방지
- 응답에 `requested_interval`, `actual_interval`, `status`를 포함하여 실제 표시 중인 데이터 구분
- 분봉 차트에서는 일봉 기준 LSTM 예측 오버레이를 숨김
- 숫자형 분봉 타임스탬프에 맞춰 RSI/MACD 타입 수정
- 프론트 TypeScript 빌드 오류 수정

## 9. 인증 토큰 갱신 반복 요청 수정

### 문제

비로그인 상태에서 `/auth/refresh`가 401을 반환하면 공통 Axios 인터셉터가 다시 refresh를 요청할 수 있었다.
이 동작이 반복되면 백엔드 rate limit에 도달하여 429 응답이 발생했다.

### 수정

- `/auth/*` 요청에서 발생한 401은 공통 인터셉터가 refresh하지 않도록 제한
- 현재 access token이 없는 비로그인 상태의 401은 refresh하지 않도록 제한
- access token이 존재하는 일반 API 요청의 첫 번째 401만 refresh 후 재시도
- `shouldAttemptTokenRefresh` 회귀 테스트 추가

## 10. 테스트 및 Docker 통합 검증

2026-06-09 최종 검증 결과:

- 백엔드 전체 테스트: `130 passed, 1 skipped`
- 프론트 전체 테스트: `107 passed`
- 프론트 ESLint: 통과
- 프론트 production build: 통과
- backend/frontend Docker 이미지 빌드: 통과
- Docker Compose 통합 기동: backend, frontend, PostgreSQL, Redis, Celery 정상 실행
- 백엔드 health check: `GET /health` → `200 {"status":"ok"}`
- OpenAPI에서 `/account/config`, `/account/balance` 경로 노출 확인
- `git diff --check`: 통과

### 남은 비차단 경고

- 프론트 production build에서 메인 JS chunk가 500 kB를 초과한다는 경고가 있음
- 일부 기존 프론트 테스트에서 React `act(...)` 경고 및 의도된 네트워크 실패 로그가 출력되지만 테스트는 통과함

---

현재 변경사항은 `hwang` 브랜치 작업 트리에만 있으며 아직 커밋하거나 push하지 않았다.
