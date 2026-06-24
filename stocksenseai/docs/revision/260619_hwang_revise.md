# 260619 hwang revise

## 수정 계획

1. 포트폴리오 탭의 보유 현황은 앱 DB가 아니라 KIS 계좌 잔고를 우선 원본으로 사용한다.
2. 실계좌/모의계좌 전환은 설정의 `SYSTEM_KIS_MODE`만 따르고, 포트폴리오 조회 화면에서 주문 모드를 바꾸지 않는다.
3. KIS 조회가 실패하는 개발/오프라인 상황에서는 기존 앱 DB 포트폴리오 계산으로 fallback한다.
4. 화면에는 보유 현황과 성과 지표의 기준이 다르다는 점을 명확히 표시한다.
5. 수정 후 백엔드 단위 테스트, 프론트 포트폴리오 테스트, 프론트 빌드를 확인한다.

## 백엔드 수정

- `GET /portfolio`가 `kis_account_service.get_account_balance(settings.SYSTEM_KIS_MODE)`를 먼저 호출하도록 변경했다.
- `paper`, `real` 모드 모두 같은 KIS 계좌 잔고 서비스를 사용한다.
- KIS 응답을 포트폴리오 응답 형태로 변환하는 `_portfolio_from_kis_balance()`를 추가했다.
- endpoint wrapper와 실제 계산 로직을 분리하기 위해 `_get_portfolio_response()`를 추가했다.
  - rate limit decorator나 HTTP client fixture에 묶이지 않고 핵심 로직을 단위 테스트할 수 있다.
- KIS 조회 실패 시 기존 DB 기반 보유 종목 계산으로 fallback한다.
- `/portfolio` 응답에 아래 필드를 추가했다.
  - `total_asset`
  - `deposit`
  - `holding_source`
  - `performance_source`

## 프론트엔드 수정

- `PortfolioResponse` 타입에 신규 응답 필드를 추가했다.
- 포트폴리오 탭 상단에 데이터 출처 라벨을 추가했다.
  - `보유 현황: KIS ... 계좌`
  - `성과 지표: 앱 거래 기록 기준`
- 포트폴리오 mock 테스트 데이터에도 출처 필드를 반영했다.
- 프론트 build를 막던 기존 Recharts `Tooltip.formatter` 타입 오류 2건을 보정했다.
  - `InvestorPanel`
  - `BacktestTab`

## 테스트

통과:

```bash
docker compose run --rm -v /Users/hwang/Gwang/Class/aiclass/FinalProject:/project -w /project -e PYTHONPATH=/project/backend backend sh -c 'pip install --no-cache-dir pytest==8.4.2 pytest-asyncio==1.4.0 >/tmp/test-deps.log && pytest -q tests/test_portfolio.py::test_portfolio_paper_mode_uses_kis_account_balance tests/test_portfolio.py::test_portfolio_real_mode_uses_kis tests/test_portfolio.py::test_portfolio_real_mode_fallback_to_db'
```

결과: `3 passed, 2 warnings`

```bash
docker compose run --rm --no-deps frontend npm test -- --run src/test/PortfolioTab.test.tsx
```

결과: `4 passed`

```bash
docker compose run --rm --no-deps frontend npm run build
```

결과: 성공

추가 운영 확인:

```bash
docker compose restart backend frontend
docker compose exec -T backend python -c 'import httpx; r=httpx.get("http://localhost:8000/health", timeout=5); print(r.status_code); print(r.text)'
```

결과: `200 {"status":"ok"}`

- 재시작 후 인증 토큰을 붙여 `GET /portfolio`를 실제 호출했을 때 `holding_source`가 `KIS 모의투자 계좌`로 반환되는 것을 확인했다.
- KIS 잔고 기준 보유종목이 API 응답에 포함되는 것을 확인했다.
- 기존에 화면이 비어 보이던 원인은 실행 중인 백엔드 프로세스가 수정 전 코드를 메모리에 들고 있었기 때문이며, 컨테이너 재시작 후 정상 표시되었다.

## 남은 이슈

- `npm run lint`는 이번 포트폴리오 수정과 직접 관련 없는 기존 React hooks lint 오류로 실패한다.
- 대표 위치:
  - `frontend/src/components/Analysis/ComprehensivePanel.tsx`
  - `frontend/src/components/Analysis/FundamentalPanel.tsx`
  - `frontend/src/components/Analysis/InvestorPanel.tsx`
  - `frontend/src/components/Analysis/RecommendationPanel.tsx`
  - `frontend/src/components/MainPanel/MarketTab/MarketTab.tsx`
  - `frontend/src/components/MainPanel/RecommendTab/RecommendTab.tsx`

## 판단

현재 구조에서는 포트폴리오의 보유 현황은 KIS 계좌 잔고를 기준으로 보여주고, 누적 손익/승률/MDD 같은 성과 지표는 앱 내부 거래 기록 기준으로 보여준다. 이 둘은 원본이 다르므로 화면에 출처를 표시하는 편이 사용자 혼동을 줄인다.

---

## 추가 수정: dev 병합 및 데모 안정화 보완

### 배경

`hwang` 브랜치의 데모 안정화 작업 8개 커밋과 `origin/dev`의 AI 차트 분석 추가 작업이 동시에 진행되어 merge conflict가 발생했다. dev의 신규 기능은 유지하면서, 데모 실행 안정성에 영향을 주는 부분을 추가로 보정했다.

병합 및 반영 PR:

- PR: https://github.com/hygrenn/FinalProject/pull/6
- hwang 최종 커밋: `b3510a7 fix: dev 병합 충돌 해결 및 데모 안정화 보완`
- dev merge commit: `f41dfdf213cb99edb68e30dbec4fad5bb3909666`

### 병합 정리

- `origin/dev`의 AI 차트 분석 기능을 유지했다.
  - 지지/저항 분석
  - 추세선 분석
  - 이상감지 분석
  - 차트 오버레이 컴포넌트
- `hwang`의 데모 안정화 기능도 유지했다.
  - 시스템 상태 진단 패널
  - 계좌/포트폴리오 데이터 출처 표시
  - 추천 근거 칩
  - 최근 주문 내역 패널
  - 빈 상태/오류 상태 안내 개선
- React hooks lint 수정을 양쪽 변경사항 기준으로 다시 정리했다.
- 분봉 로딩 중 일봉을 임시 표시하는 UX는 유지했다.

### 추가 버그 수정

- `backend/services/anomaly_service.py`
  - PyTorch가 설치되지 않은 Docker 환경에서도 백엔드 import가 실패하지 않도록 수정했다.
  - `torch` import와 모델 정의를 이상감지 실행 함수 내부로 이동했다.
  - PyTorch가 없으면 빈 이상감지 결과와 `available: false`를 반환한다.
- `backend/api/routes/system.py`
  - 로그인하지 않은 상태에서는 KIS 잔고 조회를 시도하지 않도록 수정했다.
  - 미로그인 상태의 account status는 `login_required`로 반환한다.
- `frontend/src/components/Trade/OrderModal.tsx`
  - 주문 요청 API가 성공하기 전에 완료 화면으로 넘어가던 문제를 수정했다.
  - 요청 중에는 입력/버튼을 비활성화하고 `주문 요청 중...` 상태를 표시한다.
- `frontend/src/components/Account/AccountPanel.tsx`
  - 사용자 객체의 mode fallback 때문에 모의계좌가 실계좌처럼 표시될 수 있던 문제를 보정했다.
  - `/account/config` 값을 함께 조회해 현재 계좌 모드를 표시한다.
- 테스트 코드
  - system status의 미로그인 응답 기대값을 `login_required` 기준으로 수정했다.
  - account panel의 비동기 최근 주문 조회가 테스트 종료 후 state update를 일으키지 않도록 대기 조건을 보강했다.

### 검증

통과:

```bash
docker compose run --rm --no-deps frontend npm test -- --run src/test/SystemStatusPanel.test.tsx src/test/PortfolioTab.test.tsx src/test/RecommendTab.test.tsx src/test/AccountPanel.test.tsx
```

결과: `4 passed`, `39 tests passed`

```bash
docker compose run --rm --no-deps frontend npm run build
```

결과: 성공. Vite chunk size warning만 발생했다.

```bash
docker compose run --rm --no-deps frontend npm run lint
```

결과: 성공

```bash
docker compose run --rm -v /Users/hwang/Gwang/Class/aiclass/FinalProject:/project -w /project -e PYTHONPATH=/project/backend backend sh -c 'python -m compileall -q backend tests'
```

결과: 성공

```bash
docker compose run --rm -v /Users/hwang/Gwang/Class/aiclass/FinalProject:/project -w /project -e PYTHONPATH=/project/backend backend sh -c 'pip install --no-cache-dir pytest==8.4.2 pytest-asyncio==1.4.0 >/tmp/test-deps.log && pytest -q tests/test_portfolio.py::test_portfolio_paper_mode_uses_kis_account_balance tests/test_portfolio.py::test_portfolio_real_mode_uses_kis tests/test_portfolio.py::test_portfolio_real_mode_fallback_to_db tests/test_phase2.py::test_intraday_query_time_is_clamped_to_regular_session tests/test_config.py'
```

결과: `5 passed, 2 warnings`

제한 사항:

```bash
docker compose run --rm -v /Users/hwang/Gwang/Class/aiclass/FinalProject:/project -w /project -e PYTHONPATH=/project/backend backend sh -c 'pip install --no-cache-dir pytest==8.4.2 pytest-asyncio==1.4.0 >/tmp/test-deps.log && pytest -q tests/test_system_status.py tests/test_trades.py tests/test_portfolio.py --tb=short'
```

결과: Docker 테스트 DB의 `stocksense` 사용자 비밀번호 인증 실패로 fixture setup 단계에서 중단되었다. 코드 assertion 실패는 아니며, 테스트 DB credential 또는 volume 상태 문제로 판단한다.

### 판단

이번 수정 후 dev에는 데모 안정화 기능과 AI 차트 분석 기능이 모두 포함되었다. PyTorch가 없는 기본 Docker 백엔드에서도 앱 import와 일반 API 실행이 깨지지 않으며, AI 이상감지는 의존성이 준비된 환경에서만 동작하고 미설치 환경에서는 graceful fallback한다.
