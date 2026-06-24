# 260620 seogu-Jeong 자동매매 병합 검토

## 목적

`origin/seogu-Jeong` 브랜치의 자동매매 구현을 `dev`에 병합하기 전, 새 커밋 내용과 병합 위험성, 필요한 수정 사항을 정리한다. 이 문서는 후속 작업 agent가 현재 상황을 빠르게 이해하고 안전하게 수정/병합할 수 있도록 작성되었다.

## 현재 결론

`origin/seogu-Jeong`은 자동매매 기능을 포함하고 있지만, 현재 상태 그대로 `dev`에 merge하면 안 된다.

병합 충돌 자체는 작지만, 프론트 lint 실패와 백엔드 DB 마이그레이션 누락이 있어 실제 실행 시 자동매매 API가 깨질 가능성이 높다.

## 브랜치/커밋 상태

확인 기준:

```bash
git fetch --all --prune
git branch -r --contains cbe5f9d
```

결과:

```text
origin/seogu-Jeong
```

자동매매 대표 커밋:

```text
cbe5f9d feat: 자동매매 5분 자동 폴링 + 전종목 AI 분석 패널
3295366 refactor: 자동매매 완전 자율화 — 예산만 입력하면 AI가 총괄
2293201 feat: AI 기반 자동매매 기능 추가
```

위 커밋들은 현재 `origin/dev`, `origin/hwang`에는 포함되어 있지 않고 `origin/seogu-Jeong`에만 존재한다.

## seogu-Jeong 브랜치의 주요 변경 파일

`origin/dev..origin/seogu-Jeong` 기준 주요 변경:

```text
A backend/api/routes/auto_trade.py
A backend/models/auto_trade.py
A backend/services/auto_trade_service.py
A backend/tasks/auto_trade_tasks.py
A frontend/src/components/AutoTrade/AutoTradePanel.tsx
A frontend/src/components/ui/badge.tsx
A frontend/src/components/ui/label.tsx
A frontend/src/components/ui/slider.tsx
A frontend/src/components/ui/switch.tsx
M backend/main.py
M backend/tasks/__init__.py
M frontend/src/components/MainPanel/MainPanel.tsx
M frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
M frontend/src/types/index.ts
M frontend/package.json
M frontend/package-lock.json
M dev_startup.py
```

자동매매 라우터는 `backend/main.py`에 다음 형태로 연결된다.

```python
app.include_router(auto_trade_router.router, prefix="/auto-trade", tags=["auto-trade"])
```

프론트에서는 `MainPanel`에 `autotrade` 탭이 추가되고, `AutoTradePanel`이 렌더링된다.

## 병합 시뮬레이션 결과

임시 worktree에서 아래 방식으로 병합을 시뮬레이션했다.

```bash
git worktree add /Users/hwang/Gwang/Class/aiclass/FinalProject_merge_seogu_check -b merge-seogu-check-2 origin/dev
cd /Users/hwang/Gwang/Class/aiclass/FinalProject_merge_seogu_check
git merge --no-commit --no-ff origin/seogu-Jeong
```

충돌 파일:

```text
frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
```

충돌 내용은 `prediction` state 초기값 차이다.

`dev` 쪽:

```tsx
const [prediction, setPrediction] = useState<Prediction | null>(null)
```

`seogu-Jeong` 쪽:

```tsx
const [prediction, setPrediction] = useState<Prediction | null>({
  bullish: [],
  base: [],
  bearish: [],
  confidence: 0,
})
```

임시 검증에서는 `dev` 쪽 `null` 초기값을 유지하는 방향으로 해결했다. 현재 `ChartTab`은 종목 전환 시 `setPrediction(null)`을 사용하고, LSTM 사용 불가 시 예측선을 숨기는 흐름이 있으므로 `null` 유지가 자연스럽다.

## 검증 결과

### Frontend build

명령:

```bash
cd frontend
npm ci
npm run build
```

결과:

```text
success
```

단, Vite bundle size warning은 존재한다.

### Frontend lint

명령:

```bash
cd frontend
npm run lint
```

결과:

```text
failed
```

실패 파일:

```text
frontend/src/components/AutoTrade/AutoTradePanel.tsx
```

에러 요약:

```text
react-hooks/refs:
- render 중 configRef.current 직접 갱신
- render 중 watchlistRef.current 직접 갱신

react-hooks/set-state-in-effect:
- effect 본문에서 fetchConfig() 호출이 즉시 setState를 유발

@typescript-eslint/no-explicit-any:
- catch (e: any) 4곳
```

### Backend compile/import

명령:

```bash
docker compose run --rm --no-deps \
  -v /Users/hwang/Gwang/Class/aiclass/FinalProject_merge_seogu_check:/project \
  -w /project \
  -e PYTHONPATH=/project/backend \
  backend python -m compileall -q backend

docker compose run --rm --no-deps \
  -v /Users/hwang/Gwang/Class/aiclass/FinalProject_merge_seogu_check:/project \
  -w /project \
  -e PYTHONPATH=/project/backend \
  backend python -c 'import main; print("main import ok")'
```

결과:

```text
compile success
main import ok
```

`main` import 중 `KRX_ID`/`KRX_PW` 미설정 경고가 출력되지만 import 자체는 성공했다.

### Backend full test

전체 테스트는 검증 환경에서 DB 서비스 없이 `--no-deps`로 실행되어 `postgres` hostname을 찾지 못해 대량 실패했다. 이 실패 자체는 자동매매 코드 결함으로 단정하지 않는다.

다만 아래 DB 마이그레이션 누락 문제는 별도로 확인된 실제 결함이다.

## 주요 위험성

### P0. 자동매매 DB 마이그레이션 누락

신규 모델:

```text
backend/models/auto_trade.py
```

신규 테이블:

```text
auto_trade_configs
auto_trade_logs
```

하지만 다음 문제가 있다.

```text
db/migrations/versions/ 하위에 auto_trade 테이블 생성 migration 없음
db/migrations/env.py 에서 models.auto_trade import 없음
backend/models/__init__.py 에도 AutoTradeConfig/AutoTradeLog export 없음
```

따라서 실제 DB에 테이블이 생성되지 않는다. 그 상태로 `/auto-trade/config`, `/auto-trade/logs`, `/auto-trade/run` 등을 호출하면 `relation "auto_trade_configs" does not exist` 계열 오류가 날 가능성이 높다.

필수 수정:

1. Alembic migration 추가
2. `db/migrations/env.py`에 `models.auto_trade` import 추가
3. migration 적용 후 `/auto-trade/config`가 기본 config를 생성/반환하는지 테스트

### P0. 자동매매는 주문성 기능이므로 안전장치 검토 필요

현재 Celery beat에 자동매매 태스크가 등록된다.

```python
"run-auto-trade": {
    "task": "tasks.auto_trade_tasks.run_auto_trade_all",
    "schedule": crontab(minute="*/10", hour="9-15", day_of_week="mon-fri"),
}
```

기본값은 `enabled=False`라 즉시 주문되지는 않는다. 그래도 자동매매는 주문성 기능이므로 병합 전 아래를 확인해야 한다.

필수 검토:

1. 기본 모드가 반드시 `paper`인지
2. `real` 모드 활성화 시 사용자 확인 UI가 있는지
3. 백엔드에서도 `real` 모드 자동매매를 막거나 명시적으로 허용하는 정책이 있는지
4. 중복 주문 방지 로직이 충분한지
5. 예산 한도, 종목별 한도, 현금 보유 비율이 실제로 지켜지는지
6. 자동 실행 주기가 장 마감/휴장/비거래일을 고려하는지

현재 구현은 config 기본값이 `paper`이고 `enabled=False`인 점은 좋다. 다만 백엔드 레벨에서 real 자동매매에 대한 추가 guard는 더 확인/보강하는 편이 안전하다.

### P1. Frontend lint 실패

`AutoTradePanel.tsx`의 lint 에러 때문에 현재 프로젝트 기준 품질 게이트를 통과하지 못한다.

수정 방향:

1. `configRef.current = config`, `watchlistRef.current = watchlist`를 render 중 직접 실행하지 말고 `useEffect`로 이동한다.
2. 초기 로딩 effect에서 `fetchConfig()`처럼 내부에서 즉시 setState하는 함수를 직접 호출하지 않도록 구조를 바꾼다.
3. `catch (e: any)`를 `unknown`으로 받고 axios error helper로 정리한다.
4. 수정 후 `npm run lint`가 통과해야 한다.

### P1. 자동매매 API 테스트 부재

현재 자동매매 기능은 새 라우터/서비스/모델을 추가하지만 해당 API 테스트가 없다.

최소 추가 테스트:

```text
tests/test_auto_trade.py
```

권장 케이스:

1. 인증 없이 `/auto-trade/config` 접근 시 401/403 정책 확인
2. 인증 후 `/auto-trade/config` 호출 시 기본 config 생성
3. `/auto-trade/config` update 시 total_budget, stop_loss_pct, take_profit_pct 반영
4. `enabled=False` 상태에서 `/auto-trade/run` 호출 시 skipped 반환
5. `/auto-trade/stop` 호출 시 enabled가 false로 바뀜
6. 잘못된 mode 입력 시 422
7. limit 초과 logs 요청 시 최대 200으로 clamp

### P1. 실제 주문 실행 여부 명확화 필요

서비스 함수 이름은 자동매매지만 내부 실행 함수는 `_execute_paper_order`이고, `cfg.mode`가 `real`이어도 현재 로직상 `Trade`/`Portfolio`에 기록하는 방식으로 보인다. 즉 실제 KIS 주문 API를 호출하는 구조인지, 앱 내부 모의 체결인지 정책이 불명확하다.

확인 필요:

1. `mode="real"`일 때 실제 KIS 주문을 보낼 의도인지
2. 아니면 자동매매는 항상 내부 paper execution만 할 것인지
3. UI에서 `real`을 선택 가능하게 둘지
4. 백엔드에서 real 자동매매를 금지할지

과제/데모 기준이면 자동매매는 `paper` 전용으로 제한하는 것이 더 안전하다.

### P2. Docker/검증 환경 주의

임시 worktree를 `/private/tmp`에 만들었을 때 Docker bind mount가 비정상적으로 보이는 문제가 있었다. 검증용 worktree는 `/Users/hwang/Gwang/Class/aiclass/` 아래에 만드는 편이 낫다.

또한 `docker compose run --rm --no-deps frontend npm run build`는 named volume `frontend_node_modules:/app/node_modules`가 이미지의 `node_modules`를 덮어 `tsc: not found`를 만들 수 있다. 프론트 build 검증은 현재 구조에서는 로컬 `npm ci && npm run build` 또는 node_modules 볼륨 초기화 후 실행이 더 안정적이다.

## 후속 작업 권장 순서

### 1. 병합용 임시 브랜치 생성

```bash
git fetch --all --prune
git checkout dev
git pull origin dev
git checkout -b merge-seogu-autotrade
git merge origin/seogu-Jeong
```

`ChartTab.tsx` 충돌은 `dev`의 `null` 초기값을 유지하는 방향으로 해결한다.

### 2. 자동매매 migration 추가

해야 할 일:

1. `db/migrations/env.py`에 `from models.auto_trade import AutoTradeConfig, AutoTradeLog  # noqa: F401` 추가
2. Alembic revision 생성 또는 수동 migration 파일 추가
3. `auto_trade_configs`, `auto_trade_logs` 테이블 생성
4. `user_id` FK, index, server default 반영

검증:

```bash
alembic upgrade head
```

### 3. 자동매매 API 테스트 추가

`tests/test_auto_trade.py`를 추가해 최소 API 흐름을 고정한다.

우선 테스트를 작성하고 실패를 확인한 뒤 구현을 수정한다.

권장 명령:

```bash
pytest tests/test_auto_trade.py -q
```

### 4. Frontend lint 수정

`frontend/src/components/AutoTrade/AutoTradePanel.tsx`의 lint 에러 7개를 해결한다.

검증:

```bash
cd frontend
npm run lint
npm run build
```

### 5. 자동매매 안전 정책 확정

권장 정책:

```text
과제/데모 환경에서는 자동매매를 paper 전용으로 제한한다.
real 모드는 UI에 표시하더라도 활성화 불가 또는 별도 명시적 backend guard를 둔다.
```

백엔드에서 `mode="real"` 자동매매 실행을 막는다면 `/auto-trade/run`에서 명확한 400/403을 반환하도록 테스트를 추가한다.

### 6. 전체 검증

권장 검증:

```bash
docker compose run --rm backend pytest -q tests --tb=short
cd frontend && npm run lint && npm run build && npm test -- --run
```

현재 프로젝트 compose는 로컬 포트 충돌과 node_modules volume 영향을 받을 수 있으므로, 실행 전 기존 컨테이너 상태를 확인한다.

```bash
docker compose ps
```

## merge 여부 판단

현재 상태 그대로는 merge 비추천.

merge 전 필수 조건:

```text
[ ] ChartTab 충돌 해결
[ ] AutoTradePanel lint 통과
[ ] 자동매매 Alembic migration 추가
[ ] /auto-trade 기본 API 테스트 추가
[ ] backend 테스트 통과
[ ] frontend lint/build/test 통과
[ ] 자동매매 real/paper 정책 명확화
```

위 조건을 만족하면 `seogu-Jeong -> dev` PR로 병합하는 것이 가장 안전하다.

---

## 후속 수정 현황 (PR #8, 2026-06-20)

PR #8(`merge-seogu-autotrade → dev`)에서 위 필수 조건 7개를 모두 해결했다.

```text
[x] ChartTab 충돌 해결 (null 초기값 유지)
[x] AutoTradePanel lint 통과 (0 errors)
[x] 자동매매 Alembic migration 추가 (g7h8i9j0k1l2)
[x] /auto-trade 기본 API 테스트 추가 (7개)
[x] backend 테스트 통과 (145 passed)
[x] frontend lint/build 통과
[x] 자동매매 real/paper 정책 명확화 (PUT → 400, run_cycle → skipped)
```

단, PR #8 추가 검토(`260620_pr8_autotrade_fix_instructions.md`)에서 아래 2개 이슈가 발견되어 동일 브랜치에서 추가 수정함:

- **P0**: `g7h8i9j0k1l2` migration의 `down_revision`이 `f6a1b2c3d4e5`를 가리켜 head가 2개(`c9d0e1f2a3b4`, `g7h8i9j0k1l2`)로 갈라짐 → `down_revision = "c9d0e1f2a3b4"`로 수정하여 단일 head로 통합
- **P1**: Celery beat의 `run-auto-trade` 스케줄과 브라우저 5분 폴링이 동시에 동작하면 같은 계정에 중복 실행될 수 있음 → Celery beat 스케줄 제거 (UI 폴링만 유지)

