# 데모 안정화 및 설명력 강화 설계

**작성일:** 2026-06-19
**담당:** hygrenn
**범위:** 상태 진단, 빈/오류 상태 개선, 포트폴리오/계좌 설명 보강, AI 추천 근거 요약, 주문/체결 상태 개선, lint 정리

---

## 1. 배경

현재 StockSenseAI는 차트, AI 분석, 추천, 시장/섹터, 스크리너, 백테스트, 포트폴리오, 계좌 잔고, 주문 기능을 갖추고 있다. 기능 수는 충분하지만 과제 데모 관점에서는 다음 문제가 남아 있다.

- 데이터가 비었을 때 원인이 불명확하다.
- KIS API, 로그인, 계좌, 포트폴리오, AI 예측 데이터 상태를 한눈에 확인하기 어렵다.
- 포트폴리오 보유 현황과 성과 지표의 원본이 다르지만 설명이 약하다.
- AI 추천 결과가 왜 추천인지 빠르게 이해하기 어렵다.
- 주문 후 체결 상태가 사용자에게 충분히 설명되지 않는다.
- `npm run lint`가 기존 React hooks lint 오류로 실패한다.

이 설계의 목표는 새 핵심 기능을 늘리는 것이 아니라, 이미 구현된 기능이 데모에서 안정적이고 설득력 있게 보이도록 만드는 것이다.

---

## 2. 목표

1. 사용자가 앱 상태를 스스로 진단할 수 있게 한다.
2. 빈 화면과 오류 화면을 구분해 원인을 명확히 보여준다.
3. 포트폴리오/계좌 데이터의 출처와 의미를 설명한다.
4. AI 추천 결과에 짧은 근거를 붙여 설득력을 높인다.
5. 주문/체결 상태를 사용자가 이해할 수 있게 표시한다.
6. lint/build/test 기준을 제출 전 안정 상태로 만든다.

---

## 3. 범위 밖

- 자동매매 기능
- 실계좌 주문 모드 자동 전환
- 새로운 ML 모델 학습/배포
- KIS 키를 프론트엔드에 노출하는 기능
- 데이터베이스 스키마 대규모 재설계
- 모바일 전용 화면 재설계

---

## 4. 구현 순서

작업은 네 개 PR 또는 네 개 커밋 단위로 나눈다.

| 단계 | 이름 | 목적 |
|---|---|---|
| PR 1 | 상태 진단 + 빈/오류 상태 개선 | 데모 중 문제 원인을 즉시 확인 |
| PR 2 | 포트폴리오/계좌 설명 + AI 추천 근거 | 앱의 의사결정과 데이터 출처 설명 |
| PR 3 | 주문/체결 상태 개선 | 주문 기능의 실제 동작 흐름 표시 |
| PR 4 | lint 정리 | 제출 전 코드 품질 안정화 |

각 PR은 독립적으로 build/test 가능해야 한다.

---

## 5. PR 1 — 상태 진단 + 빈/오류 상태 개선

### 5.1 사용자 경험

헤더 오른쪽에 작은 상태 버튼을 추가한다.

- 아이콘: `Activity`, `CircleCheck`, `AlertTriangle` 중 하나
- 위치: 설정 버튼 왼쪽 또는 사용자 버튼 왼쪽
- 클릭 시 우측 사이드 패널을 연다.
- 패널 이름: `시스템 상태`

상태 패널은 다음 항목을 카드 형태로 보여준다.

| 항목 | 표시 내용 |
|---|---|
| 로그인 | 로그인됨 / 로그인 필요 |
| 백엔드 | 정상 / 응답 없음 |
| KIS 설정 | 모의투자 / 실계좌 / 키 미설정 |
| 계좌 잔고 | 조회 성공 / 조회 실패 / 보유 없음 |
| 포트폴리오 | KIS 계좌 기준 / 앱 DB fallback / 조회 실패 |
| AI 예측 | 업로드 예측 있음 / 로컬 추론 fallback / 없음 |
| 최근 확인 시각 | `HH:mm:ss` |

### 5.2 백엔드 API

신규 endpoint를 추가한다.

```text
GET /system/status
```

인증은 선택으로 한다.

- 로그인하지 않아도 백엔드/KIS 설정 일부 상태는 볼 수 있다.
- 사용자별 정보가 필요한 항목은 로그인하지 않았을 때 `login_required`로 표시한다.

응답 예시:

```json
{
  "backend": {
    "ok": true,
    "message": "백엔드 정상"
  },
  "auth": {
    "logged_in": true,
    "email": "user@example.com"
  },
  "kis": {
    "mode": "paper",
    "configured": true,
    "account_no": "5019****-01",
    "message": "KIS 모의투자 설정됨"
  },
  "account": {
    "ok": true,
    "holdings_count": 2,
    "data_source": "KIS 모의투자 계좌",
    "message": "잔고 조회 성공"
  },
  "portfolio": {
    "ok": true,
    "holding_source": "KIS 모의투자 계좌",
    "performance_source": "앱 거래 기록 기준",
    "message": "보유 현황은 KIS 계좌 기준입니다."
  },
  "ai": {
    "prediction_source": "uploaded",
    "message": "업로드된 예측 데이터 사용"
  },
  "checked_at": "2026-06-19T15:30:00+09:00"
}
```

### 5.3 백엔드 구현 위치

| 파일 | 작업 |
|---|---|
| `backend/api/routes/system.py` | 신규 라우터 추가 |
| `backend/main.py` | `system.router` 등록 |
| `backend/services/kis_account_service.py` | 기존 `get_account_balance()` 재사용 |
| `backend/services/ai_service.py` | 예측 데이터 source 확인용 helper 추가 가능 |
| `tests/test_system_status.py` | status endpoint 테스트 추가 |

### 5.4 프론트엔드 구현 위치

| 파일 | 작업 |
|---|---|
| `frontend/src/components/Layout/Header.tsx` | 상태 버튼 추가 |
| `frontend/src/components/System/SystemStatusPanel.tsx` | 신규 사이드 패널 |
| `frontend/src/types/index.ts` | `SystemStatusResponse` 타입 추가 |
| `frontend/src/test/SystemStatusPanel.test.tsx` | 렌더링/오류 상태 테스트 |

### 5.5 빈/오류 상태 개선 대상

| 화면 | 현재 문제 | 개선 |
|---|---|---|
| `PortfolioTab` | 보유 없음과 조회 실패 구분 약함 | source/error/empty 상태 문구 분리 |
| `AccountPanel` | 보유 없음 문구 단순 | KIS 조회 성공 후 보유 없음임을 명시 |
| `RecommendTab` | 데이터 없음 문구 단순 | 추천 API 실패/결과 없음/로딩 지연 구분 |
| `ChartTab` | 분봉 데이터 지연 시 오해 가능 | 장 시간/조회 범위 안내 문구 추가 |

### 5.6 수용 기준

- 로그인 상태에서 상태 패널이 열린다.
- KIS 계좌 조회 성공 시 holdings count가 표시된다.
- KIS 키 미설정 시 비밀값 없이 원인 메시지만 표시된다.
- 포트폴리오가 비었을 때 `조회 성공 + 보유 없음`과 `조회 실패`가 구분된다.
- `.env` 값 원문은 어떤 응답/로그/UI에도 노출하지 않는다.

---

## 6. PR 2 — 포트폴리오/계좌 설명 + AI 추천 근거

### 6.1 포트폴리오 설명 보강

현재 `/portfolio`는 보유 현황을 KIS 계좌 잔고 기준으로 우선 조회한다. 성과 지표는 앱 내부 거래 기록 기준이다. 이 차이를 화면에서 더 명확하게 설명한다.

`PortfolioTab` 상단에 설명 박스를 추가한다.

문구 예시:

```text
보유 현황은 현재 설정된 KIS 모의투자 계좌에서 조회합니다.
수익 추이, 승률, MDD는 앱에서 발생한 체결 기록을 기준으로 계산합니다.
```

fallback 상황 문구:

```text
KIS 잔고 조회에 실패해 앱 DB 포트폴리오 기록을 표시합니다.
```

### 6.2 계좌 패널 설명 보강

`AccountPanel`에 다음 문구를 추가한다.

- 현재 조회 모드: `모의투자` 또는 `실계좌`
- 주문 모드는 설정에서만 변경된다는 설명
- 계좌번호는 항상 마스킹

문구 예시:

```text
이 패널은 현재 설정된 SYSTEM_KIS_MODE 계좌를 조회합니다.
실계좌/모의계좌 전환은 리스크/설정 화면에서만 변경합니다.
```

### 6.3 AI 추천 근거 요약

추천 탭의 각 종목에 `근거` 칩을 2~3개 표시한다.

백엔드 변경 없이 프론트에서 현재 응답 필드 기반으로 1차 생성한다.

입력 필드:

- `signal`
- `signal_score`
- `tech_score`
- `lstm_score`
- `lstm_available`

근거 생성 규칙:

| 조건 | 표시 |
|---|---|
| `signal_score >= 70` | AI 점수 우수 |
| `signal_score >= 50 && < 70` | AI 점수 양호 |
| `tech_score >= 70` | 기술 점수 강세 |
| `tech_score <= 35` | 기술 점수 약세 |
| `lstm_available && lstm_score >= 60` | LSTM 긍정 |
| `lstm_available && lstm_score <= 40` | LSTM 부정 |
| `!lstm_available` | LSTM 미사용 |
| `signal === "BUY"` | 매수 후보 |
| `signal === "SELL"` | 주의 후보 |

### 6.4 구현 위치

| 파일 | 작업 |
|---|---|
| `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx` | 설명 박스/상태 문구 보강 |
| `frontend/src/components/Account/AccountPanel.tsx` | 계좌/주문 모드 안내 추가 |
| `frontend/src/components/MainPanel/RecommendTab/RecommendTab.tsx` | 추천 근거 칩 표시 |
| `frontend/src/lib/recommendReasons.ts` | 근거 생성 helper 신규 추가 |
| `frontend/src/test/PortfolioTab.test.tsx` | 설명 문구 테스트 |
| `frontend/src/test/RecommendTab.test.tsx` | 근거 칩 테스트 |

### 6.5 수용 기준

- 추천 탭에서 각 종목의 추천 이유가 최소 1개 이상 표시된다.
- LSTM 데이터가 없을 때 `LSTM 미사용`으로 명확히 표시된다.
- 포트폴리오/계좌 화면에서 데이터 출처 설명이 한눈에 보인다.
- 설명 문구는 API 키, 계좌번호 원문, 잔고 상세 민감정보를 불필요하게 노출하지 않는다.

---

## 7. PR 3 — 주문/체결 상태 개선

### 7.1 사용자 경험

주문 후 사용자가 알 수 있어야 하는 것은 세 가지다.

1. 주문 요청이 KIS에 접수되었는가?
2. 아직 체결 대기 중인가?
3. 체결되었거나 확인 실패했는가?

`OrderModal` 또는 주문 영역에 최근 주문 상태를 표시한다.

상태 문구:

| 상태 | 표시 |
|---|---|
| `PENDING` | 주문 접수됨, 체결 확인 중 |
| `PARTIALLY_FILLED` | 일부 체결 |
| `FILLED` | 체결 완료 |
| `CANCELLED` | 주문 취소 |
| `UNKNOWN` | 체결 여부 확인 필요 |

### 7.2 최근 주문 패널

작은 `RecentTradesPanel` 컴포넌트를 추가한다.

위치 후보:

1. 계좌 패널 하단
2. 주문 모달 안
3. 포트폴리오 탭 하단

추천 위치는 계좌 패널 하단이다. 계좌/보유/주문 상태가 같은 맥락에 있기 때문이다.

### 7.3 API

기존 endpoint 재사용:

```text
GET /trades?mode=paper
```

필요하면 query를 추가한다.

```text
GET /trades?limit=10&status=PENDING
```

기존 `list_trades()`는 현재 100개 limit 고정이므로, PR 3에서 `limit` query param을 추가해도 된다.

### 7.4 구현 위치

| 파일 | 작업 |
|---|---|
| `frontend/src/components/Trade/RecentTradesPanel.tsx` | 신규 컴포넌트 |
| `frontend/src/components/Account/AccountPanel.tsx` | 최근 주문 패널 삽입 |
| `frontend/src/components/Trade/OrderModal.tsx` | 주문 성공 후 안내 문구 개선 |
| `backend/api/routes/trades.py` | `limit` query param 선택 추가 |
| `tests/test_trades.py` | limit/status 필터 테스트 보강 |

### 7.5 수용 기준

- 주문 성공 후 `주문 접수됨, 체결 확인 중` 메시지가 보인다.
- 최근 주문 목록에서 상태 badge가 표시된다.
- `UNKNOWN` 상태는 실패가 아니라 `체결 확인 필요`로 표시한다.
- 체결 상태 표시가 실제 매수/매도 가능 수량 검증 로직을 우회하지 않는다.

---

## 8. PR 4 — lint 정리

### 8.1 현재 문제

`npm run lint`가 React hooks 규칙으로 실패한다.

대표 파일:

- `frontend/src/components/Analysis/ComprehensivePanel.tsx`
- `frontend/src/components/Analysis/FundamentalPanel.tsx`
- `frontend/src/components/Analysis/InvestorPanel.tsx`
- `frontend/src/components/Analysis/RecommendationPanel.tsx`
- `frontend/src/components/MainPanel/MarketTab/MarketTab.tsx`
- `frontend/src/components/MainPanel/RecommendTab/RecommendTab.tsx`

주요 오류:

```text
react-hooks/set-state-in-effect
Calling setState synchronously within an effect can trigger cascading renders
```

### 8.2 정리 원칙

- 기능 변경 없이 lint 통과를 목표로 한다.
- API 호출 함수는 `useCallback` 또는 effect 내부 async 함수로 정리한다.
- loading 초기값을 적절히 설정해 effect 시작 시 동기 `setLoading(true)` 호출을 줄인다.
- 불필요한 `setData(null)` 초기화를 제거하거나 요청 시작 전 상태 모델을 통합한다.

### 8.3 권장 패턴

현재 패턴:

```tsx
useEffect(() => {
  setLoading(true)
  setData(null)
  api.get(...).then(...)
}, [selectedStock])
```

권장 패턴:

```tsx
const [state, setState] = useState<{
  loading: boolean
  data: Data | null
  error: string | null
}>({ loading: false, data: null, error: null })

useEffect(() => {
  if (!selectedStock?.code) return
  let cancelled = false

  async function load() {
    setState((prev) => ({ ...prev, loading: true, error: null }))
    try {
      const { data } = await api.get(...)
      if (!cancelled) setState({ loading: false, data, error: null })
    } catch {
      if (!cancelled) setState({ loading: false, data: null, error: '...' })
    }
  }

  void load()
  return () => { cancelled = true }
}, [selectedStock?.code])
```

단, lint rule이 이 패턴도 문제 삼으면 컴포넌트별로 `useAsyncResource` 같은 작은 custom hook을 도입한다.

### 8.4 수용 기준

- `npm run lint` 통과
- `npm run build` 통과
- 주요 탭 렌더링 테스트 통과
- 기능 변경 없이 UI 결과가 유지된다.

---

## 9. 보안 원칙

상태 진단/계좌/포트폴리오 개선 작업은 민감정보를 다루므로 아래 원칙을 지킨다.

- `.env` 원문 출력 금지
- KIS app key, app secret 출력 금지
- 계좌번호는 항상 마스킹
- 상태 진단 API는 보유 종목 상세를 기본 반환하지 않는다.
- 상태 진단 API는 holdings count, source, success/failure 정도만 반환한다.
- 프론트 콘솔에 KIS 응답 원문을 출력하지 않는다.
- 테스트 fixture에도 실제 키/계좌번호를 넣지 않는다.

---

## 10. 테스트 계획

### 백엔드

```bash
pytest -q tests/test_system_status.py
pytest -q tests/test_portfolio.py
pytest -q tests/test_trades.py
```

주의:

- Docker 환경에서는 테스트 DB가 개발 DB를 drop하지 않도록 `TEST_DATABASE_URL` 또는 `_test` DB 보호 로직을 유지한다.
- KIS 외부 호출은 mock 처리한다.

### 프론트엔드

```bash
npm test -- --run src/test/SystemStatusPanel.test.tsx
npm test -- --run src/test/PortfolioTab.test.tsx
npm test -- --run src/test/RecommendTab.test.tsx
npm run build
npm run lint
```

### 수동 확인

1. 로그인 전 상태 패널
2. 로그인 후 상태 패널
3. KIS 키 있음 + 보유종목 있음
4. KIS 키 없음 또는 실패 mock
5. 포트폴리오 KIS 조회 성공
6. 포트폴리오 DB fallback
7. 추천 탭 근거 칩 표시
8. 주문 후 최근 주문 상태 표시

---

## 11. 예상 파일 변경 요약

### 신규 파일

| 파일 | 목적 |
|---|---|
| `backend/api/routes/system.py` | 시스템 상태 진단 API |
| `frontend/src/components/System/SystemStatusPanel.tsx` | 상태 진단 사이드 패널 |
| `frontend/src/components/Trade/RecentTradesPanel.tsx` | 최근 주문/체결 상태 패널 |
| `frontend/src/lib/recommendReasons.ts` | AI 추천 근거 생성 helper |
| `tests/test_system_status.py` | 시스템 상태 API 테스트 |
| `frontend/src/test/SystemStatusPanel.test.tsx` | 상태 패널 테스트 |
| `frontend/src/test/RecommendTab.test.tsx` | 추천 근거 테스트 |

### 수정 파일

| 파일 | 목적 |
|---|---|
| `backend/main.py` | system router 등록 |
| `backend/api/routes/trades.py` | 최근 주문 조회 limit/status 보강 |
| `frontend/src/components/Layout/Header.tsx` | 상태 버튼 추가 |
| `frontend/src/components/Account/AccountPanel.tsx` | 설명/최근 주문 상태 추가 |
| `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx` | 설명/빈 상태 개선 |
| `frontend/src/components/MainPanel/RecommendTab/RecommendTab.tsx` | 추천 근거 표시 |
| `frontend/src/components/Trade/OrderModal.tsx` | 주문 성공/대기 안내 개선 |
| lint 대상 Analysis/Market 컴포넌트 | hooks lint 정리 |

---

## 12. 최종 완료 기준

전체 개선 작업은 다음 조건을 만족하면 완료로 본다.

- 상태 패널이 데모 중 앱 상태를 설명할 수 있다.
- 포트폴리오/계좌/추천/주문 화면에서 빈 상태와 오류 상태가 구분된다.
- AI 추천 결과에 사람이 이해할 수 있는 근거가 표시된다.
- 주문 후 체결 대기/완료/확인 필요 상태가 표시된다.
- `npm run lint`, `npm run build`, 관련 테스트가 통과한다.
- `docs/revision/`에 각 PR별 수정사항이 기록된다.
