# 프론트엔드-백엔드 통합 작업 보고서

**작업일:** 2026-06-05  
**작업자:** seogu-Jeong (Claude Code 지원)  
**브랜치:** seogu-Jeong → dev merge 완료

---

## 1. 작업 개요

`docs/frontend-api-integration-guide.md` (hygrenn 작성)를 기준으로, 프론트엔드의 mock 데이터를 실제 백엔드 API 호출로 전면 교체하는 통합 작업을 수행했다.

---

## 2. 변경 파일 목록

| 파일 | 변경 내용 |
|---|---|
| `frontend/src/types/index.ts` | 타입 정의 백엔드 응답에 맞게 수정 |
| `frontend/src/store/authStore.ts` | 로그인 흐름 수정 |
| `frontend/src/hooks/useStockWebSocket.ts` | WebSocket → SSE 전환 |
| `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` | 실제 차트 API 연동 |
| `frontend/src/components/Trade/OrderModal.tsx` | 실제 주문 API 연동 |
| `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx` | 실제 포트폴리오 API 연동 |
| `frontend/src/components/Risk/RiskSettingsModal.tsx` | 실제 리스크 설정 API 연동 |
| `frontend/src/store/stockStore.ts` | 종목 목록 API 로드 기능 추가 |
| `frontend/src/components/Layout/MainLayout.tsx` | 앱 시작 시 종목 목록 자동 로드 |
| `frontend/src/lib/mockData.ts` | 타입 변경에 따른 mock 데이터 수정 |
| `frontend/src/components/Trade/OrderBook.tsx` | 미사용 import 제거 (빌드 오류 수정) |
| `frontend/src/test/types.test.ts` | 변경된 타입에 맞게 테스트 수정 |
| `frontend/src/test/RiskSettingsModal.test.tsx` | async API 흐름에 맞게 테스트 수정 |
| `frontend/src/test/PortfolioTab.test.tsx` | API mock + waitFor 적용 |

---

## 3. 상세 변경 내용

### 3-1. 타입 정의 수정 (`types/index.ts`)

백엔드 실제 응답 구조와 불일치하는 필드를 수정했다.

```typescript
// 변경 전
interface User {
  access_allowed: boolean  // 제거 — /auth/me 응답에 없음
  ...
}

interface Holding {
  ai_signal: 'BUY' | 'HOLD' | 'SELL'  // 제거
  // eval_amount 누락                    // 추가
}

interface PortfolioMetrics {
  total_value: number       // 제거
  total_return_pct: number  // 제거
  mdd: number               // 제거
}

interface RiskSettings {
  max_position_pct: number   // 제거
  stop_loss_pct: number      // 제거
  daily_loss_limit: number   // 제거
}

// 변경 후
interface User { id, email, mode, is_verified, dark_mode }

interface Holding {
  eval_amount: number  // 추가
  // (나머지 기존 필드 유지)
}

interface PortfolioMetrics {
  total_trades: number
  win_rate_pct: number
  sharpe_ratio: number
  mdd_pct: number
}

interface RiskSettings {
  max_per_stock_pct: number
  daily_loss_limit_pct: number
  stop_loss_enabled: boolean
  enforce_hard_stop: boolean
  trading_blocked: boolean
}

// 신규 추가
interface PortfolioResponse {
  holdings: Holding[]
  total_eval: number
  total_cost: number
  total_return_pct: number
}
```

### 3-2. 인증 흐름 수정 (`authStore.ts`)

백엔드 `/auth/login`은 `{ access_token }` 만 반환한다. 사용자 정보는 별도로 `/auth/me`를 호출해야 한다.

```typescript
// 변경 전
const { data } = await api.post('/auth/login', { email, password })
useAuthTokenRef.setToken(data.access_token)
set({ user: data.user, isLoading: false })  // data.user가 없어서 null

// 변경 후
const { data } = await api.post('/auth/login', { email, password })
useAuthTokenRef.setToken(data.access_token)
const { data: user } = await api.get('/auth/me')  // 토큰 저장 후 유저 정보 조회
set({ user, isLoading: false })
```

### 3-3. 실시간 시세: WebSocket → SSE (`useStockWebSocket.ts`)

백엔드가 WebSocket이 아닌 SSE(EventSource)로 실시간 시세를 제공한다.  
엔드포인트: `GET /ws/stocks/{code}` (HTTP SSE)

```typescript
// 변경 전: new WebSocket(`ws://host/ws/stocks/${code}`)
// 변경 후: new EventSource(`http://localhost:8000/ws/stocks/${code}`)
```

- `VITE_API_BASE` 미설정 또는 `EventSource` 미지원 환경(테스트, 구형 브라우저)에서는 3초 간격 mock 데이터로 자동 fallback.

### 3-4. 차트 데이터 연동 및 날짜 변환 (`ChartTab.tsx`)

백엔드 `/stocks/{code}/chart` 응답의 날짜 형식이 `YYYYMMDD`이지만, Lightweight Charts는 `YYYY-MM-DD`를 요구한다.

```typescript
// 변환 로직 추가
time: d.date.length === 8
  ? `${d.date.slice(0,4)}-${d.date.slice(4,6)}-${d.date.slice(6,8)}`
  : d.date
```

종목 변경 시 해당 종목의 차트 데이터를 자동으로 새로 로드한다. API 실패 시 mock 캔들 데이터로 fallback.

### 3-5. 주문 API 연동 (`OrderModal.tsx`)

기존 `setTimeout` mock을 실제 `POST /trades/order` 호출로 교체했다.

```typescript
await api.post('/trades/order', {
  stock_code: stock.code,
  order_type: orderType,
  price_type: priceType,
  quantity: Number(quantity),
  price: priceType === 'LIMIT' ? Number(price) : undefined,
  mode: user?.mode ?? 'paper',
})
```

API 오류 시 에러 메시지를 모달 내에 표시한다.

### 3-6. 포트폴리오 탭 연동 (`PortfolioTab.tsx`)

3개 엔드포인트를 병렬 호출로 로드한다:

| 엔드포인트 | 사용 데이터 |
|---|---|
| `GET /portfolio` | holdings, total_eval, total_return_pct |
| `GET /portfolio/metrics` | mdd_pct, win_rate_pct, sharpe_ratio, total_trades |
| `GET /portfolio/performance` | 누적 손익 차트 (일별 PNL → 누적합 변환) |

성과 지표 카드 3개 추가: 승률, 샤프비율, 총 거래수.

### 3-7. 리스크 설정 연동 (`RiskSettingsModal.tsx`)

모달 열릴 때 `GET /risk/settings`로 현재 설정을 불러오고, 저장 시 `PUT /risk/settings`를 호출한다.

필드명 변경: `max_position_pct` → `max_per_stock_pct`, `daily_loss_limit` → `daily_loss_limit_pct`

### 3-8. 종목 목록 API 연동 (`stockStore.ts`)

`loadStocks()` 액션 추가. `MainLayout` 마운트 시 자동 호출되어 `GET /stocks?limit=100` 에서 종목 목록을 받아온다. API 실패 시 기존 5개 mock 데이터로 fallback.

---

## 4. 테스트 결과

```
Test Files  25 passed (25)
Tests       103 passed (103)
```

빌드 오류 없음 (`tsc -b && vite build` 통과).

---

## 5. 동작 확인 (스크린샷)

| 화면 | 상태 |
|---|---|
| 랜딩 페이지 | 정상 — StockSenseAI 로고, 로그인/둘러보기 버튼 |
| 차트 탭 | 정상 — 캔들스틱 + RSI + MACD + 패턴 배지 + 호가창 |
| AI 탭 | 정상 — BUY 시그널, 점수 분해, 멀티 타임프레임 |
| 포트폴리오 탭 | 정상 — 로딩 후 API 응답 표시 (백엔드 미연결 시 빈 상태) |

---

## 6. 백엔드 연결 방법

```bash
# 루트에서 실행
cp .env.example .env  # (이미 있으면 생략)
docker-compose up -d postgres redis
cd backend && python -m uvicorn main:app --reload --port 8000

# 프론트엔드 (별도 터미널)
cd frontend && npm run dev
```

`frontend/.env`에 `VITE_API_BASE=http://localhost:8000`이 이미 설정되어 있다.

---

## 7. 미연동 항목 (향후 작업)

다음 항목은 UI가 완성되어 있으나 아직 mock 데이터를 사용 중이다:

| 항목 | 엔드포인트 |
|---|---|
| AI 시그널 탭 | `GET /ai/{code}/signal`, `/ai/{code}/predict` |
| 관심종목 패널 | `GET /watchlist`, `POST /watchlist` |
| 시뮬레이터 탭 | `POST /simulate/lumpsum` |
| 백테스팅 탭 | `POST /backtest/run`, `GET /backtest/{id}` |
| 호가창 | `GET /stocks/{code}/orderbook` |
| 패턴 배지 | `GET /ai/{code}/patterns` |
