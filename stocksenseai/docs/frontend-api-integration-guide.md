# 프론트엔드 API 연동 가이드

> **작성일:** 2026-06-05  
> **대상:** seogu-Jeong — mock 데이터를 실제 백엔드 API로 교체할 때 참고  
> **백엔드 Base URL:** `http://localhost:8000`  
> **모든 엔드포인트 상세:** `docs/superpowers/specs/2026-05-31-stocksenseai-TRD.md`

---

## 0. 환경 설정

`frontend/.env.example`에 `VITE_API_BASE`만 있는데, **SSE용 변수도 동일하게** 쓰면 됩니다.

```bash
# frontend/.env
VITE_API_BASE=http://localhost:8000
```

`VITE_WS_BASE`는 **사용하지 마세요** (아래 SSE 섹션 참고).

---

## 1. Axios 설정 — 이미 완성됨, 주의사항만

`frontend/src/lib/api.ts`는 아래가 이미 구현되어 있습니다:
- `withCredentials: true` (refresh 쿠키 자동 전송) ✅
- `Authorization: Bearer {token}` 자동 첨부 ✅
- 401 응답 시 `/auth/refresh` 자동 재시도 ✅

**수정 필요 없음.**

---

## 2. 인증 (authStore.ts)

### 문제

`authStore.ts:24`에서 `data.user`를 기대하지만, 백엔드 `/auth/login` 응답에는 `user` 객체가 **없습니다**.

```typescript
// 현재 (틀림)
const { data } = await api.post('/auth/login', { email, password })
useAuthTokenRef.setToken(data.access_token)
set({ user: data.user, isLoading: false })  // ❌ data.user는 undefined
```

### 수정

```typescript
// login 수정
login: async (email, password) => {
  set({ isLoading: true })
  try {
    const { data } = await api.post('/auth/login', { email, password })
    useAuthTokenRef.setToken(data.access_token)
    // 로그인 후 me 조회
    const { data: me } = await api.get('/auth/me')
    set({ user: me, isLoading: false })
  } catch (e) {
    set({ isLoading: false })
    throw e
  }
},
```

### `/auth/me` 실제 응답 필드

```typescript
// 백엔드가 실제로 반환하는 필드 (backend/api/routes/auth.py:84)
interface MeResponse {
  id: string
  email: string
  is_verified: boolean
  mode: 'demo' | 'paper' | 'real'   // demo = KIS 키 미등록
  dark_mode: boolean
}
// ※ access_allowed 필드는 없음 — types/index.ts에서 제거하거나 무시
```

### 앱 시작 시 토큰 복원

페이지 새로고침 후 로그인 유지를 위해 `App.tsx` 또는 최상위에 추가:

```typescript
useEffect(() => {
  api.get('/auth/me')
    .then(({ data }) => {
      useAuthTokenRef.setToken(/* 로컬스토리지나 메모리에서 */token)
      useAuthStore.getState().setUser(data)
    })
    .catch(() => {/* 미로그인 상태 */})
}, [])
```

---

## 3. 실시간 시세 — WebSocket이 아니라 SSE

### 문제

`useStockWebSocket.ts`가 `new WebSocket(...)` 으로 연결을 시도하는데, 백엔드는 **SSE (EventSource)** 를 씁니다. WebSocket으로 연결하면 연결 자체가 실패합니다.

**백엔드 엔드포인트:** `GET /ws/stocks/{code}` — `text/event-stream` 반환

### 수정 (`frontend/src/hooks/useStockWebSocket.ts` 전체 교체)

```typescript
import { useEffect, useState } from 'react'
import { useStockStore } from '@/store/stockStore'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export function useStockWebSocket(stockCode: string): { isConnected: boolean } {
  const [isConnected, setIsConnected] = useState(false)
  const updateRealtimePrice = useStockStore((s) => s.updateRealtimePrice)

  useEffect(() => {
    if (!stockCode || !API_BASE) return

    const url = `${API_BASE}/ws/stocks/${stockCode}`
    const es = new EventSource(url, { withCredentials: true })

    es.onopen = () => setIsConnected(true)
    es.onerror = () => setIsConnected(false)
    es.onmessage = (event) => {
      try {
        // 백엔드가 KIS WebSocket 데이터를 Redis Pub/Sub으로 전달하는 형식
        // 실제 데이터 구조는 KIS 원본 포맷 (price, change_pct 파싱 필요)
        const raw = JSON.parse(event.data)
        updateRealtimePrice({
          code: stockCode,
          price: Number(raw.stck_prpr ?? raw.price ?? 0),
          change_pct: Number(raw.prdy_ctrt ?? raw.change_pct ?? 0),
        })
      } catch {}
    }

    return () => {
      es.close()
      setIsConnected(false)
    }
  }, [stockCode, updateRealtimePrice])

  return { isConnected }
}
```

> **주의:** KIS WebSocket 실시간 데이터가 없으면 (장 외 시간, 키 미등록) SSE는 연결되지만 메시지가 안 올 수 있음. `isConnected`가 true여도 데이터 없으면 현재가 표시 안 함.

---

## 4. 종목 목록 / 검색 (stockStore.ts)

### mock 제거 후 실제 API

```typescript
// stockStore.ts 에 추가
fetchStockList: async (market = 'KOSPI', page = 1) => {
  const { data } = await api.get('/stocks', { params: { market, page, limit: 30 } })
  set({ stockList: data.items })   // 응답: { items: Stock[], total, page }
},

searchStocks: async (query: string) => {
  const { data } = await api.get('/stocks/search', { params: { q: query } })
  return data as Stock[]   // 응답: Stock[]
},
```

### 종목 상세 (StockInfoBar)

```typescript
// GET /stocks/{code}
const { data } = await api.get(`/stocks/${code}`)
// 응답:
// { code, name, close, open, high, low, volume, change_pct, ... }
```

### 차트 데이터 (CandleChart) — **날짜 형식 주의**

```typescript
// GET /stocks/{code}/chart?period=3m&interval=day
const { data } = await api.get(`/stocks/${code}/chart`, {
  params: { period: '3m', interval: 'day' }
})

// 백엔드 응답 날짜 형식: "20260530" (YYYYMMDD)
// Lightweight Charts는 "2026-05-30" (YYYY-MM-DD) 필요
const candles = data.data.map((c: any) => ({
  time: `${c.date.slice(0,4)}-${c.date.slice(4,6)}-${c.date.slice(6,8)}`,
  open: c.open,
  high: c.high,
  low: c.low,
  close: c.close,
  volume: c.volume,
}))
```

`period` 옵션: `1d`, `1w`, `1m`, `3m`, `1y`, `3y`  
`interval` 옵션: `1min`, `5min`, `15min`, `1h`, `1d`, `1w`, `1mo`

---

## 5. AI 탭

### AI 시그널 (SignalCard)

```typescript
// GET /ai/{code}/signal
const { data } = await api.get(`/ai/${code}/signal`)
// 응답 형식은 types/index.ts의 AISignal과 동일 ✅
```

### 5일 예측 (PredictionOverlay)

```typescript
// GET /ai/{code}/predict
const { data } = await api.get(`/ai/${code}/predict`)
// data.prediction.bullish, .base, .bearish: number[]
// data.confidence: number

// ※ LSTM 가중치가 없으면 빈 배열 반환됨 — 실제 학습 전까지 mock 유지 권장
```

### 캔들 패턴 (PatternBadges)

```typescript
// GET /ai/{code}/patterns
const { data } = await api.get(`/ai/${code}/patterns`)
// 응답: { patterns: [{ name, type, description, detected_at }] }
// type: 'bullish' | 'bearish' | 'neutral' ✅
```

### 멀티프레임 (MultiframePanel)

```typescript
// GET /ai/{code}/multiframe
const { data } = await api.get(`/ai/${code}/multiframe`)
// 응답: [{ timeframe: '1D'|'1W'|'1M', signal, score }] ✅
```

---

## 6. 주문 모달 (OrderModal.tsx) — **mock → 실제 API**

### 현재 문제

`handleSubmit`이 API를 전혀 호출하지 않고 `setTimeout`으로 완료 처리합니다.

### 수정

```typescript
const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault()
  setLoading(true)
  try {
    await api.post('/trades/order', {
      stock_code: stock.code,
      order_type: orderType,            // 'BUY' | 'SELL'
      price_type: priceType,            // 'MARKET' | 'LIMIT'
      quantity: Number(quantity),
      price: priceType === 'LIMIT' ? Number(price) : 0,
      // mode 필드 불필요 — 백엔드가 user.mode에서 자동 결정
    })
    setSubmitted(true)
    setTimeout(() => { setSubmitted(false); onClose() }, 1500)
  } catch (err: any) {
    alert(err.response?.data?.detail ?? '주문 실패')
  } finally {
    setLoading(false)
  }
}
```

### 주문 타입 안내 텍스트

| 상황 | 동작 |
|---|---|
| 시장가 주문 현재가 조회 실패 | 400 에러 — "현재가를 조회할 수 없습니다" |
| `demo` 모드 | 주문 API 자체 차단됨 — KIS 키 등록 안내 필요 |
| `trading_blocked: true` | 400 — "거래가 차단된 상태입니다" |

---

## 7. 포트폴리오 탭 — **필드명 불일치 주의**

### Holding 타입 불일치

```typescript
// types/index.ts 현재
interface Holding {
  ai_signal: 'BUY' | 'HOLD' | 'SELL'   // ❌ 백엔드 응답에 없음
}

// 백엔드 실제 응답 (backend/api/routes/portfolio.py)
{
  stock_code: string
  stock_name: string
  quantity: number
  avg_price: number
  current_price: number
  eval_amount: number    // ← 추가 필드 (총 평가금액)
  profit_loss: number
  return_pct: number
  // ai_signal 없음
}
```

→ `Holding` 타입에서 `ai_signal` 제거, `eval_amount` 추가.

### 포트폴리오 전체 응답

```typescript
const { data } = await api.get('/portfolio')
// {
//   holdings: Holding[],
//   total_eval: number,      // 총 평가금액
//   total_cost: number,      // 총 매입금액
//   total_return_pct: number
// }
```

### PortfolioMetrics 불일치

```typescript
// types/index.ts 현재 (틀림)
interface PortfolioMetrics {
  total_value: number       // ❌ 없음
  total_return_pct: number  // ❌ 없음
  mdd: number               // ❌ 필드명 다름
}

// 백엔드 실제 응답 (backend/api/routes/portfolio.py:metrics)
{
  total_trades: number
  win_rate_pct: number
  sharpe_ratio: number
  mdd_pct: number           // mdd가 아니라 mdd_pct
}
```

→ `PortfolioMetrics` 타입 교체:

```typescript
export interface PortfolioMetrics {
  total_trades: number
  win_rate_pct: number
  sharpe_ratio: number
  mdd_pct: number
}
```

### 일별 수익 (performance)

```typescript
const { data } = await api.get('/portfolio/performance')
// [{ date: "2026-05-30", pnl: 120000 }, ...]
// ※ value가 아니라 pnl (당일 실현손익)
```

---

## 8. 관심종목 (WatchlistPanel / stockStore.ts)

현재 `stockStore.ts`의 `watchlist`는 로컬 배열. 백엔드 `/watchlist` API와 연동 필요.

### 관심종목 그룹 조회

```typescript
// GET /watchlist/groups
const { data } = await api.get('/watchlist/groups')
// [{ id, name, sort_order, created_at, items: [{ id, stock_code, stock_name, target_price_high, target_price_low, sort_order }] }]
```

### 종목 추가

```typescript
await api.post('/watchlist/items', {
  group_id: '그룹UUID',
  stock_code: '005930',
  stock_name: '삼성전자',
  target_price_high: 80000,   // 선택
  target_price_low: 65000,    // 선택
})
// 409 → 이미 추가된 종목
```

### 목표가 수정

```typescript
// null을 명시적으로 보내면 목표가 삭제됨
await api.put(`/watchlist/items/${itemId}`, {
  target_price_high: null,   // 삭제
  target_price_low: 65000,   // 변경
})
```

---

## 9. 리스크 설정 (RiskSettingsModal.tsx) — **필드명 불일치**

```typescript
// types/index.ts 현재 (틀림)
interface RiskSettings {
  max_position_pct: number   // ❌
  stop_loss_pct: number      // ❌
  daily_loss_limit: number   // ❌
}

// 백엔드 실제 필드 (backend/api/routes/risk.py)
interface RiskSettings {
  max_per_stock_pct: number        // 종목별 최대 비중 (%)
  daily_loss_limit_pct: number     // 일일 손실 한도 (%)
  stop_loss_enabled: boolean       // 손절 활성화 여부
  enforce_hard_stop: boolean       // 한도 초과 시 강제 차단 여부
  trading_blocked: boolean         // 현재 거래 차단 상태
}
```

### 조회/수정

```typescript
// 조회
const { data } = await api.get('/risk/settings')

// 수정
await api.put('/risk/settings', {
  max_per_stock_pct: 20,
  daily_loss_limit_pct: 5,
  stop_loss_enabled: false,
  enforce_hard_stop: true,
})
```

---

## 10. 투자 시뮬레이터 (SimulatorTab)

### 일시불

```typescript
const { data } = await api.post('/simulate/lumpsum', {
  tickers: ['005930', '000660'],
  buy_date: '2022-01-03',
  sell_date: '2026-05-31',
  amount_krw: 1000000,
})
// {
//   buy_date_actual: string,
//   sell_date_actual: string,
//   results: [{
//     ticker, name, shares, buy_price, sell_price,
//     buy_value_krw, sell_value_krw, cash_left_krw,
//     profit_krw, return_pct,
//     buy_date_actual, sell_date_actual,
//     chart_data: [{ date, return_pct }]
//   }]
// }
```

### 적립식

```typescript
const { data } = await api.post('/simulate/recurring', {
  tickers: ['005930'],
  start_date: '2020-01-02',
  end_date: '2026-05-31',
  monthly_amount_krw: 300000,
})
// {
//   results: [{
//     ticker, name,
//     total_invested_krw, total_shares, avg_buy_price,
//     current_value_krw, return_pct, total_purchases,
//     start_date_actual, end_date_actual,
//     chart_data: [{ date, invested, value }]
//   }]
// }
```

### 데이터 준비 상태 확인

```typescript
// 시뮬레이터 첫 진입 시 실행
const { data } = await api.get('/simulate/data-status')
// { ready: boolean, ticker_count: number, last_updated: string | null }
if (!data.ready) {
  // SSE 스트리밍으로 다운로드
  const es = new EventSource(`${API_BASE}/simulate/download`, { withCredentials: true })
  es.addEventListener('progress', (e) => {
    const p = JSON.parse(e.data)
    // p.current / p.total 로 진행률 표시
  })
  es.addEventListener('complete', () => es.close())
}
```

---

## 11. 백테스팅 (BacktestTab)

```typescript
// POST /backtest/run
const { data } = await api.post('/backtest/run', {
  code: '005930',
  start_date: '2020-01-01',
  end_date: '2026-05-31',
  initial_cash: 10000000,
  entry_signal_score: 65.0,
  exit_signal_score: 35.0,
  stop_loss_pct: 0.05,
  take_profit_pct: 0.15,
  commission_rate: 0.00015,
})
// {
//   id: string,
//   stock_code, period_start, period_end,
//   total_return_pct, mdd_pct, sharpe_ratio, win_rate_pct,
//   total_trades,
//   strategy_config: {},
//   result_detail: { trades: [], equity_curve: [{ date, equity }] }
// }

// ⚠️ 동기 실행 (요청 중에 로딩 표시 필요)
// 긴 기간은 수 초 걸릴 수 있음
```

---

## 12. types/index.ts 수정 필요 항목 요약

`types/index.ts` 수정 전 hygrenn과 협의 (공동 관리 파일).

| 타입 | 수정 내용 |
|---|---|
| `User` | `access_allowed` 제거 (백엔드 없음) |
| `Holding` | `ai_signal` 제거, `eval_amount: number` 추가 |
| `PortfolioMetrics` | 전체 교체 (`mdd_pct`, `win_rate_pct`, `sharpe_ratio`, `total_trades`) |
| `RiskSettings` | 전체 교체 (위 9번 참고) |
| `OrderRequest` | `mode` 필드 제거 (백엔드가 user.mode 사용) |

---

## 13. 교체 우선순위

| 순서 | 항목 | 이유 |
|---|---|---|
| 1 | authStore.ts (login) | 로그인 안 되면 아무것도 안 됨 |
| 2 | 종목 목록/검색 | 사이드바 기본 데이터 |
| 3 | 차트 데이터 | 메인 화면 |
| 4 | 주문 모달 | 핵심 기능 |
| 5 | 포트폴리오 탭 | 잔고 확인 |
| 6 | AI 탭 | 가중치 없으면 빈 값 |
| 7 | 관심종목 | 부가 기능 |
| 8 | 시뮬레이터 | 부가 기능 |
| 9 | 백테스팅 | 부가 기능 |

---

## 14. 자주 나올 에러 코드

| HTTP | 의미 | 대응 |
|---|---|---|
| 400 | 요청 오류 (주문 차단, 현재가 조회 실패 등) | `error.response.data.detail` 메시지 표시 |
| 401 | 토큰 만료 | api.ts 인터셉터가 자동 refresh 처리 |
| 403 | 권한 없음 (타인 리소스) | — |
| 404 | 리소스 없음 | — |
| 409 | 중복 (관심종목 중복 추가 등) | "이미 추가됨" 안내 |
| 422 | Pydantic 검증 실패 | `error.response.data.detail` 배열 확인 |
| 502 | 외부 API(KIS/pykrx) 오류 | "일시적 오류, 다시 시도하세요" 안내 |
