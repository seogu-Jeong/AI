# StockSenseAI Frontend Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 2 프론트엔드 완성 — RSI/MACD 서브차트, Mock WebSocket 실시간 시세, 카드형 종목 상세 정보

**Architecture:** ChartTab을 StockInfoBar·CandleChart·RSIChart·MACDChart 4개 파일로 분리하고, MockWebSocket 클래스 + useStockWebSocket 훅으로 실시간 시세를 시뮬레이션한다. 모든 백엔드 의존 데이터는 Mock으로 대체하고, 교체 지점(mockWebSocket URL, MOCK_CANDLES → API 호출)을 명확히 분리한다.

**Tech Stack:** React 18, TypeScript 5, Lightweight Charts 4, Zustand 4, Vitest, React Testing Library

---

## File Map

| 파일 | 역할 | 상태 |
|---|---|---|
| `frontend/src/types/index.ts` | StockDetail 타입 추가 | 수정 |
| `frontend/src/lib/indicators.ts` | RSI/MACD 순수 계산 함수 | 신규 |
| `frontend/src/lib/mockData.ts` | MOCK_STOCK_DETAILS 추가 | 수정 |
| `frontend/src/lib/mockWebSocket.ts` | Mock WebSocket 클래스 | 신규 |
| `frontend/src/store/stockStore.ts` | realtimePrice + updateRealtimePrice | 수정 |
| `frontend/src/hooks/useStockWebSocket.ts` | WebSocket 연결/cleanup 훅 | 신규 |
| `frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx` | 카드형 종목 정보 | 신규 |
| `frontend/src/components/MainPanel/ChartTab/CandleChart.tsx` | 캔들스틱 차트 | 신규 (기존 로직 분리) |
| `frontend/src/components/MainPanel/ChartTab/RSIChart.tsx` | RSI 서브차트 | 신규 |
| `frontend/src/components/MainPanel/ChartTab/MACDChart.tsx` | MACD 서브차트 | 신규 |
| `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` | 조립만 (리팩터) | 수정 |

---

## Task 1: types/index.ts에 StockDetail 타입 추가

**Files:**
- Modify: `frontend/src/types/index.ts`
- Test: `frontend/src/test/types.test.ts`

- [ ] **Step 1: StockDetail 타입 추가**

`frontend/src/types/index.ts` 맨 끝에 추가:

```typescript
export interface StockDetail {
  open: number
  high: number
  low: number
  volume: number
}

export interface RealtimePrice {
  code: string
  price: number
  change_pct: number
}

export interface MACDPoint {
  time: string
  macd: number
  signal: number
  histogram: number
}
```

- [ ] **Step 2: 타입 테스트 추가**

`frontend/src/test/types.test.ts`의 `describe('types', ...)` 블록에 추가:

```typescript
it('StockDetail has OHLV fields', () => {
  const detail: StockDetail = { open: 72800, high: 74200, low: 72100, volume: 12300000 }
  expect(detail.high).toBe(74200)
})
```

import 줄도 업데이트:
```typescript
import type { User, Candle, TabId, StockDetail } from '@/types'
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `34 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/types/index.ts frontend/src/test/types.test.ts
git commit -m "feat: add StockDetail, RealtimePrice, MACDPoint types"
git push origin seogu-Jeong
```

---

## Task 2: lib/indicators.ts — RSI/MACD 계산 함수

**Files:**
- Create: `frontend/src/lib/indicators.ts`
- Create: `frontend/src/test/indicators.test.ts`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/indicators.test.ts`:

```typescript
import { calculateRSI, calculateMACD } from '@/lib/indicators'
import type { Candle } from '@/types'

function makeCandlesAscending(count: number): Candle[] {
  return Array.from({ length: count }, (_, i) => ({
    time: `2026-0${Math.floor(i / 28) + 1}-${String((i % 28) + 1).padStart(2, '0')}`,
    open: 1000 + i * 10,
    high: 1010 + i * 10,
    low: 990 + i * 10,
    close: 1005 + i * 10,
    volume: 1000000,
  }))
}

function makeCandlesDescending(count: number): Candle[] {
  return Array.from({ length: count }, (_, i) => ({
    time: `2026-0${Math.floor(i / 28) + 1}-${String((i % 28) + 1).padStart(2, '0')}`,
    open: 2000 - i * 10,
    high: 2010 - i * 10,
    low: 1990 - i * 10,
    close: 1995 - i * 10,
    volume: 1000000,
  }))
}

describe('calculateRSI', () => {
  it('returns empty array for fewer than 15 candles', () => {
    const candles = makeCandlesAscending(10)
    expect(calculateRSI(candles)).toHaveLength(0)
  })

  it('returns array of length candles.length - 14 for sufficient data', () => {
    const candles = makeCandlesAscending(30)
    expect(calculateRSI(candles)).toHaveLength(30 - 14)
  })

  it('RSI is above 70 for consistently rising prices', () => {
    const candles = makeCandlesAscending(40)
    const result = calculateRSI(candles)
    const last = result[result.length - 1]
    expect(last.value).toBeGreaterThan(70)
  })

  it('RSI is below 30 for consistently falling prices', () => {
    const candles = makeCandlesDescending(40)
    const result = calculateRSI(candles)
    const last = result[result.length - 1]
    expect(last.value).toBeLessThan(30)
  })

  it('each result has time and value fields', () => {
    const candles = makeCandlesAscending(20)
    const result = calculateRSI(candles)
    expect(result[0]).toHaveProperty('time')
    expect(result[0]).toHaveProperty('value')
  })
})

describe('calculateMACD', () => {
  it('returns empty array for fewer than 27 candles', () => {
    const candles = makeCandlesAscending(20)
    expect(calculateMACD(candles)).toHaveLength(0)
  })

  it('each result has time, macd, signal, histogram', () => {
    const candles = makeCandlesAscending(50)
    const result = calculateMACD(candles)
    expect(result[0]).toHaveProperty('time')
    expect(result[0]).toHaveProperty('macd')
    expect(result[0]).toHaveProperty('signal')
    expect(result[0]).toHaveProperty('histogram')
  })

  it('histogram equals macd minus signal', () => {
    const candles = makeCandlesAscending(50)
    const result = calculateMACD(candles)
    result.forEach((r) => {
      expect(Math.abs(r.histogram - (r.macd - r.signal))).toBeLessThan(0.0001)
    })
  })
})
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd ~/FinalProject/frontend
npm run test:run -- --reporter=verbose 2>&1 | grep "indicators"
```

Expected: FAIL (indicators 모듈 없음)

- [ ] **Step 3: indicators.ts 구현**

`frontend/src/lib/indicators.ts`:

```typescript
import type { Candle, MACDPoint } from '@/types'

function ema(values: number[], period: number): number[] {
  const k = 2 / (period + 1)
  const result: number[] = []
  let emaPrev = values.slice(0, period).reduce((a, b) => a + b, 0) / period
  result.push(emaPrev)
  for (let i = period; i < values.length; i++) {
    emaPrev = values[i] * k + emaPrev * (1 - k)
    result.push(emaPrev)
  }
  return result
}

export function calculateRSI(
  candles: Candle[],
  period = 14
): { time: string; value: number }[] {
  if (candles.length <= period) return []

  const result: { time: string; value: number }[] = []
  const closes = candles.map((c) => c.close)

  let avgGain = 0
  let avgLoss = 0

  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1]
    if (diff > 0) avgGain += diff
    else avgLoss += Math.abs(diff)
  }
  avgGain /= period
  avgLoss /= period

  for (let i = period; i < candles.length; i++) {
    if (i > period) {
      const diff = closes[i] - closes[i - 1]
      const gain = diff > 0 ? diff : 0
      const loss = diff < 0 ? Math.abs(diff) : 0
      avgGain = (avgGain * (period - 1) + gain) / period
      avgLoss = (avgLoss * (period - 1) + loss) / period
    }
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
    result.push({ time: candles[i].time, value: Math.round((100 - 100 / (1 + rs)) * 100) / 100 })
  }

  return result
}

export function calculateMACD(candles: Candle[]): MACDPoint[] {
  if (candles.length < 27) return []

  const closes = candles.map((c) => c.close)
  const ema12 = ema(closes, 12)
  const ema26 = ema(closes, 26)

  const macdLine: number[] = []
  const macdTimes: string[] = []
  // ema26[i] = EMA26 at time (i+25), ema12[i+14] = EMA12 at time (i+25)
  const macdOffset = ema12.length - ema26.length  // = 14

  for (let i = 0; i < ema26.length; i++) {
    macdLine.push(ema12[i + macdOffset] - ema26[i])
    macdTimes.push(candles[i + 25].time)
  }

  if (macdLine.length < 9) return []

  const signalLine = ema(macdLine, 9)
  const sigOffset = macdLine.length - signalLine.length

  return signalLine.map((sig, i) => {
    const macdVal = macdLine[i + sigOffset]
    return {
      time: macdTimes[i + sigOffset],
      macd: Math.round(macdVal * 100) / 100,
      signal: Math.round(sig * 100) / 100,
      histogram: Math.round((macdVal - sig) * 100) / 100,
    }
  })
}
```

- [ ] **Step 4: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `42 passed` (기존 34 + indicators 8)

- [ ] **Step 5: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/lib/indicators.ts frontend/src/test/indicators.test.ts
git commit -m "feat: add RSI and MACD indicator calculations"
git push origin seogu-Jeong
```

---

## Task 3: mockData.ts에 MOCK_STOCK_DETAILS 추가

**Files:**
- Modify: `frontend/src/lib/mockData.ts`
- Test: `frontend/src/test/mockData.test.ts`

- [ ] **Step 1: StockDetail import + MOCK_STOCK_DETAILS 추가**

`frontend/src/lib/mockData.ts` 상단 import 수정:

```typescript
import type { Candle, Stock, StockDetail } from '@/types'
```

파일 맨 끝에 추가:

```typescript
export const MOCK_STOCK_DETAILS: Record<string, StockDetail> = {
  '005930': { open: 72800, high: 74200, low: 72100, volume: 12300000 },
  '000660': { open: 184000, high: 186500, low: 183500, volume: 5200000 },
  '035420': { open: 209000, high: 211500, low: 208500, volume: 3100000 },
  '035720': { open: 41800, high: 42500, low: 41600, volume: 8700000 },
  '051910': { open: 318000, high: 322000, low: 317000, volume: 1200000 },
}
```

- [ ] **Step 2: 테스트 추가**

`frontend/src/test/mockData.test.ts`에 추가:

```typescript
import { MOCK_STOCKS, MOCK_CANDLES, MOCK_WATCHLIST, MOCK_STOCK_DETAILS } from '@/lib/mockData'

// 기존 테스트 유지하고 아래 추가
it('MOCK_STOCK_DETAILS covers all MOCK_STOCKS', () => {
  MOCK_STOCKS.forEach((s) => {
    expect(MOCK_STOCK_DETAILS[s.code]).toBeDefined()
  })
})

it('MOCK_STOCK_DETAILS high >= low for all stocks', () => {
  Object.values(MOCK_STOCK_DETAILS).forEach((d) => {
    expect(d.high).toBeGreaterThanOrEqual(d.low)
  })
})
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `44 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/lib/mockData.ts frontend/src/test/mockData.test.ts
git commit -m "feat: add MOCK_STOCK_DETAILS for stock info bar"
git push origin seogu-Jeong
```

---

## Task 4: lib/mockWebSocket.ts — Mock WebSocket 클래스

**Files:**
- Create: `frontend/src/lib/mockWebSocket.ts`
- Create: `frontend/src/test/mockWebSocket.test.ts`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/mockWebSocket.test.ts`:

```typescript
import { MockWebSocket } from '@/lib/mockWebSocket'

describe('MockWebSocket', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('calls onopen after construction', () => {
    const onopen = vi.fn()
    const ws = new MockWebSocket('ws://localhost/test')
    ws.onopen = onopen
    vi.advanceTimersByTime(1)  // fire setTimeout(0) only
    ws.close()                 // stop interval to prevent infinite tick
    expect(onopen).toHaveBeenCalled()
  })

  it('readyState is 1 (OPEN) after connection', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    vi.advanceTimersByTime(1)  // fire setTimeout(0) only
    expect(ws.readyState).toBe(1)
    ws.close()  // cleanup interval
  })

  it('calls onmessage with price data on interval', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    const onmessage = vi.fn()
    ws.onmessage = onmessage
    vi.advanceTimersByTime(2100)
    expect(onmessage).toHaveBeenCalled()
    const parsed = JSON.parse(onmessage.mock.calls[0][0].data)
    expect(parsed).toHaveProperty('price')
    expect(parsed).toHaveProperty('change_pct')
  })

  it('close() sets readyState to 3 and calls onclose', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    const onclose = vi.fn()
    ws.onclose = onclose
    ws.close()
    expect(ws.readyState).toBe(3)
    expect(onclose).toHaveBeenCalled()
  })

  it('does not call onmessage after close()', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    const onmessage = vi.fn()
    ws.onmessage = onmessage
    ws.close()
    vi.advanceTimersByTime(5000)
    expect(onmessage).not.toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd ~/FinalProject/frontend
npm run test:run -- --reporter=verbose 2>&1 | grep "mockWebSocket"
```

Expected: FAIL

- [ ] **Step 3: mockWebSocket.ts 구현**

`frontend/src/lib/mockWebSocket.ts`:

```typescript
export class MockWebSocket {
  url: string
  readyState: number = 0
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null

  private _interval: ReturnType<typeof setInterval> | null = null
  private _basePrice: number = 73400

  constructor(url: string) {
    this.url = url
    // Extract base price from stock code in URL if available
    const match = url.match(/\/(\d{6})$/)
    if (match) {
      const prices: Record<string, number> = {
        '005930': 73400,
        '000660': 185000,
        '035420': 210000,
        '035720': 42100,
        '051910': 320000,
      }
      this._basePrice = prices[match[1]] ?? 73400
    }

    setTimeout(() => {
      this.readyState = 1
      this.onopen?.()
      this._startInterval()
    }, 0)
  }

  private _startInterval() {
    this._interval = setInterval(() => {
      const change = (Math.random() - 0.5) * this._basePrice * 0.006
      this._basePrice = Math.round(this._basePrice + change)
      const change_pct = Math.round((change / (this._basePrice - change)) * 10000) / 100
      this.onmessage?.({
        data: JSON.stringify({ price: this._basePrice, change_pct }),
      })
    }, 2000)
  }

  send(_data: string): void {}

  close(): void {
    if (this._interval) {
      clearInterval(this._interval)
      this._interval = null
    }
    this.readyState = 3
    this.onclose?.()
  }
}
```

- [ ] **Step 4: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `49 passed`

- [ ] **Step 5: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/lib/mockWebSocket.ts frontend/src/test/mockWebSocket.test.ts
git commit -m "feat: add MockWebSocket class for realtime price simulation"
git push origin seogu-Jeong
```

---

## Task 5: stockStore에 realtimePrice 추가

**Files:**
- Modify: `frontend/src/store/stockStore.ts`
- Modify: `frontend/src/test/stores.test.ts`

- [ ] **Step 1: stockStore.ts 수정**

`frontend/src/store/stockStore.ts` 전체 교체:

```typescript
// frontend/src/store/stockStore.ts
import { create } from 'zustand'
import type { Stock, RealtimePrice } from '@/types'
import { MOCK_STOCKS, MOCK_WATCHLIST } from '@/lib/mockData'

interface StockState {
  selectedStock: Stock | null
  watchlist: string[]
  stockList: Stock[]
  realtimePrice: RealtimePrice | null
  setSelectedStock: (stock: Stock) => void
  addToWatchlist: (code: string) => void
  removeFromWatchlist: (code: string) => void
  updateRealtimePrice: (data: RealtimePrice) => void
}

export const useStockStore = create<StockState>((set) => ({
  selectedStock: MOCK_STOCKS[0],
  watchlist: MOCK_WATCHLIST,
  stockList: MOCK_STOCKS,
  realtimePrice: null,

  setSelectedStock: (stock) => set({ selectedStock: stock, realtimePrice: null }),

  addToWatchlist: (code) =>
    set((state) => ({
      watchlist: state.watchlist.includes(code)
        ? state.watchlist
        : [...state.watchlist, code],
    })),

  removeFromWatchlist: (code) =>
    set((state) => ({
      watchlist: state.watchlist.filter((c) => c !== code),
    })),

  updateRealtimePrice: (data) => set({ realtimePrice: data }),
}))
```

- [ ] **Step 2: 스토어 테스트 추가**

`frontend/src/test/stores.test.ts`의 `describe('stockStore', ...)` 블록에 추가:

```typescript
it('updateRealtimePrice stores realtime data', () => {
  const { result } = renderHook(() => useStockStore())
  act(() => result.current.updateRealtimePrice({ code: '005930', price: 74000, change_pct: 2.1 }))
  expect(result.current.realtimePrice?.price).toBe(74000)
})

it('setSelectedStock resets realtimePrice', () => {
  const { result } = renderHook(() => useStockStore())
  act(() => result.current.updateRealtimePrice({ code: '005930', price: 74000, change_pct: 2.1 }))
  act(() => result.current.setSelectedStock(MOCK_STOCKS[1]))
  expect(result.current.realtimePrice).toBeNull()
})
```

`stores.test.ts` 상단 import에 MOCK_STOCKS 추가:
```typescript
import { MOCK_STOCKS } from '@/lib/mockData'
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `51 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/store/stockStore.ts frontend/src/test/stores.test.ts
git commit -m "feat: add realtimePrice to stockStore"
git push origin seogu-Jeong
```

---

## Task 6: hooks/useStockWebSocket.ts

**Files:**
- Create: `frontend/src/hooks/useStockWebSocket.ts`
- Create: `frontend/src/test/useStockWebSocket.test.ts`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/useStockWebSocket.test.ts`:

```typescript
import { renderHook, act } from '@testing-library/react'
import { useStockWebSocket } from '@/hooks/useStockWebSocket'
import { useStockStore } from '@/store/stockStore'

describe('useStockWebSocket', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    useStockStore.setState({ realtimePrice: null })
  })
  afterEach(() => vi.useRealTimers())

  it('returns isConnected true after mount', () => {
    const { result } = renderHook(() => useStockWebSocket('005930'))
    act(() => vi.runAllTimers())
    expect(result.current.isConnected).toBe(true)
  })

  it('updates realtimePrice in store after interval', () => {
    renderHook(() => useStockWebSocket('005930'))
    act(() => vi.advanceTimersByTime(3000))
    const price = useStockStore.getState().realtimePrice
    expect(price).not.toBeNull()
    expect(price?.price).toBeGreaterThan(0)
  })

  it('reconnects when stockCode changes — new connection becomes live', () => {
    const { result, rerender } = renderHook(({ code }) => useStockWebSocket(code), {
      initialProps: { code: '005930' },
    })
    act(() => vi.advanceTimersByTime(1))   // first connection opens
    expect(result.current.isConnected).toBe(true)
    rerender({ code: '000660' })
    // cleanup runs (setIsConnected(false)), new ws created
    act(() => vi.advanceTimersByTime(1))   // new connection opens
    expect(result.current.isConnected).toBe(true)
  })
})
```

- [ ] **Step 2: useStockWebSocket.ts 구현**

`frontend/src/hooks/useStockWebSocket.ts`:

```typescript
import { useEffect, useState } from 'react'
import { MockWebSocket } from '@/lib/mockWebSocket'
import { useStockStore } from '@/store/stockStore'

const WS_BASE = import.meta.env.VITE_WS_BASE ?? null

export function useStockWebSocket(stockCode: string): { isConnected: boolean } {
  const [isConnected, setIsConnected] = useState(false)
  const updateRealtimePrice = useStockStore((s) => s.updateRealtimePrice)

  useEffect(() => {
    const url = WS_BASE
      ? `${WS_BASE}/ws/stocks/${stockCode}`
      : `mock://stocks/${stockCode}`

    const ws = WS_BASE
      ? (new WebSocket(url) as unknown as MockWebSocket)
      : new MockWebSocket(url)

    ws.onopen = () => setIsConnected(true)
    ws.onclose = () => setIsConnected(false)
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        updateRealtimePrice({ code: stockCode, price: data.price, change_pct: data.change_pct })
      } catch {}
    }

    return () => {
      ws.close()
      setIsConnected(false)
    }
  }, [stockCode, updateRealtimePrice])

  return { isConnected }
}
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `54 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/hooks/useStockWebSocket.ts frontend/src/test/useStockWebSocket.test.ts
git commit -m "feat: add useStockWebSocket hook with MockWebSocket"
git push origin seogu-Jeong
```

---

## Task 7: StockInfoBar 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx`
- Create: `frontend/src/test/StockInfoBar.test.tsx`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/StockInfoBar.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { StockInfoBar } from '@/components/MainPanel/ChartTab/StockInfoBar'
import type { Stock, StockDetail } from '@/types'

const mockStock: Stock = { code: '005930', name: '삼성전자', price: 73400, change_pct: 1.2 }
const mockDetail: StockDetail = { open: 72800, high: 74200, low: 72100, volume: 12300000 }

describe('StockInfoBar', () => {
  it('renders stock name and code', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText('삼성전자')).toBeInTheDocument()
    expect(screen.getByText('005930')).toBeInTheDocument()
  })

  it('renders current price', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText(/73,400/)).toBeInTheDocument()
  })

  it('renders OHLV detail cards', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText('시가')).toBeInTheDocument()
    expect(screen.getByText('고가')).toBeInTheDocument()
    expect(screen.getByText('저가')).toBeInTheDocument()
    expect(screen.getByText('거래량')).toBeInTheDocument()
  })

  it('shows live indicator when isLive is true', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText('실시간')).toBeInTheDocument()
  })

  it('shows green color for positive change', () => {
    const { container } = render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={false} />)
    expect(container.querySelector('.text-green-500')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: StockInfoBar.tsx 구현**

`frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx`:

```tsx
import type { Stock, StockDetail } from '@/types'
import { cn } from '@/lib/utils'

interface StockInfoBarProps {
  stock: Stock
  detail: StockDetail
  isLive: boolean
  realtimePrice?: number | null
  realtimeChangePct?: number | null
}

export function StockInfoBar({ stock, detail, isLive, realtimePrice, realtimeChangePct }: StockInfoBarProps) {
  const price = realtimePrice ?? stock.price ?? 0
  const changePct = realtimeChangePct ?? stock.change_pct ?? 0
  const isPositive = changePct >= 0

  return (
    <div className="bg-card border-b border-border px-4 py-2 shrink-0">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-bold text-base">{stock.name}</span>
          <span className="text-muted-foreground text-sm">{stock.code}</span>
          {isLive && (
            <span className="text-xs bg-green-500/20 text-green-500 px-1.5 py-0.5 rounded">
              실시간
            </span>
          )}
        </div>
        <div className="text-right">
          <span className={cn('font-bold text-xl', isPositive ? 'text-green-500' : 'text-red-500')}>
            {price.toLocaleString()}
          </span>
          <span className={cn('ml-2 text-sm', isPositive ? 'text-green-500' : 'text-red-500')}>
            {isPositive ? '▲' : '▼'}{Math.abs(changePct).toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2">
        {[
          { label: '시가', value: detail.open, color: 'text-foreground' },
          { label: '고가', value: detail.high, color: 'text-green-500' },
          { label: '저가', value: detail.low, color: 'text-red-500' },
          { label: '거래량', value: detail.volume, color: 'text-foreground', isVolume: true },
        ].map(({ label, value, color, isVolume }) => (
          <div key={label} className="bg-background rounded px-2 py-1 text-center">
            <div className="text-muted-foreground text-xs mb-0.5">{label}</div>
            <div className={cn('text-xs font-semibold', color)}>
              {isVolume ? `${(value / 1000000).toFixed(1)}M` : value.toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `59 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx frontend/src/test/StockInfoBar.test.tsx
git commit -m "feat: add StockInfoBar with OHLV card grid"
git push origin seogu-Jeong
```

---

## Task 8: CandleChart 컴포넌트 (기존 로직 분리)

**Files:**
- Create: `frontend/src/components/MainPanel/ChartTab/CandleChart.tsx`

- [ ] **Step 1: CandleChart.tsx 생성 (기존 ChartTab 로직 이동)**

`frontend/src/components/MainPanel/ChartTab/CandleChart.tsx`:

```tsx
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'
import type { Candle } from '@/types'

interface CandleChartProps {
  candles: Candle[]
}

export function CandleChart({ candles }: CandleChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current) return

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: chartRef.current.clientWidth,
      height: chartRef.current.clientHeight,
    })

    const series = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    series.setData(candles)
    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chartRef.current) {
        chart.applyOptions({ width: chartRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [candles])

  return <div ref={chartRef} className="w-full h-full" />
}
```

- [ ] **Step 2: 테스트 실행 (기존 테스트 회귀 없음 확인)**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `59 passed` (CandleChart은 DOM 의존으로 단위 테스트 없음)

- [ ] **Step 3: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/CandleChart.tsx
git commit -m "feat: extract CandleChart component from ChartTab"
git push origin seogu-Jeong
```

---

## Task 9: RSIChart 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/ChartTab/RSIChart.tsx`

- [ ] **Step 1: RSIChart.tsx 생성**

`frontend/src/components/MainPanel/ChartTab/RSIChart.tsx`:

```tsx
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'

interface RSIChartProps {
  data: { time: string; value: number }[]
}

export function RSIChart({ data }: RSIChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || data.length === 0) return

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: chartRef.current.clientWidth,
      height: chartRef.current.clientHeight,
      rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.1 } },
      timeScale: { visible: false },
    })

    const series = chart.addLineSeries({ color: '#58a6ff', lineWidth: 1 })
    series.setData(data)

    // 과매수(70) / 과매도(30) 기준선
    const overbought = chart.addLineSeries({ color: '#f85149', lineWidth: 1, lineStyle: 2 })
    const oversold = chart.addLineSeries({ color: '#3fb950', lineWidth: 1, lineStyle: 2 })
    if (data.length > 0) {
      overbought.setData([{ time: data[0].time, value: 70 }, { time: data[data.length - 1].time, value: 70 }])
      oversold.setData([{ time: data[0].time, value: 30 }, { time: data[data.length - 1].time, value: 30 }])
    }

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth })
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data])

  return (
    <div className="w-full h-full relative">
      <span className="absolute top-1 left-2 text-xs text-blue-400 font-semibold z-10">RSI (14)</span>
      <div ref={chartRef} className="w-full h-full" />
    </div>
  )
}
```

- [ ] **Step 2: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `59 passed`

- [ ] **Step 3: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/RSIChart.tsx
git commit -m "feat: add RSIChart subchart with overbought/oversold lines"
git push origin seogu-Jeong
```

---

## Task 10: MACDChart 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/ChartTab/MACDChart.tsx`

- [ ] **Step 1: MACDChart.tsx 생성**

`frontend/src/components/MainPanel/ChartTab/MACDChart.tsx`:

```tsx
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'
import type { MACDPoint } from '@/types'

interface MACDChartProps {
  data: MACDPoint[]
}

export function MACDChart({ data }: MACDChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || data.length === 0) return

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: chartRef.current.clientWidth,
      height: chartRef.current.clientHeight,
      timeScale: { visible: false },
    })

    // 히스토그램
    const histogram = chart.addHistogramSeries({
      color: '#3fb950',
      priceFormat: { type: 'price', precision: 0 },
    })
    histogram.setData(
      data.map((d) => ({
        time: d.time,
        value: d.histogram,
        color: d.histogram >= 0 ? '#3fb950' : '#ef4444',
      }))
    )

    // MACD 라인
    const macdLine = chart.addLineSeries({ color: '#58a6ff', lineWidth: 1 })
    macdLine.setData(data.map((d) => ({ time: d.time, value: d.macd })))

    // Signal 라인
    const signalLine = chart.addLineSeries({ color: '#e3b341', lineWidth: 1 })
    signalLine.setData(data.map((d) => ({ time: d.time, value: d.signal })))

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth })
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data])

  return (
    <div className="w-full h-full relative">
      <span className="absolute top-1 left-2 text-xs text-blue-400 font-semibold z-10">MACD</span>
      <div ref={chartRef} className="w-full h-full" />
    </div>
  )
}
```

- [ ] **Step 2: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `59 passed`

- [ ] **Step 3: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/MACDChart.tsx
git commit -m "feat: add MACDChart subchart with histogram and signal line"
git push origin seogu-Jeong
```

---

## Task 11: ChartTab.tsx 리팩터 — 전체 조립

**Files:**
- Modify: `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx`

- [ ] **Step 1: ChartTab.tsx 교체**

`frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` 전체 교체:

```tsx
// frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
import { useStockStore } from '@/store/stockStore'
import { MOCK_CANDLES, MOCK_STOCK_DETAILS } from '@/lib/mockData'
import { calculateRSI, calculateMACD } from '@/lib/indicators'
import { useStockWebSocket } from '@/hooks/useStockWebSocket'
import { StockInfoBar } from './StockInfoBar'
import { CandleChart } from './CandleChart'
import { RSIChart } from './RSIChart'
import { MACDChart } from './MACDChart'
import { useMemo } from 'react'

export function ChartTab() {
  const { selectedStock, realtimePrice } = useStockStore()
  const { isConnected } = useStockWebSocket(selectedStock?.code ?? '')

  const rsiData = useMemo(() => calculateRSI(MOCK_CANDLES), [])
  const macdData = useMemo(() => calculateMACD(MOCK_CANDLES), [])

  const stockCode = selectedStock?.code ?? '005930'
  const detail = MOCK_STOCK_DETAILS[stockCode] ?? MOCK_STOCK_DETAILS['005930']

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {selectedStock && (
        <StockInfoBar
          stock={selectedStock}
          detail={detail}
          isLive={isConnected}
          realtimePrice={realtimePrice?.price}
          realtimeChangePct={realtimePrice?.change_pct}
        />
      )}
      <div className="flex flex-col flex-1 min-h-0 gap-0.5 p-1">
        <div className="flex-[3] min-h-0">
          <CandleChart candles={MOCK_CANDLES} />
        </div>
        <div className="flex-1 min-h-0">
          <RSIChart data={rsiData} />
        </div>
        <div className="flex-1 min-h-0">
          <MACDChart data={macdData} />
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: 전체 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `59 passed` (회귀 없음)

- [ ] **Step 3: 앱 실제 동작 확인**

```bash
npm run dev
```

확인 항목:
1. `http://localhost:5173` → LandingPage
2. "둘러보기 (데모)" → MainLayout
3. 중앙 ChartTab에 StockInfoBar 표시 (삼성전자 시가/고가/저가/거래량 카드)
4. 캔들스틱 차트 (60%)
5. RSI 서브차트 (20%) + 기준선
6. MACD 서브차트 (20%) + 히스토그램
7. StockInfoBar에 "실시간" 뱃지 표시
8. 2초마다 가격 업데이트 확인
9. `Ctrl+C` 종료

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
git commit -m "feat: refactor ChartTab — assemble all Phase 2 components"
git push origin seogu-Jeong
```

---

## Task 12: CLAUDE.md 완료 표시 + dev 머지

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md §12 업데이트**

`~/FinalProject/CLAUDE.md`의 Phase 2 항목을:

```markdown
- [x] Phase 2 — 실시간 시세 + 차트 (프론트엔드 완료 2026-06-02)
```

- [ ] **Step 2: 커밋 + dev 머지**

```bash
cd ~/FinalProject
git add CLAUDE.md
git commit -m "docs: mark Phase 2 frontend complete"
git push origin seogu-Jeong

git checkout dev
git pull origin dev
git merge seogu-Jeong
git push origin dev
git checkout seogu-Jeong
```
