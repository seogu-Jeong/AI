# StockSenseAI Frontend Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 4 프론트엔드 완성 — 주문 모달, 호가창, 포트폴리오, 시뮬레이터, 백테스팅, 리스크 설정

**Architecture:** 우측 패널(w-56 placeholder)에 OrderBook을 활성화하고, StockInfoBar에 매수/매도 버튼을 추가해 OrderModal을 트리거한다. PortfolioTab·SimulatorTab·BacktestTab·RiskSettingsModal을 각각 독립 컴포넌트로 구현하고 MainPanel 라우팅에 연결한다.

**Tech Stack:** React 18, TypeScript 5, Recharts 2.x, shadcn/ui, Zustand, Vitest

---

## File Map

| 파일 | 역할 | 상태 |
|---|---|---|
| `frontend/src/types/index.ts` | OrderBookEntry, PortfolioMetrics 타입 | 수정 |
| `frontend/src/lib/mockData.ts` | Phase 4 Mock 데이터 6종 | 수정 |
| `frontend/src/components/Trade/OrderBook.tsx` | 호가창 10단 | 신규 |
| `frontend/src/components/Trade/OrderModal.tsx` | 주문 모달 | 신규 |
| `frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx` | 매수/매도 버튼 추가 | 수정 |
| `frontend/src/components/Layout/MainLayout.tsx` | 우측 패널 OrderBook 활성화 | 수정 |
| `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx` | 포트폴리오 현황 + 차트 | 신규 |
| `frontend/src/components/MainPanel/SimulatorTab/SimulatorTab.tsx` | 투자 시뮬레이터 | 신규 |
| `frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx` | 백테스팅 UI | 신규 |
| `frontend/src/components/Risk/RiskSettingsModal.tsx` | 리스크 설정 모달 | 신규 |
| `frontend/src/components/Layout/Header.tsx` | ⚙️ 버튼 추가 | 수정 |
| `frontend/src/components/MainPanel/MainPanel.tsx` | 탭 연결 | 수정 |

---

## Task 1: types/index.ts — Phase 4 타입 추가

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/test/types.test.ts`

- [ ] **Step 1: types/index.ts 맨 끝에 추가**

```typescript
export interface OrderBookEntry {
  price: number
  quantity: number
}

export interface PortfolioMetrics {
  total_value: number
  total_return_pct: number
  mdd: number
}

export interface RiskSettings {
  max_position_pct: number   // 1회 최대 투자 비중 (%)
  stop_loss_pct: number      // 손절 기준 (%)
  daily_loss_limit: number   // 일일 최대 손실 한도 (원)
}
```

- [ ] **Step 2: types.test.ts에 테스트 추가**

import 줄에 추가: `OrderBookEntry, PortfolioMetrics`

describe 블록 안에:
```typescript
it('OrderBookEntry has price and quantity', () => {
  const entry: OrderBookEntry = { price: 73400, quantity: 500 }
  expect(entry.price).toBe(73400)
})

it('PortfolioMetrics has total_value, return_pct, mdd', () => {
  const m: PortfolioMetrics = { total_value: 5000000, total_return_pct: 12.5, mdd: -8.3 }
  expect(m.total_return_pct).toBe(12.5)
})
```

- [ ] **Step 3: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `83 passed`

- [ ] **Step 4: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/types/index.ts frontend/src/test/types.test.ts
git commit -m "feat: add Phase 4 types (OrderBookEntry, PortfolioMetrics, RiskSettings)"
git push origin seogu-Jeong
```

---

## Task 2: mockData.ts — Phase 4 Mock 데이터

**Files:**
- Modify: `frontend/src/lib/mockData.ts`
- Modify: `frontend/src/test/mockData.test.ts`

- [ ] **Step 1: mockData.ts import 수정 + 데이터 추가**

상단 import에 `OrderBookEntry, PortfolioMetrics` 추가.

파일 맨 끝에:
```typescript
export const MOCK_ORDER_BOOK = {
  asks: [
    { price: 74200, quantity: 1200 },
    { price: 74100, quantity: 850 },
    { price: 74000, quantity: 2300 },
    { price: 73900, quantity: 600 },
    { price: 73800, quantity: 1800 },
    { price: 73700, quantity: 950 },
    { price: 73600, quantity: 3100 },
    { price: 73500, quantity: 720 },
    { price: 73450, quantity: 1500 },
    { price: 73420, quantity: 2800 },
  ] as OrderBookEntry[],
  bids: [
    { price: 73400, quantity: 4200 },
    { price: 73380, quantity: 1100 },
    { price: 73350, quantity: 2600 },
    { price: 73300, quantity: 800 },
    { price: 73250, quantity: 1900 },
    { price: 73200, quantity: 650 },
    { price: 73150, quantity: 3400 },
    { price: 73100, quantity: 1200 },
    { price: 73050, quantity: 2100 },
    { price: 73000, quantity: 900 },
  ] as OrderBookEntry[],
}

export const MOCK_HOLDINGS: Holding[] = [
  { stock_code: '005930', stock_name: '삼성전자', quantity: 10, avg_price: 70000, current_price: 73400, profit_loss: 34000, return_pct: 4.86, ai_signal: 'BUY' },
  { stock_code: '000660', stock_name: 'SK하이닉스', quantity: 5, avg_price: 180000, current_price: 185000, profit_loss: 25000, return_pct: 2.78, ai_signal: 'HOLD' },
  { stock_code: '035420', stock_name: 'NAVER', quantity: 3, avg_price: 220000, current_price: 210000, profit_loss: -30000, return_pct: -4.55, ai_signal: 'SELL' },
]

export const MOCK_PORTFOLIO_PERFORMANCE: { date: string; value: number }[] = Array.from({ length: 30 }, (_, i) => {
  const date = new Date('2026-05-01')
  date.setDate(date.getDate() + i)
  return {
    date: date.toISOString().split('T')[0],
    value: 5000000 + Math.round((Math.random() - 0.45) * 100000 * (i + 1)),
  }
})

export const MOCK_PORTFOLIO_METRICS: PortfolioMetrics = {
  total_value: 5290000,
  total_return_pct: 5.8,
  mdd: -3.2,
}

export const MOCK_BACKTEST_RESULT = {
  return_pct: 18.4,
  win_rate: 62.5,
  mdd: -7.3,
  trades: 24,
}
```

- [ ] **Step 2: mockData.test.ts에 테스트 추가**

```typescript
it('MOCK_ORDER_BOOK has 10 asks and 10 bids', () => {
  expect(MOCK_ORDER_BOOK.asks).toHaveLength(10)
  expect(MOCK_ORDER_BOOK.bids).toHaveLength(10)
})

it('MOCK_HOLDINGS has at least 1 holding', () => {
  expect(MOCK_HOLDINGS.length).toBeGreaterThan(0)
  expect(MOCK_HOLDINGS[0]).toHaveProperty('stock_code')
})
```

import 줄에 추가: `MOCK_ORDER_BOOK, MOCK_HOLDINGS`

- [ ] **Step 3: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `85 passed`

- [ ] **Step 4: 커밋**
```bash
cd ~/FinalProject
git add -f frontend/src/lib/mockData.ts frontend/src/test/mockData.test.ts
git commit -m "feat: add Phase 4 mock data (orderbook, holdings, portfolio, backtest)"
git push origin seogu-Jeong
```

---

## Task 3: shadcn 추가 컴포넌트 설치

**Files:**
- Create: `frontend/src/components/ui/sonner.tsx` (또는 toast)
- Create: `frontend/src/components/ui/select.tsx`

- [ ] **Step 1: shadcn Select 설치**
```bash
cd ~/FinalProject/frontend
npx shadcn@latest add select
```
설치 실패 시 수동 생성 (아래 참고):
`frontend/src/components/ui/select.tsx`:
```tsx
import * as React from "react"
import * as SelectPrimitive from "@radix-ui/react-select"
import { Check, ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

const Select = SelectPrimitive.Root
const SelectGroup = SelectPrimitive.Group
const SelectValue = SelectPrimitive.Value

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
))
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        "relative z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md",
        position === "popper" && "translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <SelectPrimitive.Viewport className="p-1">{children}</SelectPrimitive.Viewport>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
))
SelectContent.displayName = SelectPrimitive.Content.displayName

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
))
SelectItem.displayName = SelectPrimitive.Item.displayName

const SelectLabel = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Label>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn("px-8 py-1.5 text-sm font-semibold", className)}
    {...props}
  />
))
SelectLabel.displayName = SelectPrimitive.Label.displayName

export { Select, SelectGroup, SelectValue, SelectTrigger, SelectContent, SelectItem, SelectLabel }
```

- [ ] **Step 2: Radix Select 설치**
```bash
cd ~/FinalProject/frontend
npm install @radix-ui/react-select
```

- [ ] **Step 3: 테스트 실행 (회귀 없음)**
```bash
npm run test:run
```
Expected: `85 passed`

- [ ] **Step 4: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/components/ui/select.tsx frontend/package.json frontend/package-lock.json
git commit -m "feat: add Select shadcn component for Phase 4"
git push origin seogu-Jeong
```

---

## Task 4: OrderBook + OrderModal

**Files:**
- Create: `frontend/src/components/Trade/OrderBook.tsx`
- Create: `frontend/src/components/Trade/OrderModal.tsx`
- Create: `frontend/src/test/OrderBook.test.tsx`
- Create: `frontend/src/test/OrderModal.test.tsx`

- [ ] **Step 1: OrderBook 테스트**

`frontend/src/test/OrderBook.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { OrderBook } from '@/components/Trade/OrderBook'
import { MOCK_ORDER_BOOK } from '@/lib/mockData'

describe('OrderBook', () => {
  it('renders 매도/매수 labels', () => {
    render(<OrderBook asks={MOCK_ORDER_BOOK.asks} bids={MOCK_ORDER_BOOK.bids} currentPrice={73400} />)
    expect(screen.getByText('매도')).toBeInTheDocument()
    expect(screen.getByText('매수')).toBeInTheDocument()
  })

  it('renders 10 ask rows', () => {
    render(<OrderBook asks={MOCK_ORDER_BOOK.asks} bids={MOCK_ORDER_BOOK.bids} currentPrice={73400} />)
    const rows = screen.getAllByText(/74,200|74,100|74,000/)
    expect(rows.length).toBeGreaterThan(0)
  })

  it('renders current price', () => {
    render(<OrderBook asks={MOCK_ORDER_BOOK.asks} bids={MOCK_ORDER_BOOK.bids} currentPrice={73400} />)
    expect(screen.getByText(/73,400/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: OrderBook 구현**

`frontend/src/components/Trade/OrderBook.tsx`:
```tsx
import type { OrderBookEntry } from '@/types'
import { cn } from '@/lib/utils'

interface OrderBookProps {
  asks: OrderBookEntry[]
  bids: OrderBookEntry[]
  currentPrice: number
}

export function OrderBook({ asks, bids, currentPrice }: OrderBookProps) {
  const maxQty = Math.max(...asks.map((a) => a.quantity), ...bids.map((b) => b.quantity))

  return (
    <div className="flex flex-col h-full text-xs overflow-hidden">
      <div className="px-2 py-1.5 border-b border-border font-semibold text-xs">호가</div>

      {/* 매도호가 - 위에서 아래로 (높은가격→낮은가격) */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="text-center text-red-400 text-xs py-0.5 bg-red-400/5">매도</div>
        <div className="flex-1 overflow-y-auto">
          {[...asks].reverse().map((ask, i) => (
            <div key={i} className="relative flex items-center justify-between px-2 py-0.5 hover:bg-accent/50">
              <div
                className="absolute right-0 top-0 bottom-0 bg-red-400/10"
                style={{ width: `${(ask.quantity / maxQty) * 100}%` }}
              />
              <span className="text-red-400 font-medium relative z-10">{ask.price.toLocaleString()}</span>
              <span className="text-muted-foreground relative z-10">{ask.quantity.toLocaleString()}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 현재가 */}
      <div className="text-center py-1 border-y border-border bg-card font-bold text-sm">
        {currentPrice.toLocaleString()}
      </div>

      {/* 매수호가 */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto">
          {bids.map((bid, i) => (
            <div key={i} className="relative flex items-center justify-between px-2 py-0.5 hover:bg-accent/50">
              <div
                className="absolute right-0 top-0 bottom-0 bg-green-400/10"
                style={{ width: `${(bid.quantity / maxQty) * 100}%` }}
              />
              <span className="text-green-400 font-medium relative z-10">{bid.price.toLocaleString()}</span>
              <span className="text-muted-foreground relative z-10">{bid.quantity.toLocaleString()}</span>
            </div>
          ))}
        </div>
        <div className="text-center text-green-400 text-xs py-0.5 bg-green-400/5">매수</div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: OrderModal 테스트**

`frontend/src/test/OrderModal.test.tsx`:
```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { OrderModal } from '@/components/Trade/OrderModal'
import { MOCK_STOCKS } from '@/lib/mockData'

const stock = MOCK_STOCKS[0]

describe('OrderModal', () => {
  it('renders BUY modal title', () => {
    render(<OrderModal open={true} onClose={() => {}} stock={stock} orderType="BUY" />)
    expect(screen.getByText(/매수/)).toBeInTheDocument()
  })

  it('renders SELL modal title', () => {
    render(<OrderModal open={true} onClose={() => {}} stock={stock} orderType="SELL" />)
    expect(screen.getByText(/매도/)).toBeInTheDocument()
  })

  it('renders stock name', () => {
    render(<OrderModal open={true} onClose={() => {}} stock={stock} orderType="BUY" />)
    expect(screen.getByText('삼성전자')).toBeInTheDocument()
  })

  it('calls onClose when cancelled', () => {
    const fn = vi.fn()
    render(<OrderModal open={true} onClose={fn} stock={stock} orderType="BUY" />)
    fireEvent.click(screen.getByText('취소'))
    expect(fn).toHaveBeenCalled()
  })
})
```

- [ ] **Step 4: OrderModal 구현**

`frontend/src/components/Trade/OrderModal.tsx`:
```tsx
import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { Stock } from '@/types'
import { cn } from '@/lib/utils'

interface OrderModalProps {
  open: boolean
  onClose: () => void
  stock: Stock
  orderType: 'BUY' | 'SELL'
}

export function OrderModal({ open, onClose, stock, orderType }: OrderModalProps) {
  const [priceType, setPriceType] = useState<'MARKET' | 'LIMIT'>('LIMIT')
  const [quantity, setQuantity] = useState('1')
  const [price, setPrice] = useState(String(stock.price ?? 0))
  const [submitted, setSubmitted] = useState(false)

  const isBuy = orderType === 'BUY'
  const total = Number(quantity) * Number(price)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitted(true)
    setTimeout(() => {
      setSubmitted(false)
      onClose()
    }, 1500)
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle className={cn(isBuy ? 'text-green-400' : 'text-red-400')}>
            {stock.name} {isBuy ? '매수' : '매도'}
          </DialogTitle>
        </DialogHeader>

        {submitted ? (
          <div className="text-center py-6">
            <div className={cn('text-2xl font-bold mb-2', isBuy ? 'text-green-400' : 'text-red-400')}>
              {isBuy ? '매수 완료' : '매도 완료'}
            </div>
            <div className="text-sm text-muted-foreground">모의거래 체결됨</div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-3">
            {/* 주문 타입 */}
            <div className="flex gap-2">
              {(['LIMIT', 'MARKET'] as const).map((type) => (
                <Button
                  key={type}
                  type="button"
                  variant={priceType === type ? 'default' : 'outline'}
                  size="sm"
                  className="flex-1"
                  onClick={() => setPriceType(type)}
                >
                  {type === 'LIMIT' ? '지정가' : '시장가'}
                </Button>
              ))}
            </div>

            {priceType === 'LIMIT' && (
              <div>
                <label className="text-xs text-muted-foreground">주문가격</label>
                <Input
                  type="number"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  className="mt-1"
                />
              </div>
            )}

            <div>
              <label className="text-xs text-muted-foreground">수량</label>
              <Input
                type="number"
                min="1"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                className="mt-1"
              />
            </div>

            {priceType === 'LIMIT' && (
              <div className="text-sm text-right text-muted-foreground">
                주문금액: <span className="text-foreground font-medium">{total.toLocaleString()}원</span>
              </div>
            )}

            <div className="flex gap-2 pt-1">
              <Button type="button" variant="outline" className="flex-1" onClick={onClose}>취소</Button>
              <Button
                type="submit"
                className={cn('flex-1', isBuy ? 'bg-green-500 hover:bg-green-600' : 'bg-red-500 hover:bg-red-600')}
              >
                {isBuy ? '매수' : '매도'}
              </Button>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}
```

- [ ] **Step 5: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `92 passed`

- [ ] **Step 6: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/components/Trade/ frontend/src/test/OrderBook.test.tsx frontend/src/test/OrderModal.test.tsx
git commit -m "feat: add OrderBook and OrderModal components"
git push origin seogu-Jeong
```

---

## Task 5: StockInfoBar 매수/매도 버튼 + MainLayout 우측 패널 활성화

**Files:**
- Modify: `frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx`
- Modify: `frontend/src/components/Layout/MainLayout.tsx`

- [ ] **Step 1: StockInfoBar에 매수/매도 버튼 추가**

`frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx` 수정:

상단에 import 추가:
```tsx
import { useState } from 'react'
import { OrderModal } from '@/components/Trade/OrderModal'
```

`StockInfoBarProps`에 `onOrderClick` 대신 컴포넌트 내부에서 모달 state 관리.

`StockInfoBar` 함수 안 state 추가 (return문 바로 위):
```tsx
const [orderType, setOrderType] = useState<'BUY' | 'SELL' | null>(null)
```

카드 그리드 div 바로 아래, return closing `</div>` 바로 위에 추가:
```tsx
      {/* 매수/매도 버튼 */}
      <div className="flex gap-2 mt-2">
        <Button
          size="sm"
          className="flex-1 bg-green-500 hover:bg-green-600 text-white"
          onClick={() => setOrderType('BUY')}
        >
          매수
        </Button>
        <Button
          size="sm"
          className="flex-1 bg-red-500 hover:bg-red-600 text-white"
          onClick={() => setOrderType('SELL')}
        >
          매도
        </Button>
      </div>
      {orderType && (
        <OrderModal
          open={true}
          onClose={() => setOrderType(null)}
          stock={stock}
          orderType={orderType}
        />
      )}
```

- [ ] **Step 2: MainLayout 우측 패널에 OrderBook 활성화**

`frontend/src/components/Layout/MainLayout.tsx` 수정:

상단에 import 추가:
```tsx
import { OrderBook } from '@/components/Trade/OrderBook'
import { MOCK_ORDER_BOOK } from '@/lib/mockData'
import { useStockStore } from '@/store/stockStore'
```

`useUIStore` 아래에:
```tsx
const { selectedStock, realtimePrice } = useStockStore()
```

우측 패널 div를:
```tsx
{/* 기존 */}
<div className="hidden lg:block w-56 shrink-0 bg-card border-l border-border" />

{/* 교체 */}
<div className="hidden lg:flex lg:flex-col w-56 shrink-0 bg-card border-l border-border overflow-hidden">
  <OrderBook
    asks={MOCK_ORDER_BOOK.asks}
    bids={MOCK_ORDER_BOOK.bids}
    currentPrice={realtimePrice?.price ?? selectedStock?.price ?? 73400}
  />
</div>
```

- [ ] **Step 3: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `92 passed`

- [ ] **Step 4: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx frontend/src/components/Layout/MainLayout.tsx
git commit -m "feat: add order buttons to StockInfoBar and activate OrderBook in right panel"
git push origin seogu-Jeong
```

---

## Task 6: PortfolioTab

**Files:**
- Create: `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx`
- Create: `frontend/src/test/PortfolioTab.test.tsx`

- [ ] **Step 1: PortfolioTab 테스트**

`frontend/src/test/PortfolioTab.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { PortfolioTab } from '@/components/MainPanel/PortfolioTab/PortfolioTab'

describe('PortfolioTab', () => {
  it('renders 총 평가액', () => {
    render(<PortfolioTab />)
    expect(screen.getByText('총 평가액')).toBeInTheDocument()
  })

  it('renders 보유종목 heading', () => {
    render(<PortfolioTab />)
    expect(screen.getByText('보유종목')).toBeInTheDocument()
  })

  it('renders holding stock names', () => {
    render(<PortfolioTab />)
    expect(screen.getByText('삼성전자')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: PortfolioTab 구현**

`frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx`:
```tsx
import { MOCK_HOLDINGS, MOCK_PORTFOLIO_METRICS, MOCK_PORTFOLIO_PERFORMANCE } from '@/lib/mockData'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { cn } from '@/lib/utils'

export function PortfolioTab() {
  const { total_value, total_return_pct, mdd } = MOCK_PORTFOLIO_METRICS

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* 요약 카드 */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <div className="text-xs text-muted-foreground mb-1">총 평가액</div>
          <div className="text-sm font-bold">{(total_value / 10000).toFixed(0)}만원</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <div className="text-xs text-muted-foreground mb-1">수익률</div>
          <div className={cn('text-sm font-bold', total_return_pct >= 0 ? 'text-green-400' : 'text-red-400')}>
            {total_return_pct >= 0 ? '+' : ''}{total_return_pct.toFixed(1)}%
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <div className="text-xs text-muted-foreground mb-1">MDD</div>
          <div className="text-sm font-bold text-red-400">{mdd.toFixed(1)}%</div>
        </div>
      </div>

      {/* 수익률 차트 */}
      <div className="bg-card border border-border rounded-lg p-3">
        <div className="text-sm font-semibold mb-3">평가금액 추이</div>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={MOCK_PORTFOLIO_PERFORMANCE}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={(v) => v.slice(5)} />
            <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={(v) => `${(v / 10000).toFixed(0)}만`} />
            <Tooltip
              formatter={(v: number) => [`${v.toLocaleString()}원`, '평가금액']}
              contentStyle={{ background: '#161b22', border: '1px solid #30363d', fontSize: 11 }}
            />
            <Line type="monotone" dataKey="value" stroke="#58a6ff" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 보유종목 */}
      <div className="bg-card border border-border rounded-lg p-3">
        <div className="text-sm font-semibold mb-3">보유종목</div>
        <div className="space-y-2">
          {MOCK_HOLDINGS.map((h) => (
            <div key={h.stock_code} className="flex items-center justify-between text-sm py-1.5 border-b border-border last:border-0">
              <div>
                <div className="font-medium">{h.stock_name}</div>
                <div className="text-xs text-muted-foreground">{h.quantity}주 · 평균 {h.avg_price.toLocaleString()}원</div>
              </div>
              <div className="text-right">
                <div>{h.current_price.toLocaleString()}원</div>
                <div className={cn('text-xs', h.return_pct >= 0 ? 'text-green-400' : 'text-red-400')}>
                  {h.return_pct >= 0 ? '+' : ''}{h.return_pct.toFixed(2)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `95 passed`

- [ ] **Step 4: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/PortfolioTab/ frontend/src/test/PortfolioTab.test.tsx
git commit -m "feat: add PortfolioTab with holdings table and performance chart"
git push origin seogu-Jeong
```

---

## Task 7: SimulatorTab + BacktestTab

**Files:**
- Create: `frontend/src/components/MainPanel/SimulatorTab/SimulatorTab.tsx`
- Create: `frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx`
- Create: `frontend/src/test/SimulatorTab.test.tsx`
- Create: `frontend/src/test/BacktestTab.test.tsx`

- [ ] **Step 1: SimulatorTab 테스트**

`frontend/src/test/SimulatorTab.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { SimulatorTab } from '@/components/MainPanel/SimulatorTab/SimulatorTab'

describe('SimulatorTab', () => {
  it('renders 투자 시뮬레이터 heading', () => {
    render(<SimulatorTab />)
    expect(screen.getByText('투자 시뮬레이터')).toBeInTheDocument()
  })

  it('renders 투자금액 input label', () => {
    render(<SimulatorTab />)
    expect(screen.getByText('투자금액')).toBeInTheDocument()
  })

  it('renders simulate button', () => {
    render(<SimulatorTab />)
    expect(screen.getByRole('button', { name: /시뮬레이션/ })).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: SimulatorTab 구현**

`frontend/src/components/MainPanel/SimulatorTab/SimulatorTab.tsx`:
```tsx
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { MOCK_STOCKS, MOCK_CANDLES } from '@/lib/mockData'
import { cn } from '@/lib/utils'

export function SimulatorTab() {
  const [buyDate, setBuyDate] = useState('2026-01-02')
  const [sellDate, setSellDate] = useState('2026-03-01')
  const [amount, setAmount] = useState('1000000')
  const [result, setResult] = useState<{ profit: number; returnPct: number } | null>(null)

  const handleSimulate = (e: React.FormEvent) => {
    e.preventDefault()
    const buyCandle = MOCK_CANDLES.find((c) => c.time >= buyDate)
    const sellCandle = [...MOCK_CANDLES].reverse().find((c) => c.time <= sellDate)
    if (!buyCandle || !sellCandle) return

    const shares = Math.floor(Number(amount) / buyCandle.close)
    const profit = shares * (sellCandle.close - buyCandle.close)
    const returnPct = ((sellCandle.close - buyCandle.close) / buyCandle.close) * 100
    setResult({ profit, returnPct })
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      <h2 className="text-base font-semibold mb-4">투자 시뮬레이터</h2>

      <form onSubmit={handleSimulate} className="space-y-3 bg-card border border-border rounded-lg p-4">
        <div>
          <label className="text-xs text-muted-foreground">종목</label>
          <div className="mt-1 text-sm font-medium">{MOCK_STOCKS[0].name} (삼성전자)</div>
        </div>
        <div>
          <label className="text-xs text-muted-foreground">매수일</label>
          <Input type="date" value={buyDate} onChange={(e) => setBuyDate(e.target.value)} className="mt-1" />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">매도일</label>
          <Input type="date" value={sellDate} onChange={(e) => setSellDate(e.target.value)} className="mt-1" />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">투자금액</label>
          <Input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} className="mt-1" />
        </div>
        <Button type="submit" className="w-full">시뮬레이션 실행</Button>
      </form>

      {result && (
        <div className="mt-4 bg-card border border-border rounded-lg p-4 space-y-2">
          <div className="text-sm font-semibold">시뮬레이션 결과</div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">수익/손실</span>
            <span className={cn('font-medium', result.profit >= 0 ? 'text-green-400' : 'text-red-400')}>
              {result.profit >= 0 ? '+' : ''}{result.profit.toLocaleString()}원
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">수익률</span>
            <span className={cn('font-medium', result.returnPct >= 0 ? 'text-green-400' : 'text-red-400')}>
              {result.returnPct >= 0 ? '+' : ''}{result.returnPct.toFixed(2)}%
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: BacktestTab 테스트**

`frontend/src/test/BacktestTab.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { BacktestTab } from '@/components/MainPanel/BacktestTab/BacktestTab'

describe('BacktestTab', () => {
  it('renders 백테스팅 heading', () => {
    render(<BacktestTab />)
    expect(screen.getByText('백테스팅')).toBeInTheDocument()
  })

  it('renders run button', () => {
    render(<BacktestTab />)
    expect(screen.getByRole('button', { name: /백테스트 실행/ })).toBeInTheDocument()
  })
})
```

- [ ] **Step 4: BacktestTab 구현**

`frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx`:
```tsx
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { MOCK_BACKTEST_RESULT } from '@/lib/mockData'
import { cn } from '@/lib/utils'

export function BacktestTab() {
  const [startDate, setStartDate] = useState('2026-01-02')
  const [endDate, setEndDate] = useState('2026-04-01')
  const [strategy, setStrategy] = useState('MA교차')
  const [ran, setRan] = useState(false)

  const handleRun = (e: React.FormEvent) => {
    e.preventDefault()
    setRan(true)
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      <h2 className="text-base font-semibold mb-4">백테스팅</h2>

      <form onSubmit={handleRun} className="space-y-3 bg-card border border-border rounded-lg p-4">
        <div>
          <label className="text-xs text-muted-foreground">전략</label>
          <div className="flex gap-2 mt-1">
            {['MA교차', 'RSI'].map((s) => (
              <Button key={s} type="button" size="sm" variant={strategy === s ? 'default' : 'outline'} onClick={() => setStrategy(s)}>
                {s}
              </Button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-xs text-muted-foreground">시작일</label>
          <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="mt-1" />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">종료일</label>
          <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="mt-1" />
        </div>
        <Button type="submit" className="w-full">백테스트 실행</Button>
      </form>

      {ran && (
        <div className="mt-4 bg-card border border-border rounded-lg p-4">
          <div className="text-sm font-semibold mb-3">결과 ({strategy})</div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: '수익률', value: `+${MOCK_BACKTEST_RESULT.return_pct}%`, positive: true },
              { label: '승률', value: `${MOCK_BACKTEST_RESULT.win_rate}%`, positive: true },
              { label: 'MDD', value: `-${MOCK_BACKTEST_RESULT.mdd}%`, positive: false },
              { label: '거래 횟수', value: `${MOCK_BACKTEST_RESULT.trades}회`, positive: null },
            ].map(({ label, value, positive }) => (
              <div key={label} className="bg-background border border-border rounded p-2.5 text-center">
                <div className="text-xs text-muted-foreground mb-1">{label}</div>
                <div className={cn('text-sm font-bold',
                  positive === true ? 'text-green-400' : positive === false ? 'text-red-400' : 'text-foreground'
                )}>
                  {value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 5: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `100 passed`

- [ ] **Step 6: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/SimulatorTab/ frontend/src/components/MainPanel/BacktestTab/ frontend/src/test/SimulatorTab.test.tsx frontend/src/test/BacktestTab.test.tsx
git commit -m "feat: add SimulatorTab and BacktestTab"
git push origin seogu-Jeong
```

---

## Task 8: RiskSettingsModal + Header ⚙️ 버튼

**Files:**
- Create: `frontend/src/components/Risk/RiskSettingsModal.tsx`
- Modify: `frontend/src/components/Layout/Header.tsx`
- Create: `frontend/src/test/RiskSettingsModal.test.tsx`

- [ ] **Step 1: RiskSettingsModal 테스트**

`frontend/src/test/RiskSettingsModal.test.tsx`:
```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { RiskSettingsModal } from '@/components/Risk/RiskSettingsModal'

describe('RiskSettingsModal', () => {
  it('renders when open', () => {
    render(<RiskSettingsModal open={true} onClose={() => {}} />)
    expect(screen.getByText('리스크 설정')).toBeInTheDocument()
  })

  it('does not render when closed', () => {
    render(<RiskSettingsModal open={false} onClose={() => {}} />)
    expect(screen.queryByText('리스크 설정')).not.toBeInTheDocument()
  })

  it('calls onClose when saved', () => {
    const fn = vi.fn()
    render(<RiskSettingsModal open={true} onClose={fn} />)
    fireEvent.click(screen.getByText('저장'))
    expect(fn).toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: RiskSettingsModal 구현**

`frontend/src/components/Risk/RiskSettingsModal.tsx`:
```tsx
import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface RiskSettingsModalProps {
  open: boolean
  onClose: () => void
}

export function RiskSettingsModal({ open, onClose }: RiskSettingsModalProps) {
  const [maxPosition, setMaxPosition] = useState('20')
  const [stopLoss, setStopLoss] = useState('5')
  const [dailyLimit, setDailyLimit] = useState('500000')

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault()
    onClose()
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>리스크 설정</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSave} className="space-y-3">
          <div>
            <label className="text-xs text-muted-foreground">1회 최대 투자 비중 (%)</label>
            <Input type="number" value={maxPosition} onChange={(e) => setMaxPosition(e.target.value)} className="mt-1" min="1" max="100" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">손절 기준 (%)</label>
            <Input type="number" value={stopLoss} onChange={(e) => setStopLoss(e.target.value)} className="mt-1" min="1" max="50" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">일일 최대 손실 한도 (원)</label>
            <Input type="number" value={dailyLimit} onChange={(e) => setDailyLimit(e.target.value)} className="mt-1" />
          </div>
          <Button type="submit" className="w-full">저장</Button>
        </form>
      </DialogContent>
    </Dialog>
  )
}
```

- [ ] **Step 3: Header에 ⚙️ 버튼 추가**

`frontend/src/components/Layout/Header.tsx`:

상단 import에 추가:
```tsx
import { Search, Sun, Moon, User, Settings } from 'lucide-react'
import { RiskSettingsModal } from '@/components/Risk/RiskSettingsModal'
```

`onLoginClick` prop 다음에 state 추가:
```tsx
const [riskOpen, setRiskOpen] = useState(false)
```

버튼들 영역에 Settings 버튼 추가 (다크모드 버튼 앞에):
```tsx
<Button variant="ghost" size="icon" onClick={() => setRiskOpen(true)} aria-label="리스크 설정">
  <Settings className="h-4 w-4" />
</Button>
```

return 끝 부분에 모달 추가 (header 태그 닫기 전):
```tsx
<RiskSettingsModal open={riskOpen} onClose={() => setRiskOpen(false)} />
```

- [ ] **Step 4: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `103 passed`

- [ ] **Step 5: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/components/Risk/ frontend/src/components/Layout/Header.tsx frontend/src/test/RiskSettingsModal.test.tsx
git commit -m "feat: add RiskSettingsModal and settings button to Header"
git push origin seogu-Jeong
```

---

## Task 9: MainPanel — 모든 탭 연결

**Files:**
- Modify: `frontend/src/components/MainPanel/MainPanel.tsx`

- [ ] **Step 1: MainPanel.tsx 교체**

```tsx
// frontend/src/components/MainPanel/MainPanel.tsx
import { useUIStore } from '@/store/uiStore'
import { ChartTab } from './ChartTab/ChartTab'
import { AITab } from './AITab/AITab'
import { PortfolioTab } from './PortfolioTab/PortfolioTab'
import { SimulatorTab } from './SimulatorTab/SimulatorTab'
import { BacktestTab } from './BacktestTab/BacktestTab'
import { cn } from '@/lib/utils'
import type { TabId } from '@/types'

const ALL_TABS: { id: TabId; label: string }[] = [
  { id: 'chart', label: '차트' },
  { id: 'ai', label: 'AI' },
  { id: 'simulator', label: '시뮬' },
  { id: 'portfolio', label: '포트폴리오' },
  { id: 'screener', label: '스크리너' },
  { id: 'backtest', label: '백테스트' },
]

function PlaceholderTab({ name }: { name: string }) {
  return (
    <div className="flex items-center justify-center h-full text-muted-foreground">
      {name} — 준비 중
    </div>
  )
}

export function MainPanel() {
  const { activeTab, setActiveTab } = useUIStore()

  return (
    <div className="flex-1 min-w-0 min-h-0 overflow-hidden flex flex-col">
      <div className="hidden md:flex border-b border-border bg-card shrink-0 overflow-x-auto">
        {ALL_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'px-4 h-9 text-sm whitespace-nowrap shrink-0 border-b-2 transition-colors',
              activeTab === tab.id
                ? 'border-primary text-foreground font-medium'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === 'chart'     && <ChartTab />}
        {activeTab === 'ai'        && <AITab />}
        {activeTab === 'simulator' && <SimulatorTab />}
        {activeTab === 'portfolio' && <PortfolioTab />}
        {activeTab === 'screener'  && <PlaceholderTab name="스크리너" />}
        {activeTab === 'backtest'  && <BacktestTab />}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: 테스트 실행**
```bash
cd ~/FinalProject/frontend && npm run test:run
```
Expected: `103 passed`

- [ ] **Step 3: dev 서버 동작 확인**
```bash
cd ~/FinalProject/frontend && npm run dev
```
확인:
1. 차트탭 → 우측 호가창, StockInfoBar에 매수/매도 버튼
2. AI 탭 → 시그널 카드
3. 시뮬 탭 → 날짜/금액 입력 → 결과
4. 포트폴리오 탭 → 보유종목 + 차트
5. 백테스트 탭 → 실행 → 결과
6. Header ⚙️ → 리스크 설정 모달
7. Ctrl+C 종료

- [ ] **Step 4: 커밋**
```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/MainPanel.tsx
git commit -m "feat: connect all tabs in MainPanel — Phase 4 complete"
git push origin seogu-Jeong
```

---

## Task 10: CLAUDE.md 완료 표시 + dev 머지

- [ ] **Step 1: CLAUDE.md 업데이트**

```
- [x] Phase 4 — 거래 + 포트폴리오 (프론트엔드 완료 2026-06-02)
```

- [ ] **Step 2: 커밋 + dev 머지**

```bash
cd ~/FinalProject
git add CLAUDE.md
git commit -m "docs: mark Phase 4 frontend complete"
git push origin seogu-Jeong

git checkout dev
git pull origin dev
git merge seogu-Jeong
git push origin dev
git checkout seogu-Jeong
```
