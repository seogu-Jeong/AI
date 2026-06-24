# StockSenseAI Frontend Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 프론트엔드 완성 — Vite+TS+shadcn 세팅, 3단 레이아웃, 인증, Zustand, Mock 차트탭

**Architecture:** `frontend/` 디렉토리에 Vite+React18+TypeScript5 프로젝트를 생성한다. 전역 상태는 Zustand 3개 스토어(auth/stock/ui)로 분리하고, Axios 인스턴스에 JWT interceptor를 붙인다. 레이아웃은 데스크탑 3단(Sidebar|MainPanel|RightPanel), 모바일 상단 스크롤탭으로 분기한다. 백엔드 미완성 구간은 mockData로 대체한다.

**Tech Stack:** React 18, TypeScript 5, Vite 5, shadcn/ui, Tailwind CSS v4, Zustand 4, Axios 1.6, Lightweight Charts 4, Vitest, React Testing Library

---

## File Map

| 파일 | 역할 |
|---|---|
| `frontend/src/types/index.ts` | 공통 타입 (CLAUDE.md 기준) |
| `frontend/src/lib/utils.ts` | shadcn cn() 유틸 |
| `frontend/src/lib/api.ts` | Axios 인스턴스 + JWT interceptor |
| `frontend/src/lib/mockData.ts` | Mock OHLCV 데이터 |
| `frontend/src/store/authStore.ts` | 인증 상태 |
| `frontend/src/store/stockStore.ts` | 선택 종목, 관심종목 |
| `frontend/src/store/uiStore.ts` | darkMode, activeTab, sidebarOpen |
| `frontend/src/components/Layout/MainLayout.tsx` | 전체 레이아웃 쉘 |
| `frontend/src/components/Layout/Header.tsx` | 상단 헤더 |
| `frontend/src/components/Layout/MobileTabBar.tsx` | 모바일 상단 스크롤 탭 |
| `frontend/src/components/Sidebar/Sidebar.tsx` | 좌측 사이드바 컨테이너 |
| `frontend/src/components/Sidebar/StockGroup.tsx` | 종목 그룹 |
| `frontend/src/components/Sidebar/StockList.tsx` | 종목 리스트 |
| `frontend/src/components/MainPanel/MainPanel.tsx` | 탭 라우팅 |
| `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` | Mock 캔들스틱 차트 |
| `frontend/src/components/WatchlistPanel/WatchlistPanel.tsx` | 하단 고정 워치리스트 |
| `frontend/src/components/auth/LoginModal.tsx` | 로그인 모달 |
| `frontend/src/components/auth/RegisterModal.tsx` | 회원가입 모달 |
| `frontend/src/pages/LandingPage.tsx` | 랜딩 페이지 |
| `frontend/src/main.tsx` | 앱 진입점 |
| `frontend/src/App.tsx` | 루트 컴포넌트 |

---

## Task 1: Vite 프로젝트 생성 + 의존성 설치

**Files:**
- Create: `frontend/` (Vite scaffold)
- Create: `frontend/package.json`

- [ ] **Step 1: Vite 프로젝트 생성**

```bash
cd ~/FinalProject
npm create vite@latest frontend -- --template react-ts
```

- [ ] **Step 2: 프로덕션 의존성 설치**

```bash
cd ~/FinalProject/frontend
npm install zustand axios lightweight-charts recharts lucide-react
```

- [ ] **Step 3: 개발/테스트 의존성 설치**

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom @vitejs/plugin-react
```

- [ ] **Step 4: vite.config.ts에 테스트 설정 추가**

`frontend/vite.config.ts`를 다음으로 교체:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
})
```

- [ ] **Step 5: 테스트 setup 파일 생성**

`frontend/src/test/setup.ts`:

```typescript
import '@testing-library/jest-dom'
```

- [ ] **Step 6: package.json scripts에 test 추가**

`frontend/package.json`의 `"scripts"` 안에 추가:

```json
"test": "vitest",
"test:run": "vitest run"
```

- [ ] **Step 7: 기본 동작 확인**

```bash
cd ~/FinalProject/frontend
npm run dev
```

Expected: `http://localhost:5173` 에서 Vite 기본 페이지 뜨면 성공.
`Ctrl+C`로 종료.

- [ ] **Step 8: 커밋**

```bash
cd ~/FinalProject
git add frontend/
git commit -m "feat: scaffold Vite React TS project"
git push origin seogu-Jeong
```

---

## Task 2: Tailwind CSS v4 + shadcn/ui 설정

**Files:**
- Modify: `frontend/vite.config.ts`
- Create: `frontend/src/index.css`
- Create: `frontend/components.json`

- [ ] **Step 1: Tailwind CSS v4 설치**

```bash
cd ~/FinalProject/frontend
npm install tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: vite.config.ts에 Tailwind 플러그인 추가**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
})
```

- [ ] **Step 3: src/index.css 교체**

```css
@import "tailwindcss";

:root {
  color-scheme: dark;
}
```

- [ ] **Step 4: shadcn/ui 초기화**

```bash
cd ~/FinalProject/frontend
npx shadcn@latest init
```

프롬프트 답변:
- Style: `Default`
- Base color: `Slate`
- CSS variables: `Yes`

- [ ] **Step 5: 기본 컴포넌트 설치**

```bash
npx shadcn@latest add button input dialog tabs scroll-area
```

- [ ] **Step 6: Tailwind 동작 확인**

`frontend/src/App.tsx`를 다음으로 교체:

```tsx
export default function App() {
  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
      <h1 className="text-2xl font-bold text-primary">StockSenseAI</h1>
    </div>
  )
}
```

```bash
npm run dev
```

Expected: 어두운 배경에 "StockSenseAI" 텍스트가 보이면 성공.

- [ ] **Step 7: 커밋**

```bash
cd ~/FinalProject
git add frontend/
git commit -m "feat: add Tailwind CSS v4 and shadcn/ui"
git push origin seogu-Jeong
```

---

## Task 3: 공통 타입 정의 (types/index.ts)

**Files:**
- Create: `frontend/src/types/index.ts`

- [ ] **Step 1: types/index.ts 생성**

```typescript
// frontend/src/types/index.ts
// 공동 관리 파일 — 수정 전 반드시 hygrenn과 협의 (CLAUDE.md §5)

export interface User {
  id: string
  email: string
  mode: 'demo' | 'paper' | 'real'
  access_allowed: boolean
  is_verified: boolean
  dark_mode: boolean
}

export interface AISignal {
  signal: 'BUY' | 'HOLD' | 'SELL'
  signal_score: number
  tech_score: number
  lstm_score: number
  confidence: number
  indicators: {
    rsi_14: number
    macd: number
    macd_signal: number
    macd_hist: number
    bb_upper: number
    bb_middle: number
    bb_lower: number
    ma5: number
    ma20: number
    ma60: number
    ma120: number
  }
}

export interface Prediction {
  bullish: number[]
  base: number[]
  bearish: number[]
  confidence: number
}

export interface Candle {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Holding {
  stock_code: string
  stock_name: string
  quantity: number
  avg_price: number
  current_price: number
  profit_loss: number
  return_pct: number
  ai_signal: 'BUY' | 'HOLD' | 'SELL'
}

export interface OrderRequest {
  stock_code: string
  order_type: 'BUY' | 'SELL'
  price_type: 'MARKET' | 'LIMIT'
  quantity: number
  price?: number
  mode: 'paper' | 'real'
}

export interface LumpsumRequest {
  tickers: string[]
  buy_date: string
  sell_date: string
  amount_krw: number
}

export interface LumpsumResult {
  ticker: string
  name: string
  shares: number
  buy_price: number
  sell_price: number
  buy_value_krw: number
  sell_value_krw: number
  profit_krw: number
  return_pct: number
  chart_data: { date: string; return_pct: number }[]
}

export type TabId = 'chart' | 'ai' | 'simulator' | 'portfolio' | 'screener' | 'backtest'

export interface Stock {
  code: string
  name: string
  price?: number
  change_pct?: number
}
```

- [ ] **Step 2: 타입 임포트 테스트**

`frontend/src/test/types.test.ts`:

```typescript
import type { User, Candle, TabId } from '@/types'

describe('types', () => {
  it('User type has required fields', () => {
    const user: User = {
      id: '1',
      email: 'test@test.com',
      mode: 'demo',
      access_allowed: true,
      is_verified: true,
      dark_mode: true,
    }
    expect(user.mode).toBe('demo')
  })

  it('Candle type has OHLCV fields', () => {
    const candle: Candle = {
      time: '2026-06-01',
      open: 73000,
      high: 74000,
      low: 72500,
      close: 73400,
      volume: 1000000,
    }
    expect(candle.close).toBe(73400)
  })

  it('TabId covers all tabs', () => {
    const tabs: TabId[] = ['chart', 'ai', 'simulator', 'portfolio', 'screener', 'backtest']
    expect(tabs).toHaveLength(6)
  })
})
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `3 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/types/ frontend/src/test/types.test.ts
git commit -m "feat: add shared TypeScript types"
git push origin seogu-Jeong
```

---

## Task 4: lib/utils.ts + lib/mockData.ts

**Files:**
- Create: `frontend/src/lib/utils.ts`
- Create: `frontend/src/lib/mockData.ts`

- [ ] **Step 1: utils.ts 생성**

```typescript
// frontend/src/lib/utils.ts
import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

```bash
cd ~/FinalProject/frontend
npm install clsx tailwind-merge
```

- [ ] **Step 2: mockData.ts 생성**

```typescript
// frontend/src/lib/mockData.ts
import type { Candle, Stock } from '@/types'

export const MOCK_STOCKS: Stock[] = [
  { code: '005930', name: '삼성전자', price: 73400, change_pct: 1.2 },
  { code: '000660', name: 'SK하이닉스', price: 185000, change_pct: -0.5 },
  { code: '035420', name: 'NAVER', price: 210000, change_pct: 0.8 },
  { code: '035720', name: '카카오', price: 42100, change_pct: -1.3 },
  { code: '051910', name: 'LG화학', price: 320000, change_pct: 2.1 },
]

export const MOCK_WATCHLIST = ['005930', '000660', '035420']

function generateCandles(basePrice: number, count: number): Candle[] {
  const candles: Candle[] = []
  let price = basePrice
  const start = new Date('2026-01-02')

  for (let i = 0; i < count; i++) {
    const date = new Date(start)
    date.setDate(start.getDate() + i)
    if (date.getDay() === 0 || date.getDay() === 6) continue

    const change = (Math.random() - 0.48) * price * 0.03
    const open = price
    const close = Math.round(price + change)
    const high = Math.round(Math.max(open, close) * (1 + Math.random() * 0.01))
    const low = Math.round(Math.min(open, close) * (1 - Math.random() * 0.01))
    price = close

    candles.push({
      time: date.toISOString().split('T')[0],
      open,
      high,
      low,
      close,
      volume: Math.round(Math.random() * 2000000 + 500000),
    })
  }
  return candles
}

export const MOCK_CANDLES: Candle[] = generateCandles(73000, 120)
```

- [ ] **Step 3: mockData 테스트**

`frontend/src/test/mockData.test.ts`:

```typescript
import { MOCK_STOCKS, MOCK_CANDLES, MOCK_WATCHLIST } from '@/lib/mockData'

describe('mockData', () => {
  it('MOCK_STOCKS has at least 5 items', () => {
    expect(MOCK_STOCKS.length).toBeGreaterThanOrEqual(5)
  })

  it('MOCK_CANDLES are in ascending date order', () => {
    for (let i = 1; i < MOCK_CANDLES.length; i++) {
      expect(MOCK_CANDLES[i].time > MOCK_CANDLES[i - 1].time).toBe(true)
    }
  })

  it('MOCK_WATCHLIST contains valid stock codes', () => {
    const codes = MOCK_STOCKS.map((s) => s.code)
    MOCK_WATCHLIST.forEach((code) => {
      expect(codes).toContain(code)
    })
  })
})
```

- [ ] **Step 4: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `6 passed` (types 3 + mockData 3)

- [ ] **Step 5: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/lib/
git commit -m "feat: add utils and mock data"
git push origin seogu-Jeong
```

---

## Task 5: Axios 인스턴스 + JWT Interceptor (lib/api.ts)

**Files:**
- Create: `frontend/src/lib/api.ts`

- [ ] **Step 1: .env 파일 생성**

`frontend/.env`:

```
VITE_API_BASE=http://localhost:8000
```

`frontend/.env.example`:

```
VITE_API_BASE=http://localhost:8000
```

- [ ] **Step 2: api.ts 생성**

```typescript
// frontend/src/lib/api.ts
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE,
  withCredentials: true,
})

api.interceptors.request.use((config) => {
  const token = useAuthTokenRef.getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true
      try {
        const { data } = await axios.post(
          `${import.meta.env.VITE_API_BASE}/auth/refresh`,
          {},
          { withCredentials: true }
        )
        useAuthTokenRef.setToken(data.access_token)
        original.headers.Authorization = `Bearer ${data.access_token}`
        return api(original)
      } catch {
        useAuthTokenRef.clearToken()
        window.location.href = '/'
      }
    }
    return Promise.reject(error)
  }
)

// Axios interceptor는 Zustand store를 직접 import하면 순환 의존성이 생긴다.
// ref 패턴으로 토큰을 주입한다.
export const useAuthTokenRef = {
  _token: null as string | null,
  getToken: () => useAuthTokenRef._token,
  setToken: (t: string) => { useAuthTokenRef._token = t },
  clearToken: () => { useAuthTokenRef._token = null },
}

export default api
```

- [ ] **Step 3: api.ts 테스트**

`frontend/src/test/api.test.ts`:

```typescript
import { useAuthTokenRef } from '@/lib/api'

describe('useAuthTokenRef', () => {
  afterEach(() => useAuthTokenRef.clearToken())

  it('getToken returns null by default', () => {
    expect(useAuthTokenRef.getToken()).toBeNull()
  })

  it('setToken stores token', () => {
    useAuthTokenRef.setToken('abc123')
    expect(useAuthTokenRef.getToken()).toBe('abc123')
  })

  it('clearToken resets to null', () => {
    useAuthTokenRef.setToken('abc123')
    useAuthTokenRef.clearToken()
    expect(useAuthTokenRef.getToken()).toBeNull()
  })
})
```

- [ ] **Step 4: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `9 passed`

- [ ] **Step 5: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/lib/api.ts frontend/src/test/api.test.ts frontend/.env.example
git commit -m "feat: add Axios instance with JWT interceptor"
git push origin seogu-Jeong
```

---

## Task 6: Zustand 스토어 3개

**Files:**
- Create: `frontend/src/store/authStore.ts`
- Create: `frontend/src/store/stockStore.ts`
- Create: `frontend/src/store/uiStore.ts`

- [ ] **Step 1: authStore.ts 생성**

```typescript
// frontend/src/store/authStore.ts
import { create } from 'zustand'
import type { User } from '@/types'
import api from '@/lib/api'
import { useAuthTokenRef } from '@/lib/api'

interface AuthState {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  setUser: (user: User) => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isLoading: false,

  login: async (email, password) => {
    set({ isLoading: true })
    try {
      const { data } = await api.post('/auth/login', { email, password })
      useAuthTokenRef.setToken(data.access_token)
      set({ user: data.user, isLoading: false })
    } catch (e) {
      set({ isLoading: false })
      throw e
    }
  },

  logout: async () => {
    try {
      await api.post('/auth/logout')
    } finally {
      useAuthTokenRef.clearToken()
      set({ user: null })
    }
  },

  setUser: (user) => set({ user }),
}))
```

- [ ] **Step 2: stockStore.ts 생성**

```typescript
// frontend/src/store/stockStore.ts
import { create } from 'zustand'
import type { Stock } from '@/types'
import { MOCK_STOCKS, MOCK_WATCHLIST } from '@/lib/mockData'

interface StockState {
  selectedStock: Stock | null
  watchlist: string[]
  stockList: Stock[]
  setSelectedStock: (stock: Stock) => void
  addToWatchlist: (code: string) => void
  removeFromWatchlist: (code: string) => void
}

export const useStockStore = create<StockState>((set) => ({
  selectedStock: MOCK_STOCKS[0],
  watchlist: MOCK_WATCHLIST,
  stockList: MOCK_STOCKS,

  setSelectedStock: (stock) => set({ selectedStock: stock }),

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
}))
```

- [ ] **Step 3: uiStore.ts 생성**

```typescript
// frontend/src/store/uiStore.ts
import { create } from 'zustand'
import type { TabId } from '@/types'

interface UIState {
  darkMode: boolean
  activeTab: TabId
  sidebarOpen: boolean
  toggleDarkMode: () => void
  setActiveTab: (tab: TabId) => void
  toggleSidebar: () => void
}

export const useUIStore = create<UIState>((set) => ({
  darkMode: true,
  activeTab: 'chart',
  sidebarOpen: true,

  toggleDarkMode: () =>
    set((state) => {
      const next = !state.darkMode
      document.documentElement.classList.toggle('dark', next)
      return { darkMode: next }
    }),

  setActiveTab: (tab) => set({ activeTab: tab }),

  toggleSidebar: () =>
    set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}))
```

- [ ] **Step 4: 스토어 테스트**

`frontend/src/test/stores.test.ts`:

```typescript
import { act, renderHook } from '@testing-library/react'
import { useStockStore } from '@/store/stockStore'
import { useUIStore } from '@/store/uiStore'

describe('stockStore', () => {
  it('initial selectedStock is first mock stock', () => {
    const { result } = renderHook(() => useStockStore())
    expect(result.current.selectedStock?.code).toBe('005930')
  })

  it('addToWatchlist adds unique code', () => {
    const { result } = renderHook(() => useStockStore())
    const initialLen = result.current.watchlist.length
    act(() => result.current.addToWatchlist('999999'))
    expect(result.current.watchlist).toContain('999999')
    expect(result.current.watchlist.length).toBe(initialLen + 1)
  })

  it('addToWatchlist does not duplicate', () => {
    const { result } = renderHook(() => useStockStore())
    const existingCode = result.current.watchlist[0]
    const before = result.current.watchlist.length
    act(() => result.current.addToWatchlist(existingCode))
    expect(result.current.watchlist.length).toBe(before)
  })

  it('removeFromWatchlist removes code', () => {
    const { result } = renderHook(() => useStockStore())
    const code = result.current.watchlist[0]
    act(() => result.current.removeFromWatchlist(code))
    expect(result.current.watchlist).not.toContain(code)
  })
})

describe('uiStore', () => {
  it('default darkMode is true', () => {
    const { result } = renderHook(() => useUIStore())
    expect(result.current.darkMode).toBe(true)
  })

  it('toggleDarkMode flips darkMode', () => {
    const { result } = renderHook(() => useUIStore())
    act(() => result.current.toggleDarkMode())
    expect(result.current.darkMode).toBe(false)
    act(() => result.current.toggleDarkMode())
    expect(result.current.darkMode).toBe(true)
  })

  it('setActiveTab updates activeTab', () => {
    const { result } = renderHook(() => useUIStore())
    act(() => result.current.setActiveTab('ai'))
    expect(result.current.activeTab).toBe('ai')
  })
})
```

- [ ] **Step 5: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `16 passed`

- [ ] **Step 6: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/store/ frontend/src/test/stores.test.ts
git commit -m "feat: add Zustand stores (auth, stock, ui)"
git push origin seogu-Jeong
```

---

## Task 7: Header 컴포넌트

**Files:**
- Create: `frontend/src/components/Layout/Header.tsx`

- [ ] **Step 1: Header.tsx 생성**

```tsx
// frontend/src/components/Layout/Header.tsx
import { Search, Sun, Moon, User } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useUIStore } from '@/store/uiStore'
import { useAuthStore } from '@/store/authStore'
import { useStockStore } from '@/store/stockStore'
import { MOCK_STOCKS } from '@/lib/mockData'
import { useState } from 'react'

export function Header({ onLoginClick }: { onLoginClick: () => void }) {
  const { darkMode, toggleDarkMode } = useUIStore()
  const { user } = useAuthStore()
  const { setSelectedStock } = useStockStore()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(MOCK_STOCKS.slice(0, 0))

  const handleSearch = (q: string) => {
    setQuery(q)
    setResults(
      q.length > 0
        ? MOCK_STOCKS.filter(
            (s) => s.name.includes(q) || s.code.includes(q)
          ).slice(0, 5)
        : []
    )
  }

  return (
    <header className="flex items-center justify-between px-4 h-12 bg-card border-b border-border shrink-0">
      <span className="text-primary font-bold text-lg tracking-tight">StockSenseAI</span>

      <div className="relative w-64">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          className="pl-8 h-8 text-sm"
          placeholder="종목명 또는 코드"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onBlur={() => setTimeout(() => setResults([]), 200)}
        />
        {results.length > 0 && (
          <ul className="absolute top-9 left-0 w-full bg-popover border border-border rounded-md shadow-lg z-50">
            {results.map((s) => (
              <li
                key={s.code}
                className="px-3 py-2 text-sm cursor-pointer hover:bg-accent"
                onMouseDown={() => {
                  setSelectedStock(s)
                  setQuery('')
                  setResults([])
                }}
              >
                <span className="font-medium">{s.name}</span>
                <span className="ml-2 text-muted-foreground text-xs">{s.code}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={toggleDarkMode} aria-label="테마 전환">
          {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
        {user ? (
          <span className="text-sm text-muted-foreground">{user.email}</span>
        ) : (
          <Button variant="outline" size="sm" onClick={onLoginClick}>
            <User className="h-4 w-4 mr-1" /> 로그인
          </Button>
        )}
      </div>
    </header>
  )
}
```

- [ ] **Step 2: Header 테스트**

`frontend/src/test/Header.test.tsx`:

```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { Header } from '@/components/Layout/Header'

describe('Header', () => {
  it('renders logo', () => {
    render(<Header onLoginClick={() => {}} />)
    expect(screen.getByText('StockSenseAI')).toBeInTheDocument()
  })

  it('renders search input', () => {
    render(<Header onLoginClick={() => {}} />)
    expect(screen.getByPlaceholderText('종목명 또는 코드')).toBeInTheDocument()
  })

  it('calls onLoginClick when login button clicked', () => {
    const fn = vi.fn()
    render(<Header onLoginClick={fn} />)
    fireEvent.click(screen.getByText('로그인'))
    expect(fn).toHaveBeenCalledTimes(1)
  })

  it('shows search results when typing', async () => {
    render(<Header onLoginClick={() => {}} />)
    const input = screen.getByPlaceholderText('종목명 또는 코드')
    fireEvent.change(input, { target: { value: '삼성' } })
    expect(await screen.findByText('삼성전자')).toBeInTheDocument()
  })
})
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `20 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/Layout/Header.tsx frontend/src/test/Header.test.tsx
git commit -m "feat: add Header component with search and dark mode toggle"
git push origin seogu-Jeong
```

---

## Task 8: Sidebar (StockGroup + StockList)

**Files:**
- Create: `frontend/src/components/Sidebar/StockGroup.tsx`
- Create: `frontend/src/components/Sidebar/StockList.tsx`
- Create: `frontend/src/components/Sidebar/Sidebar.tsx`

- [ ] **Step 1: StockGroup.tsx 생성**

```tsx
// frontend/src/components/Sidebar/StockGroup.tsx
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'

interface StockGroupProps {
  name: string
  children: React.ReactNode
}

export function StockGroup({ name, children }: StockGroupProps) {
  const [open, setOpen] = useState(true)
  return (
    <div>
      <button
        className="flex items-center gap-1 w-full px-3 py-1.5 text-xs font-semibold text-muted-foreground hover:text-foreground uppercase tracking-wider"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {name}
      </button>
      {open && <div>{children}</div>}
    </div>
  )
}
```

- [ ] **Step 2: StockList.tsx 생성**

```tsx
// frontend/src/components/Sidebar/StockList.tsx
import type { Stock } from '@/types'
import { useStockStore } from '@/store/stockStore'
import { cn } from '@/lib/utils'

interface StockListProps {
  stocks: Stock[]
}

export function StockList({ stocks }: StockListProps) {
  const { selectedStock, setSelectedStock } = useStockStore()

  return (
    <ul>
      {stocks.map((stock) => (
        <li
          key={stock.code}
          onClick={() => setSelectedStock(stock)}
          className={cn(
            'flex items-center justify-between px-4 py-1.5 cursor-pointer text-sm hover:bg-accent',
            selectedStock?.code === stock.code && 'bg-accent text-accent-foreground'
          )}
        >
          <span className="truncate">{stock.name}</span>
          {stock.price && (
            <div className="text-right ml-2 shrink-0">
              <div className="text-xs font-medium">{stock.price.toLocaleString()}</div>
              {stock.change_pct !== undefined && (
                <div
                  className={cn(
                    'text-xs',
                    stock.change_pct >= 0 ? 'text-green-500' : 'text-red-500'
                  )}
                >
                  {stock.change_pct >= 0 ? '+' : ''}{stock.change_pct.toFixed(1)}%
                </div>
              )}
            </div>
          )}
        </li>
      ))}
    </ul>
  )
}
```

- [ ] **Step 3: Sidebar.tsx 생성**

```tsx
// frontend/src/components/Sidebar/Sidebar.tsx
import { useStockStore } from '@/store/stockStore'
import { StockGroup } from './StockGroup'
import { StockList } from './StockList'

export function Sidebar() {
  const { stockList, watchlist } = useStockStore()
  const watchlistStocks = stockList.filter((s) => watchlist.includes(s.code))

  return (
    <aside className="w-52 shrink-0 bg-card border-r border-border overflow-y-auto">
      <StockGroup name="관심종목">
        <StockList stocks={watchlistStocks} />
      </StockGroup>
      <StockGroup name="전체종목">
        <StockList stocks={stockList} />
      </StockGroup>
    </aside>
  )
}
```

- [ ] **Step 4: Sidebar 테스트**

`frontend/src/test/Sidebar.test.tsx`:

```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { Sidebar } from '@/components/Sidebar/Sidebar'

describe('Sidebar', () => {
  it('renders 관심종목 group', () => {
    render(<Sidebar />)
    expect(screen.getByText('관심종목')).toBeInTheDocument()
  })

  it('renders 전체종목 group', () => {
    render(<Sidebar />)
    expect(screen.getByText('전체종목')).toBeInTheDocument()
  })

  it('collapses group on click', () => {
    render(<Sidebar />)
    const btn = screen.getByText('관심종목')
    fireEvent.click(btn)
    expect(screen.queryAllByText('삼성전자').length).toBeLessThan(2)
  })
})
```

- [ ] **Step 5: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `23 passed`

- [ ] **Step 6: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/Sidebar/ frontend/src/test/Sidebar.test.tsx
git commit -m "feat: add Sidebar with StockGroup and StockList"
git push origin seogu-Jeong
```

---

## Task 9: WatchlistPanel + MobileTabBar

**Files:**
- Create: `frontend/src/components/WatchlistPanel/WatchlistPanel.tsx`
- Create: `frontend/src/components/Layout/MobileTabBar.tsx`

- [ ] **Step 1: WatchlistPanel.tsx 생성**

```tsx
// frontend/src/components/WatchlistPanel/WatchlistPanel.tsx
import { useStockStore } from '@/store/stockStore'
import { cn } from '@/lib/utils'

export function WatchlistPanel() {
  const { stockList, watchlist, selectedStock, setSelectedStock } = useStockStore()
  const stocks = stockList.filter((s) => watchlist.includes(s.code))

  return (
    <div className="h-8 flex items-center gap-4 px-4 bg-card border-t border-border overflow-x-auto shrink-0">
      {stocks.map((stock) => (
        <button
          key={stock.code}
          onClick={() => setSelectedStock(stock)}
          className={cn(
            'flex items-center gap-2 text-xs whitespace-nowrap hover:text-foreground',
            selectedStock?.code === stock.code ? 'text-foreground' : 'text-muted-foreground'
          )}
        >
          <span className="font-medium">{stock.name}</span>
          {stock.price && (
            <span>{stock.price.toLocaleString()}</span>
          )}
          {stock.change_pct !== undefined && (
            <span className={stock.change_pct >= 0 ? 'text-green-500' : 'text-red-500'}>
              {stock.change_pct >= 0 ? '▲' : '▼'}{Math.abs(stock.change_pct).toFixed(1)}%
            </span>
          )}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: MobileTabBar.tsx 생성**

```tsx
// frontend/src/components/Layout/MobileTabBar.tsx
import { useUIStore } from '@/store/uiStore'
import type { TabId } from '@/types'
import { cn } from '@/lib/utils'
import { ScrollArea } from '@/components/ui/scroll-area'

const TABS: { id: TabId; label: string }[] = [
  { id: 'chart', label: '차트' },
  { id: 'ai', label: 'AI' },
  { id: 'simulator', label: '시뮬' },
  { id: 'portfolio', label: '포트폴리오' },
  { id: 'screener', label: '스크리너' },
  { id: 'backtest', label: '백테스트' },
]

export function MobileTabBar() {
  const { activeTab, setActiveTab } = useUIStore()

  return (
    <ScrollArea className="md:hidden border-b border-border bg-card" orientation="horizontal">
      <div className="flex h-9">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'px-4 h-full text-sm whitespace-nowrap shrink-0 border-b-2 transition-colors',
              activeTab === tab.id
                ? 'border-primary text-foreground font-medium'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </ScrollArea>
  )
}
```

- [ ] **Step 3: 테스트**

`frontend/src/test/WatchlistPanel.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { WatchlistPanel } from '@/components/WatchlistPanel/WatchlistPanel'

describe('WatchlistPanel', () => {
  it('renders watchlist stocks', () => {
    render(<WatchlistPanel />)
    expect(screen.getByText('삼성전자')).toBeInTheDocument()
  })
})
```

`frontend/src/test/MobileTabBar.test.tsx`:

```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { MobileTabBar } from '@/components/Layout/MobileTabBar'
import { useUIStore } from '@/store/uiStore'

describe('MobileTabBar', () => {
  it('renders all 6 tabs', () => {
    render(<MobileTabBar />)
    expect(screen.getByText('차트')).toBeInTheDocument()
    expect(screen.getByText('AI')).toBeInTheDocument()
    expect(screen.getByText('포트폴리오')).toBeInTheDocument()
  })

  it('clicking tab updates activeTab store', () => {
    render(<MobileTabBar />)
    fireEvent.click(screen.getByText('AI'))
    expect(useUIStore.getState().activeTab).toBe('ai')
  })
})
```

- [ ] **Step 4: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `27 passed`

- [ ] **Step 5: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/WatchlistPanel/ frontend/src/components/Layout/MobileTabBar.tsx frontend/src/test/
git commit -m "feat: add WatchlistPanel and MobileTabBar"
git push origin seogu-Jeong
```

---

## Task 10: 인증 모달 (LoginModal + RegisterModal)

**Files:**
- Create: `frontend/src/components/auth/LoginModal.tsx`
- Create: `frontend/src/components/auth/RegisterModal.tsx`

- [ ] **Step 1: LoginModal.tsx 생성**

```tsx
// frontend/src/components/auth/LoginModal.tsx
import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useAuthStore } from '@/store/authStore'

interface LoginModalProps {
  open: boolean
  onClose: () => void
  onRegister: () => void
}

export function LoginModal({ open, onClose, onRegister }: LoginModalProps) {
  const { login, isLoading } = useAuthStore()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      await login(email, password)
      onClose()
    } catch {
      setError('이메일 또는 비밀번호가 올바르지 않습니다.')
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>로그인</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-3">
          <Input
            type="email"
            placeholder="이메일"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <Input
            type="password"
            placeholder="비밀번호"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          {error && <p className="text-sm text-red-500">{error}</p>}
          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? '로그인 중...' : '로그인'}
          </Button>
          <Button type="button" variant="ghost" className="w-full" onClick={onRegister}>
            회원가입
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  )
}
```

- [ ] **Step 2: RegisterModal.tsx 생성**

```tsx
// frontend/src/components/auth/RegisterModal.tsx
import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import api from '@/lib/api'

interface RegisterModalProps {
  open: boolean
  onClose: () => void
  onLogin: () => void
}

export function RegisterModal({ open, onClose, onLogin }: RegisterModalProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [done, setDone] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    if (password !== confirm) {
      setError('비밀번호가 일치하지 않습니다.')
      return
    }
    setLoading(true)
    try {
      await api.post('/auth/register', { email, password })
      setDone(true)
    } catch {
      setError('회원가입에 실패했습니다. 다시 시도해주세요.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>회원가입</DialogTitle>
        </DialogHeader>
        {done ? (
          <div className="space-y-3 text-center">
            <p className="text-sm text-muted-foreground">
              이메일 인증 링크를 발송했습니다. 확인 후 로그인해주세요.
            </p>
            <Button className="w-full" onClick={onLogin}>로그인으로 이동</Button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-3">
            <Input
              type="email"
              placeholder="이메일"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <Input
              type="password"
              placeholder="비밀번호"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <Input
              type="password"
              placeholder="비밀번호 확인"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              required
            />
            {error && <p className="text-sm text-red-500">{error}</p>}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? '가입 중...' : '회원가입'}
            </Button>
            <Button type="button" variant="ghost" className="w-full" onClick={onLogin}>
              이미 계정이 있어요
            </Button>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}
```

- [ ] **Step 3: 인증 모달 테스트**

`frontend/src/test/AuthModals.test.tsx`:

```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { LoginModal } from '@/components/auth/LoginModal'
import { RegisterModal } from '@/components/auth/RegisterModal'

describe('LoginModal', () => {
  it('renders when open', () => {
    render(<LoginModal open={true} onClose={() => {}} onRegister={() => {}} />)
    expect(screen.getByText('로그인')).toBeInTheDocument()
  })

  it('does not render when closed', () => {
    render(<LoginModal open={false} onClose={() => {}} onRegister={() => {}} />)
    expect(screen.queryByText('로그인')).not.toBeInTheDocument()
  })

  it('calls onRegister when 회원가입 clicked', () => {
    const fn = vi.fn()
    render(<LoginModal open={true} onClose={() => {}} onRegister={fn} />)
    fireEvent.click(screen.getByText('회원가입'))
    expect(fn).toHaveBeenCalled()
  })
})

describe('RegisterModal', () => {
  it('shows password mismatch error', async () => {
    render(<RegisterModal open={true} onClose={() => {}} onLogin={() => {}} />)
    fireEvent.change(screen.getByPlaceholderText('비밀번호'), { target: { value: 'abc' } })
    fireEvent.change(screen.getByPlaceholderText('비밀번호 확인'), { target: { value: 'xyz' } })
    fireEvent.submit(screen.getByRole('button', { name: '회원가입' }))
    expect(await screen.findByText('비밀번호가 일치하지 않습니다.')).toBeInTheDocument()
  })
})
```

- [ ] **Step 4: 테스트 실행**

```bash
cd ~/FinalProject/frontend
npm run test:run
```

Expected: `31 passed`

- [ ] **Step 5: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/auth/ frontend/src/test/AuthModals.test.tsx
git commit -m "feat: add LoginModal and RegisterModal"
git push origin seogu-Jeong
```

---

## Task 11: LandingPage

**Files:**
- Create: `frontend/src/pages/LandingPage.tsx`

- [ ] **Step 1: LandingPage.tsx 생성**

```tsx
// frontend/src/pages/LandingPage.tsx
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { LoginModal } from '@/components/auth/LoginModal'
import { RegisterModal } from '@/components/auth/RegisterModal'

interface LandingPageProps {
  onEnter: () => void
}

export function LandingPage({ onEnter }: LandingPageProps) {
  const [modal, setModal] = useState<'login' | 'register' | null>(null)

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-6 text-center px-4">
      <h1 className="text-4xl font-bold tracking-tight text-foreground">
        Stock<span className="text-primary">Sense</span>AI
      </h1>
      <p className="text-muted-foreground max-w-md text-sm">
        AI 기반 한국 주식 차트 예측 + 실제 거래 실행을 통합한 웹 서비스
      </p>
      <div className="flex gap-3">
        <Button onClick={() => setModal('login')}>로그인</Button>
        <Button variant="outline" onClick={onEnter}>
          둘러보기 (데모)
        </Button>
      </div>

      <LoginModal
        open={modal === 'login'}
        onClose={() => setModal(null)}
        onRegister={() => setModal('register')}
      />
      <RegisterModal
        open={modal === 'register'}
        onClose={() => setModal(null)}
        onLogin={() => setModal('login')}
      />
    </div>
  )
}
```

- [ ] **Step 2: LandingPage 테스트**

`frontend/src/test/LandingPage.test.tsx`:

```tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { LandingPage } from '@/pages/LandingPage'

describe('LandingPage', () => {
  it('renders title', () => {
    render(<LandingPage onEnter={() => {}} />)
    expect(screen.getByText('SenseAI')).toBeInTheDocument()
  })

  it('calls onEnter when 둘러보기 clicked', () => {
    const fn = vi.fn()
    render(<LandingPage onEnter={fn} />)
    fireEvent.click(screen.getByText('둘러보기 (데모)'))
    expect(fn).toHaveBeenCalled()
  })

  it('opens login modal on 로그인 click', () => {
    render(<LandingPage onEnter={() => {}} />)
    fireEvent.click(screen.getByText('로그인'))
    expect(screen.getAllByText('로그인').length).toBeGreaterThan(1)
  })
})
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
git add frontend/src/pages/LandingPage.tsx frontend/src/test/LandingPage.test.tsx
git commit -m "feat: add LandingPage"
git push origin seogu-Jeong
```

---

## Task 12: ChartTab (Mock 캔들스틱)

**Files:**
- Create: `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx`

- [ ] **Step 1: ChartTab.tsx 생성**

```tsx
// frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'
import { useStockStore } from '@/store/stockStore'
import { MOCK_CANDLES } from '@/lib/mockData'

export function ChartTab() {
  const chartRef = useRef<HTMLDivElement>(null)
  const { selectedStock } = useStockStore()

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

    series.setData(MOCK_CANDLES)
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
  }, [selectedStock])

  return (
    <div className="flex flex-col h-full p-2 gap-2">
      <div className="flex items-center gap-3 px-2">
        <span className="font-semibold">{selectedStock?.name}</span>
        <span className="text-muted-foreground text-sm">{selectedStock?.code}</span>
        {selectedStock?.price && (
          <span className="font-bold">{selectedStock.price.toLocaleString()}원</span>
        )}
        {selectedStock?.change_pct !== undefined && (
          <span className={selectedStock.change_pct >= 0 ? 'text-green-500' : 'text-red-500'}>
            {selectedStock.change_pct >= 0 ? '+' : ''}{selectedStock.change_pct.toFixed(1)}%
          </span>
        )}
      </div>
      <div ref={chartRef} className="flex-1 min-h-0" />
    </div>
  )
}
```

- [ ] **Step 2: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
git commit -m "feat: add ChartTab with mock candlestick chart"
git push origin seogu-Jeong
```

---

## Task 13: MainPanel + MainLayout + App 조립

**Files:**
- Create: `frontend/src/components/MainPanel/MainPanel.tsx`
- Create: `frontend/src/components/Layout/MainLayout.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1: MainPanel.tsx 생성**

```tsx
// frontend/src/components/MainPanel/MainPanel.tsx
import { useUIStore } from '@/store/uiStore'
import { ChartTab } from './ChartTab/ChartTab'

const PLACEHOLDER_TABS = ['ai', 'simulator', 'portfolio', 'screener', 'backtest'] as const

function PlaceholderTab({ name }: { name: string }) {
  return (
    <div className="flex items-center justify-center h-full text-muted-foreground">
      {name} — Phase 2+ 에서 구현 예정
    </div>
  )
}

export function MainPanel() {
  const { activeTab } = useUIStore()

  return (
    <div className="flex-1 min-w-0 min-h-0 overflow-hidden">
      {activeTab === 'chart' && <ChartTab />}
      {PLACEHOLDER_TABS.map((tab) =>
        activeTab === tab ? <PlaceholderTab key={tab} name={tab} /> : null
      )}
    </div>
  )
}
```

- [ ] **Step 2: MainLayout.tsx 생성**

```tsx
// frontend/src/components/Layout/MainLayout.tsx
import { Header } from './Header'
import { MobileTabBar } from './MobileTabBar'
import { Sidebar } from '@/components/Sidebar/Sidebar'
import { MainPanel } from '@/components/MainPanel/MainPanel'
import { WatchlistPanel } from '@/components/WatchlistPanel/WatchlistPanel'
import { useUIStore } from '@/store/uiStore'
import { useState } from 'react'
import { LoginModal } from '@/components/auth/LoginModal'
import { RegisterModal } from '@/components/auth/RegisterModal'

export function MainLayout() {
  const { sidebarOpen } = useUIStore()
  const [modal, setModal] = useState<'login' | 'register' | null>(null)

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <Header onLoginClick={() => setModal('login')} />
      <MobileTabBar />

      <div className="flex flex-1 min-h-0">
        {sidebarOpen && (
          <div className="hidden md:block">
            <Sidebar />
          </div>
        )}
        <MainPanel />
        {/* 우측 패널 — Phase 4에서 호가창 + 주문창 구현 */}
        <div className="hidden lg:block w-56 shrink-0 bg-card border-l border-border" />
      </div>

      <div className="hidden md:block">
        <WatchlistPanel />
      </div>

      <LoginModal
        open={modal === 'login'}
        onClose={() => setModal(null)}
        onRegister={() => setModal('register')}
      />
      <RegisterModal
        open={modal === 'register'}
        onClose={() => setModal(null)}
        onLogin={() => setModal('login')}
      />
    </div>
  )
}
```

- [ ] **Step 3: App.tsx 교체**

```tsx
// frontend/src/App.tsx
import { useState, useEffect } from 'react'
import { LandingPage } from '@/pages/LandingPage'
import { MainLayout } from '@/components/Layout/MainLayout'
import { useUIStore } from '@/store/uiStore'
import { useAuthStore } from '@/store/authStore'

export default function App() {
  const [entered, setEntered] = useState(false)
  const { darkMode } = useUIStore()
  const { user } = useAuthStore()

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  if (!entered && !user) {
    return <LandingPage onEnter={() => setEntered(true)} />
  }

  return <MainLayout />
}
```

- [ ] **Step 4: main.tsx 교체**

```tsx
// frontend/src/main.tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
```

- [ ] **Step 5: 전체 동작 확인**

```bash
cd ~/FinalProject/frontend
npm run dev
```

확인 항목:
1. `http://localhost:5173` → LandingPage 렌더링
2. "둘러보기 (데모)" 클릭 → MainLayout 진입
3. 좌측 Sidebar에 종목 목록 표시
4. 중앙에 캔들스틱 차트 렌더링
5. 우측 빈 패널 표시
6. 하단 WatchlistPanel 표시
7. Header의 달/해 아이콘 클릭 → 다크/라이트 전환
8. 모바일 너비 (<768px) → MobileTabBar만 표시

- [ ] **Step 6: 전체 테스트 실행**

```bash
npm run test:run
```

Expected: `34 passed` (차트는 DOM 의존으로 테스트 제외)

- [ ] **Step 7: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/
git commit -m "feat: assemble MainLayout, MainPanel, and App — Phase 1 complete"
git push origin seogu-Jeong
```

---

## Task 14: .gitignore + CLAUDE.md Phase 1 완료 표시

**Files:**
- Modify: `frontend/.gitignore`
- Modify: `CLAUDE.md`

- [ ] **Step 1: frontend/.gitignore 확인 및 .env 추가**

`frontend/.gitignore`에 다음이 있는지 확인, 없으면 추가:

```
.env
.env.local
dist/
node_modules/
```

- [ ] **Step 2: CLAUDE.md §12 업데이트**

`CLAUDE.md`의 `## 12. 현재 진행 상태` 섹션을:

```markdown
- [x] Phase 1 — MVP (프론트엔드 완료 2026-06-01)
- [ ] Phase 2 — 실시간 시세 + 차트
- [ ] Phase 3 — AI 기능
- [ ] Phase 4 — 거래 + 포트폴리오
```

- [ ] **Step 3: 최종 커밋 + dev 머지**

```bash
cd ~/FinalProject
git add CLAUDE.md frontend/.gitignore
git commit -m "docs: mark Phase 1 frontend complete"
git push origin seogu-Jeong

# dev에 머지 (팀원에게 카톡 먼저!)
git checkout dev
git pull origin dev
git merge seogu-Jeong
git push origin dev
git checkout seogu-Jeong
```
