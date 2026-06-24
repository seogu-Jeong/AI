# StockSenseAI Frontend Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 3 프론트엔드 완성 — AI 예측 오버레이, 캔들 패턴 배지, AI 시그널 카드, 멀티 타임프레임 패널

**Architecture:** CandleChart에 `onChartReady` 콜백을 추가해 chart 인스턴스를 ChartTab으로 노출하고, PredictionOverlay가 동일 chart에 점선 시리즈를 추가한다. AITab은 SignalCard·ScoreBreakdown·MultiframePanel을 조립하는 새 탭으로, MainPanel에서 'ai' 탭 선택 시 렌더링된다.

**Tech Stack:** React 18, TypeScript 5, Lightweight Charts 4 (v5 API), shadcn/ui (Tooltip, Progress), Zustand, Vitest

---

## File Map

| 파일 | 역할 | 상태 |
|---|---|---|
| `frontend/src/types/index.ts` | CandlePattern, MultiframeSignal 타입 추가 | 수정 |
| `frontend/src/lib/mockData.ts` | AI Mock 데이터 4종 추가 | 수정 |
| `frontend/src/components/MainPanel/ChartTab/CandleChart.tsx` | onChartReady 콜백 추가 | 수정 |
| `frontend/src/components/MainPanel/ChartTab/PredictionOverlay.tsx` | Bullish/Base/Bearish 점선 | 신규 |
| `frontend/src/components/MainPanel/ChartTab/PatternBadges.tsx` | 캔들 패턴 배지 + 툴팁 | 신규 |
| `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` | PredictionOverlay + PatternBadges 통합 | 수정 |
| `frontend/src/components/MainPanel/AITab/SignalCard.tsx` | BUY/HOLD/SELL 시그널 카드 | 신규 |
| `frontend/src/components/MainPanel/AITab/ScoreBreakdown.tsx` | 점수 분해 progress bar | 신규 |
| `frontend/src/components/MainPanel/AITab/MultiframePanel.tsx` | 멀티 타임프레임 시그널 | 신규 |
| `frontend/src/components/MainPanel/AITab/AITab.tsx` | AI 탭 루트 조립 | 신규 |
| `frontend/src/components/MainPanel/MainPanel.tsx` | AITab 연결 | 수정 |

---

## Task 1: types/index.ts에 CandlePattern, MultiframeSignal 추가 + shadcn 컴포넌트 설치

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/test/types.test.ts`

- [ ] **Step 1: shadcn Tooltip, Progress 설치**

```bash
cd ~/FinalProject/frontend
npx shadcn@latest add tooltip progress
```

- [ ] **Step 2: types/index.ts 맨 끝에 추가**

```typescript
export interface CandlePattern {
  name: string
  type: 'bullish' | 'bearish' | 'neutral'
  description: string
}

export interface MultiframeSignal {
  timeframe: '1D' | '1W' | '1M'
  signal: 'BUY' | 'HOLD' | 'SELL'
  score: number
}
```

- [ ] **Step 3: types.test.ts에 테스트 추가**

`frontend/src/test/types.test.ts`의 import 줄 수정:
```typescript
import type { User, Candle, TabId, StockDetail, CandlePattern, MultiframeSignal } from '@/types'
```

`describe('types', ...)` 블록 안에 추가:
```typescript
it('CandlePattern has name, type, description', () => {
  const p: CandlePattern = { name: '망치형', type: 'bullish', description: '반전 신호' }
  expect(p.type).toBe('bullish')
})

it('MultiframeSignal has timeframe, signal, score', () => {
  const s: MultiframeSignal = { timeframe: '1D', signal: 'BUY', score: 78 }
  expect(s.signal).toBe('BUY')
})
```

- [ ] **Step 4: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `61 passed`

- [ ] **Step 5: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/types/index.ts frontend/src/test/types.test.ts frontend/src/components/ui/
git commit -m "feat: add CandlePattern, MultiframeSignal types and shadcn tooltip/progress"
git push origin seogu-Jeong
```

---

## Task 2: mockData.ts에 AI Mock 데이터 추가

**Files:**
- Modify: `frontend/src/lib/mockData.ts`
- Modify: `frontend/src/test/mockData.test.ts`

- [ ] **Step 1: mockData.ts import 수정 + 데이터 추가**

`frontend/src/lib/mockData.ts` 상단 import:
```typescript
import type { Candle, Stock, StockDetail, CandlePattern, MultiframeSignal } from '@/types'
```

파일 맨 끝에 추가:
```typescript
export const MOCK_AI_SIGNAL = {
  signal: 'BUY' as const,
  signal_score: 78,
  tech_score: 72,
  lstm_score: 84,
  confidence: 0.81,
  indicators: {
    rsi_14: 62.4,
    macd: 12.3,
    macd_signal: 8.1,
    macd_hist: 4.2,
    bb_upper: 76200,
    bb_middle: 73400,
    bb_lower: 70600,
    ma5: 73200,
    ma20: 72100,
    ma60: 70500,
    ma120: 68900,
  },
}

export const MOCK_PREDICTION = {
  bullish: [74200, 75100, 76300, 77500, 78200],
  base:    [73800, 74300, 74900, 75400, 75800],
  bearish: [73200, 72800, 72100, 71500, 70900],
  confidence: 0.81,
}

export const MOCK_PATTERNS: CandlePattern[] = [
  { name: '망치형', type: 'bullish', description: '하락 추세 후 반전 가능성을 나타내는 강세 패턴' },
  { name: '도지', type: 'neutral', description: '시장 불확실성을 나타내며 추세 전환 신호일 수 있음' },
  { name: '상승장악형', type: 'bullish', description: '전일 하락을 완전히 상쇄하는 강한 매수 신호' },
]

export const MOCK_MULTIFRAME: MultiframeSignal[] = [
  { timeframe: '1D', signal: 'BUY', score: 78 },
  { timeframe: '1W', signal: 'HOLD', score: 52 },
  { timeframe: '1M', signal: 'BUY', score: 65 },
]
```

- [ ] **Step 2: mockData.test.ts에 테스트 추가**

기존 import 줄 수정:
```typescript
import { MOCK_STOCKS, MOCK_CANDLES, MOCK_WATCHLIST, MOCK_STOCK_DETAILS, MOCK_AI_SIGNAL, MOCK_PATTERNS, MOCK_MULTIFRAME, MOCK_PREDICTION } from '@/lib/mockData'
```

`describe('mockData', ...)` 블록 안에 추가:
```typescript
it('MOCK_AI_SIGNAL has BUY/HOLD/SELL signal', () => {
  expect(['BUY', 'HOLD', 'SELL']).toContain(MOCK_AI_SIGNAL.signal)
})

it('MOCK_PREDICTION has 5 values per scenario', () => {
  expect(MOCK_PREDICTION.bullish).toHaveLength(5)
  expect(MOCK_PREDICTION.base).toHaveLength(5)
  expect(MOCK_PREDICTION.bearish).toHaveLength(5)
})

it('MOCK_MULTIFRAME covers all timeframes', () => {
  const frames = MOCK_MULTIFRAME.map((m) => m.timeframe)
  expect(frames).toContain('1D')
  expect(frames).toContain('1W')
  expect(frames).toContain('1M')
})
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `64 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add -f frontend/src/lib/mockData.ts frontend/src/test/mockData.test.ts
git commit -m "feat: add AI mock data (signal, prediction, patterns, multiframe)"
git push origin seogu-Jeong
```

---

## Task 3: CandleChart — onChartReady 콜백 추가

**Files:**
- Modify: `frontend/src/components/MainPanel/ChartTab/CandleChart.tsx`

- [ ] **Step 1: CandleChart.tsx 수정**

`frontend/src/components/MainPanel/ChartTab/CandleChart.tsx` 전체 교체:

```tsx
import { useEffect, useRef } from 'react'
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts'
import type { IChartApi } from 'lightweight-charts'
import type { Candle } from '@/types'

interface CandleChartProps {
  candles: Candle[]
  onChartReady?: (chart: IChartApi) => void
}

export function CandleChart({ candles, onChartReady }: CandleChartProps) {
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

    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    series.setData(candles)
    chart.timeScale().fitContent()
    onChartReady?.(chart)

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

- [ ] **Step 2: 테스트 실행 (회귀 없음 확인)**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `64 passed`

- [ ] **Step 3: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/CandleChart.tsx
git commit -m "feat: add onChartReady callback to CandleChart"
git push origin seogu-Jeong
```

---

## Task 4: PredictionOverlay 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/ChartTab/PredictionOverlay.tsx`

- [ ] **Step 1: PredictionOverlay.tsx 생성**

`frontend/src/components/MainPanel/ChartTab/PredictionOverlay.tsx`:

```tsx
import { useEffect, useRef } from 'react'
import { LineSeries } from 'lightweight-charts'
import type { IChartApi } from 'lightweight-charts'

interface PredictionData {
  bullish: number[]
  base: number[]
  bearish: number[]
  confidence: number
}

interface PredictionOverlayProps {
  chart: IChartApi | null
  prediction: PredictionData
  lastCandleTime: string
}

function addBusinessDays(dateStr: string, days: number): string {
  const date = new Date(dateStr)
  let added = 0
  while (added < days) {
    date.setDate(date.getDate() + 1)
    const day = date.getDay()
    if (day !== 0 && day !== 6) added++
  }
  return date.toISOString().split('T')[0]
}

export function PredictionOverlay({ chart, prediction, lastCandleTime }: PredictionOverlayProps) {
  const seriesRef = useRef<ReturnType<IChartApi['addSeries']>[]>([])

  useEffect(() => {
    if (!chart) return

    // 기존 시리즈 제거
    seriesRef.current.forEach((s) => {
      try { chart.removeSeries(s) } catch {}
    })
    seriesRef.current = []

    const configs = [
      { values: prediction.bullish, color: '#58a6ff' },
      { values: prediction.base,    color: '#c9d1d9' },
      { values: prediction.bearish, color: '#f85149' },
    ]

    configs.forEach(({ values, color }) => {
      const series = chart.addSeries(LineSeries, {
        color,
        lineWidth: 1,
        lineStyle: 2,  // dashed
        lastValueVisible: false,
        priceLineVisible: false,
      })

      const data = values.map((value, i) => ({
        time: addBusinessDays(lastCandleTime, i + 1),
        value,
      }))

      series.setData(data)
      seriesRef.current.push(series)
    })

    return () => {
      seriesRef.current.forEach((s) => {
        try { chart.removeSeries(s) } catch {}
      })
      seriesRef.current = []
    }
  }, [chart, prediction, lastCandleTime])

  return null
}
```

- [ ] **Step 2: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `64 passed` (DOM 의존 컴포넌트, 단위 테스트 없음)

- [ ] **Step 3: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/PredictionOverlay.tsx
git commit -m "feat: add PredictionOverlay with Bullish/Base/Bearish dashed lines"
git push origin seogu-Jeong
```

---

## Task 5: PatternBadges 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/ChartTab/PatternBadges.tsx`
- Create: `frontend/src/test/PatternBadges.test.tsx`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/PatternBadges.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { PatternBadges } from '@/components/MainPanel/ChartTab/PatternBadges'
import type { CandlePattern } from '@/types'

const mockPatterns: CandlePattern[] = [
  { name: '망치형', type: 'bullish', description: '반전 신호' },
  { name: '도지', type: 'neutral', description: '불확실성' },
  { name: '상승장악형', type: 'bearish', description: '하락 신호' },
]

describe('PatternBadges', () => {
  it('renders pattern names', () => {
    render(<PatternBadges patterns={mockPatterns} />)
    expect(screen.getByText('망치형')).toBeInTheDocument()
    expect(screen.getByText('도지')).toBeInTheDocument()
  })

  it('renders nothing when patterns is empty', () => {
    const { container } = render(<PatternBadges patterns={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('bullish badge has green styling', () => {
    const { container } = render(<PatternBadges patterns={[mockPatterns[0]]} />)
    expect(container.querySelector('.text-green-400')).toBeInTheDocument()
  })

  it('bearish badge has red styling', () => {
    const { container } = render(<PatternBadges patterns={[mockPatterns[2]]} />)
    expect(container.querySelector('.text-red-400')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: PatternBadges.tsx 구현**

`frontend/src/components/MainPanel/ChartTab/PatternBadges.tsx`:

```tsx
import type { CandlePattern } from '@/types'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface PatternBadgesProps {
  patterns: CandlePattern[]
}

const typeStyles: Record<CandlePattern['type'], string> = {
  bullish: 'text-green-400 bg-green-400/10 border-green-400/30',
  bearish: 'text-red-400 bg-red-400/10 border-red-400/30',
  neutral: 'text-gray-400 bg-gray-400/10 border-gray-400/30',
}

export function PatternBadges({ patterns }: PatternBadgesProps) {
  if (patterns.length === 0) return null

  return (
    <TooltipProvider>
      <div className="flex items-center gap-1.5 px-4 py-1.5 border-b border-border">
        <span className="text-xs text-muted-foreground mr-1">패턴:</span>
        {patterns.slice(0, 5).map((p) => (
          <Tooltip key={p.name}>
            <TooltipTrigger asChild>
              <span
                className={cn(
                  'text-xs px-2 py-0.5 rounded-full border cursor-help font-medium',
                  typeStyles[p.type]
                )}
              >
                {p.name}
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-xs max-w-48">{p.description}</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  )
}
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `68 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/PatternBadges.tsx frontend/src/test/PatternBadges.test.tsx
git commit -m "feat: add PatternBadges with tooltips"
git push origin seogu-Jeong
```

---

## Task 6: ChartTab — PredictionOverlay + PatternBadges 통합

**Files:**
- Modify: `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx`

- [ ] **Step 1: ChartTab.tsx 전체 교체**

```tsx
// frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
import { useStockStore } from '@/store/stockStore'
import { MOCK_CANDLES, MOCK_STOCK_DETAILS, MOCK_PREDICTION, MOCK_PATTERNS } from '@/lib/mockData'
import { calculateRSI, calculateMACD } from '@/lib/indicators'
import { useStockWebSocket } from '@/hooks/useStockWebSocket'
import { StockInfoBar } from './StockInfoBar'
import { CandleChart } from './CandleChart'
import { PredictionOverlay } from './PredictionOverlay'
import { PatternBadges } from './PatternBadges'
import { RSIChart } from './RSIChart'
import { MACDChart } from './MACDChart'
import { useMemo, useRef, useState } from 'react'
import type { IChartApi } from 'lightweight-charts'

export function ChartTab() {
  const { selectedStock, realtimePrice } = useStockStore()
  const { isConnected } = useStockWebSocket(selectedStock?.code ?? '')
  const [chart, setChart] = useState<IChartApi | null>(null)

  const rsiData = useMemo(() => calculateRSI(MOCK_CANDLES), [])
  const macdData = useMemo(() => calculateMACD(MOCK_CANDLES), [])
  const lastCandleTime = MOCK_CANDLES[MOCK_CANDLES.length - 1]?.time ?? '2026-01-01'

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
      <PatternBadges patterns={MOCK_PATTERNS} />
      <div className="flex flex-col flex-1 min-h-0 gap-0.5 p-1">
        <div className="flex-[3] min-h-0">
          <CandleChart candles={MOCK_CANDLES} onChartReady={setChart} />
          <PredictionOverlay
            chart={chart}
            prediction={MOCK_PREDICTION}
            lastCandleTime={lastCandleTime}
          />
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

- [ ] **Step 2: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `68 passed` (회귀 없음)

- [ ] **Step 3: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/ChartTab/ChartTab.tsx
git commit -m "feat: integrate PredictionOverlay and PatternBadges into ChartTab"
git push origin seogu-Jeong
```

---

## Task 7: SignalCard 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/AITab/SignalCard.tsx`
- Create: `frontend/src/test/SignalCard.test.tsx`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/SignalCard.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { SignalCard } from '@/components/MainPanel/AITab/SignalCard'

describe('SignalCard', () => {
  it('renders BUY signal', () => {
    render(<SignalCard signal="BUY" signal_score={78} confidence={0.81} />)
    expect(screen.getByText('BUY')).toBeInTheDocument()
  })

  it('renders score', () => {
    render(<SignalCard signal="BUY" signal_score={78} confidence={0.81} />)
    expect(screen.getByText(/78/)).toBeInTheDocument()
  })

  it('renders SELL signal with red style', () => {
    const { container } = render(<SignalCard signal="SELL" signal_score={20} confidence={0.7} />)
    expect(container.querySelector('.text-red-400')).toBeInTheDocument()
  })

  it('renders HOLD signal', () => {
    render(<SignalCard signal="HOLD" signal_score={50} confidence={0.6} />)
    expect(screen.getByText('HOLD')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: SignalCard.tsx 구현**

`frontend/src/components/MainPanel/AITab/SignalCard.tsx`:

```tsx
import { cn } from '@/lib/utils'

interface SignalCardProps {
  signal: 'BUY' | 'HOLD' | 'SELL'
  signal_score: number
  confidence: number
}

const signalStyles = {
  BUY:  { text: 'text-green-400', bg: 'bg-green-400/10', border: 'border-green-400/30' },
  HOLD: { text: 'text-gray-400',  bg: 'bg-gray-400/10',  border: 'border-gray-400/30' },
  SELL: { text: 'text-red-400',   bg: 'bg-red-400/10',   border: 'border-red-400/30' },
}

export function SignalCard({ signal, signal_score, confidence }: SignalCardProps) {
  const styles = signalStyles[signal]

  return (
    <div className={cn('rounded-lg border p-4 text-center', styles.bg, styles.border)}>
      <div className={cn('text-3xl font-bold mb-1', styles.text)}>{signal}</div>
      <div className="text-2xl font-semibold text-foreground mb-1">{signal_score}<span className="text-sm text-muted-foreground">/100</span></div>
      <div className="text-xs text-muted-foreground">신뢰도 {Math.round(confidence * 100)}%</div>
    </div>
  )
}
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `72 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/AITab/SignalCard.tsx frontend/src/test/SignalCard.test.tsx
git commit -m "feat: add SignalCard component"
git push origin seogu-Jeong
```

---

## Task 8: ScoreBreakdown 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/AITab/ScoreBreakdown.tsx`
- Create: `frontend/src/test/ScoreBreakdown.test.tsx`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/ScoreBreakdown.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { ScoreBreakdown } from '@/components/MainPanel/AITab/ScoreBreakdown'

describe('ScoreBreakdown', () => {
  it('renders 기술적 지표 label', () => {
    render(<ScoreBreakdown tech_score={72} lstm_score={84} confidence={0.81} />)
    expect(screen.getByText('기술적 지표')).toBeInTheDocument()
  })

  it('renders LSTM 예측 label', () => {
    render(<ScoreBreakdown tech_score={72} lstm_score={84} confidence={0.81} />)
    expect(screen.getByText('LSTM 예측')).toBeInTheDocument()
  })

  it('renders confidence as percentage', () => {
    render(<ScoreBreakdown tech_score={72} lstm_score={84} confidence={0.81} />)
    expect(screen.getByText(/81%/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: ScoreBreakdown.tsx 구현**

`frontend/src/components/MainPanel/AITab/ScoreBreakdown.tsx`:

```tsx
import { Progress } from '@/components/ui/progress'

interface ScoreBreakdownProps {
  tech_score: number
  lstm_score: number
  confidence: number
}

export function ScoreBreakdown({ tech_score, lstm_score, confidence }: ScoreBreakdownProps) {
  const items = [
    { label: '기술적 지표', value: tech_score, max: 100 },
    { label: 'LSTM 예측', value: lstm_score, max: 100 },
    { label: '신뢰도', value: Math.round(confidence * 100), max: 100, suffix: '%' },
  ]

  return (
    <div className="space-y-3 p-4 rounded-lg bg-card border border-border">
      <h3 className="text-sm font-semibold text-foreground">점수 분해</h3>
      {items.map(({ label, value, suffix }) => (
        <div key={label}>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-muted-foreground">{label}</span>
            <span className="text-foreground font-medium">{value}{suffix ?? ''}</span>
          </div>
          <Progress value={value} className="h-1.5" />
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `75 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/AITab/ScoreBreakdown.tsx frontend/src/test/ScoreBreakdown.test.tsx
git commit -m "feat: add ScoreBreakdown with progress bars"
git push origin seogu-Jeong
```

---

## Task 9: MultiframePanel 컴포넌트

**Files:**
- Create: `frontend/src/components/MainPanel/AITab/MultiframePanel.tsx`
- Create: `frontend/src/test/MultiframePanel.test.tsx`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/MultiframePanel.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { MultiframePanel } from '@/components/MainPanel/AITab/MultiframePanel'
import type { MultiframeSignal } from '@/types'

const mockData: MultiframeSignal[] = [
  { timeframe: '1D', signal: 'BUY', score: 78 },
  { timeframe: '1W', signal: 'HOLD', score: 52 },
  { timeframe: '1M', signal: 'BUY', score: 65 },
]

describe('MultiframePanel', () => {
  it('renders all timeframes', () => {
    render(<MultiframePanel signals={mockData} />)
    expect(screen.getByText('1D')).toBeInTheDocument()
    expect(screen.getByText('1W')).toBeInTheDocument()
    expect(screen.getByText('1M')).toBeInTheDocument()
  })

  it('renders signals for each timeframe', () => {
    render(<MultiframePanel signals={mockData} />)
    expect(screen.getAllByText('BUY').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('HOLD')).toBeInTheDocument()
  })

  it('renders scores', () => {
    render(<MultiframePanel signals={mockData} />)
    expect(screen.getByText(/78/)).toBeInTheDocument()
    expect(screen.getByText(/52/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: MultiframePanel.tsx 구현**

`frontend/src/components/MainPanel/AITab/MultiframePanel.tsx`:

```tsx
import type { MultiframeSignal } from '@/types'
import { cn } from '@/lib/utils'

interface MultiframePanelProps {
  signals: MultiframeSignal[]
}

const signalColor = {
  BUY:  'text-green-400',
  HOLD: 'text-gray-400',
  SELL: 'text-red-400',
}

export function MultiframePanel({ signals }: MultiframePanelProps) {
  return (
    <div className="p-4 rounded-lg bg-card border border-border">
      <h3 className="text-sm font-semibold text-foreground mb-3">멀티 타임프레임</h3>
      <div className="grid grid-cols-3 gap-2">
        {signals.map((s) => (
          <div key={s.timeframe} className="text-center p-3 rounded bg-background border border-border">
            <div className="text-xs text-muted-foreground mb-1">{s.timeframe}</div>
            <div className={cn('text-base font-bold', signalColor[s.signal])}>{s.signal}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{s.score}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `78 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/AITab/MultiframePanel.tsx frontend/src/test/MultiframePanel.test.tsx
git commit -m "feat: add MultiframePanel component"
git push origin seogu-Jeong
```

---

## Task 10: AITab — 전체 조립

**Files:**
- Create: `frontend/src/components/MainPanel/AITab/AITab.tsx`
- Create: `frontend/src/test/AITab.test.tsx`

- [ ] **Step 1: 테스트 먼저 작성**

`frontend/src/test/AITab.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { AITab } from '@/components/MainPanel/AITab/AITab'

describe('AITab', () => {
  it('renders signal (BUY/HOLD/SELL)', () => {
    render(<AITab />)
    expect(
      screen.getByText(/BUY|HOLD|SELL/)
    ).toBeInTheDocument()
  })

  it('renders 점수 분해 section', () => {
    render(<AITab />)
    expect(screen.getByText('점수 분해')).toBeInTheDocument()
  })

  it('renders 멀티 타임프레임 section', () => {
    render(<AITab />)
    expect(screen.getByText('멀티 타임프레임')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: AITab.tsx 구현**

`frontend/src/components/MainPanel/AITab/AITab.tsx`:

```tsx
import { MOCK_AI_SIGNAL, MOCK_MULTIFRAME } from '@/lib/mockData'
import { SignalCard } from './SignalCard'
import { ScoreBreakdown } from './ScoreBreakdown'
import { MultiframePanel } from './MultiframePanel'

export function AITab() {
  const { signal, signal_score, tech_score, lstm_score, confidence } = MOCK_AI_SIGNAL

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <SignalCard
        signal={signal}
        signal_score={signal_score}
        confidence={confidence}
      />
      <ScoreBreakdown
        tech_score={tech_score}
        lstm_score={lstm_score}
        confidence={confidence}
      />
      <MultiframePanel signals={MOCK_MULTIFRAME} />
    </div>
  )
}
```

- [ ] **Step 3: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `81 passed`

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/AITab/ frontend/src/test/AITab.test.tsx
git commit -m "feat: add AITab assembling SignalCard, ScoreBreakdown, MultiframePanel"
git push origin seogu-Jeong
```

---

## Task 11: MainPanel — AITab 연결

**Files:**
- Modify: `frontend/src/components/MainPanel/MainPanel.tsx`

- [ ] **Step 1: MainPanel.tsx 수정**

```tsx
// frontend/src/components/MainPanel/MainPanel.tsx
import { useUIStore } from '@/store/uiStore'
import { ChartTab } from './ChartTab/ChartTab'
import { AITab } from './AITab/AITab'

const PLACEHOLDER_TABS = ['simulator', 'portfolio', 'screener', 'backtest'] as const

function PlaceholderTab({ name }: { name: string }) {
  return (
    <div className="flex items-center justify-center h-full text-muted-foreground">
      {name} — Phase 4에서 구현 예정
    </div>
  )
}

export function MainPanel() {
  const { activeTab } = useUIStore()

  return (
    <div className="flex-1 min-w-0 min-h-0 overflow-hidden">
      {activeTab === 'chart' && <ChartTab />}
      {activeTab === 'ai' && <AITab />}
      {PLACEHOLDER_TABS.map((tab) =>
        activeTab === tab ? <PlaceholderTab key={tab} name={tab} /> : null
      )}
    </div>
  )
}
```

- [ ] **Step 2: 테스트 실행**

```bash
cd ~/FinalProject/frontend && npm run test:run
```

Expected: `81 passed`

- [ ] **Step 3: dev 서버 동작 확인**

```bash
cd ~/FinalProject/frontend && npm run dev
```

확인 항목:
1. 차트탭 → StockInfoBar 아래 패턴 배지 (망치형, 도지, 상승장악형)
2. 캔들스틱 차트 끝에서 Bullish(파랑), Base(흰), Bearish(빨강) 점선 3개
3. AI 탭 클릭 → BUY/HOLD/SELL 시그널 카드
4. 점수 분해 progress bar
5. 1D/1W/1M 멀티 타임프레임 카드
6. Ctrl+C 종료

- [ ] **Step 4: 커밋**

```bash
cd ~/FinalProject
git add frontend/src/components/MainPanel/MainPanel.tsx
git commit -m "feat: connect AITab to MainPanel"
git push origin seogu-Jeong
```

---

## Task 12: CLAUDE.md 완료 표시 + dev 머지

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md §12 업데이트**

`~/FinalProject/CLAUDE.md`의 Phase 3 항목:

```markdown
- [x] Phase 3 — AI 기능 (프론트엔드 완료 2026-06-02)
```

- [ ] **Step 2: 커밋 + dev 머지**

```bash
cd ~/FinalProject
git add CLAUDE.md
git commit -m "docs: mark Phase 3 frontend complete"
git push origin seogu-Jeong

git checkout dev
git pull origin dev
git merge seogu-Jeong
git push origin dev
git checkout seogu-Jeong
```
