# StockSenseAI Frontend Phase 3 — Design

**Date:** 2026-06-02
**Author:** seogu-Jeong
**Branch:** seogu-Jeong

---

## 1. 목표

CLAUDE.md Phase 3 프론트엔드 요구사항 구현:
- AI 예측 오버레이 (차트 위 Bullish/Base/Bearish 3시나리오 점선)
- 캔들 패턴 배지 + 툴팁 (ChartTab 헤더)
- AI 시그널 카드 + 점수 분해 (AITab)
- 멀티 타임프레임 패널 (AITab)

백엔드 미완성 → Mock 데이터로 UI 먼저 완성. 교체 지점 명확히 분리.

---

## 2. 변경 범위

### 신규 파일

| 파일 | 역할 |
|---|---|
| `frontend/src/components/MainPanel/ChartTab/PredictionOverlay.tsx` | Bullish/Base/Bearish 점선 오버레이 |
| `frontend/src/components/MainPanel/ChartTab/PatternBadges.tsx` | 캔들 패턴 배지 + 툴팁 |
| `frontend/src/components/MainPanel/AITab/AITab.tsx` | AITab 루트 |
| `frontend/src/components/MainPanel/AITab/SignalCard.tsx` | AI 시그널 카드 (BUY/HOLD/SELL) |
| `frontend/src/components/MainPanel/AITab/ScoreBreakdown.tsx` | 점수 분해 (tech/lstm/confidence) |
| `frontend/src/components/MainPanel/AITab/MultiframePanel.tsx` | 멀티 타임프레임 시그널 |

### 수정 파일

| 파일 | 변경 내용 |
|---|---|
| `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` | PredictionOverlay + PatternBadges 추가 |
| `frontend/src/components/MainPanel/MainPanel.tsx` | AITab 연결 |
| `frontend/src/lib/mockData.ts` | MOCK_AI_SIGNAL, MOCK_PREDICTION, MOCK_PATTERNS, MOCK_MULTIFRAME 추가 |
| `frontend/src/types/index.ts` | CandlePattern, MultiframeSignal 타입 추가 |

---

## 3. 컴포넌트 설계

### 3-1. PredictionOverlay

CandleChart와 **동일한 lightweight-charts 인스턴스**를 공유하면 구현이 복잡해짐.
→ 대신 ChartTab 내에서 CandleChart 위에 **별도 LineSeries를 추가하는 방식**으로 구현.
→ CandleChart가 chart 인스턴스를 ref로 노출하거나, ChartTab이 직접 chart를 생성하는 방식 선택.

**선택: ChartTab이 chart 인스턴스를 생성하고 CandleChart + PredictionOverlay에 전달**

```
ChartTab
  ├─ chart 인스턴스 생성 (useRef)
  ├─ CandleChart (chart prop 수신)
  ├─ PredictionOverlay (chart prop + prediction data 수신)
  └─ PatternBadges (패턴 목록 수신)
```

PredictionOverlay props:
```typescript
interface PredictionOverlayProps {
  chart: IChartApi | null
  prediction: Prediction  // { bullish: number[], base: number[], bearish: number[], confidence: number }
  lastCandleTime: string  // 마지막 캔들 날짜 기준으로 미래 5일 계산
}
```

- Bullish: 파란 점선 (`#58a6ff`, `lineStyle: 1`)
- Base: 흰 점선 (`#c9d1d9`, `lineStyle: 1`)
- Bearish: 빨간 점선 (`#f85149`, `lineStyle: 1`)

### 3-2. PatternBadges

ChartTab 헤더(StockInfoBar 아래)에 작은 배지로 표시.

```typescript
interface CandlePattern {
  name: string        // e.g. "망치형", "도지", "상승장악형"
  type: 'bullish' | 'bearish' | 'neutral'
  description: string // 툴팁 텍스트
}
```

- 최대 5개 표시, 색상: bullish=초록, bearish=빨강, neutral=회색
- hover 시 `description` 툴팁 표시 (shadcn Tooltip 컴포넌트 사용)

### 3-3. AITab

```
AITab
  ├─ SignalCard        (BUY/HOLD/SELL + 전체 점수)
  ├─ ScoreBreakdown   (기술적 점수 / LSTM 점수 / 신뢰도 바 차트)
  └─ MultiframePanel  (1D/1W/1M 각 타임프레임 시그널)
```

### 3-4. SignalCard

```typescript
interface SignalCardProps {
  signal: 'BUY' | 'HOLD' | 'SELL'
  signal_score: number   // 0~100
  confidence: number
}
```

- BUY: 초록 배경, SELL: 빨간 배경, HOLD: 회색 배경
- 점수 크게 표시 (e.g. 78/100)

### 3-5. ScoreBreakdown

```typescript
interface ScoreBreakdownProps {
  tech_score: number    // 0~100
  lstm_score: number    // 0~100
  confidence: number    // 0~1
}
```

- 각 항목에 progress bar (shadcn Progress 컴포넌트 사용)
- 기술적 지표 / LSTM 예측 / 신뢰도 3개 항목

### 3-6. MultiframePanel

```typescript
interface MultiframeSignal {
  timeframe: '1D' | '1W' | '1M'
  signal: 'BUY' | 'HOLD' | 'SELL'
  score: number
}
```

- 3개 카드 (1D / 1W / 1M)
- 각 카드에 시그널 + 점수

---

## 4. CandleChart 리팩터링

현재 CandleChart는 chart 인스턴스를 내부에서 관리함.
PredictionOverlay를 같은 chart에 추가하려면 chart를 밖으로 노출해야 함.

**방법: `onChartReady` 콜백 prop 추가**

```typescript
interface CandleChartProps {
  candles: Candle[]
  onChartReady?: (chart: IChartApi) => void
}
```

ChartTab에서 chart ref를 받아 PredictionOverlay에 전달.

---

## 5. Mock 데이터

```typescript
// lib/mockData.ts에 추가

export const MOCK_AI_SIGNAL: AISignal = {
  signal: 'BUY',
  signal_score: 78,
  tech_score: 72,
  lstm_score: 84,
  confidence: 0.81,
  indicators: { rsi_14: 62.4, macd: 12.3, ... }
}

export const MOCK_PREDICTION: Prediction = {
  bullish: [74200, 75100, 76300, 77500, 78200],
  base:    [73800, 74300, 74900, 75400, 75800],
  bearish: [73200, 72800, 72100, 71500, 70900],
  confidence: 0.81,
}

export const MOCK_PATTERNS: CandlePattern[] = [
  { name: '망치형', type: 'bullish', description: '하락 추세 후 반전 가능성을 나타내는 패턴' },
  { name: '도지', type: 'neutral', description: '시장 불확실성을 나타내는 패턴' },
  { name: '상승장악형', type: 'bullish', description: '강한 매수 신호' },
]

export const MOCK_MULTIFRAME: MultiframeSignal[] = [
  { timeframe: '1D', signal: 'BUY', score: 78 },
  { timeframe: '1W', signal: 'HOLD', score: 52 },
  { timeframe: '1M', signal: 'BUY', score: 65 },
]
```

---

## 6. 백엔드 연동 교체 계획

| Mock | 실제 API |
|---|---|
| `MOCK_AI_SIGNAL` | `GET /ai/{code}/signal` |
| `MOCK_PREDICTION` | `GET /ai/{code}/predict` |
| `MOCK_PATTERNS` | `GET /ai/{code}/patterns` |
| `MOCK_MULTIFRAME` | `GET /ai/{code}/multiframe` |

---

## 7. shadcn 컴포넌트 추가 필요

```bash
npx shadcn@latest add tooltip progress
```

---

## 8. 구현 순서

1. types/index.ts — CandlePattern, MultiframeSignal 타입 추가
2. mockData.ts — AI Mock 데이터 추가
3. CandleChart.tsx — onChartReady 콜백 추가
4. PredictionOverlay.tsx — 3시나리오 점선
5. PatternBadges.tsx — 배지 + 툴팁
6. ChartTab.tsx — PredictionOverlay + PatternBadges 통합
7. SignalCard.tsx + ScoreBreakdown.tsx
8. MultiframePanel.tsx
9. AITab.tsx — 조립
10. MainPanel.tsx — AITab 연결
11. CLAUDE.md 완료 + dev 머지
