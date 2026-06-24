# StockSenseAI Frontend Phase 2 — Design

**Date:** 2026-06-02
**Author:** seogu-Jeong
**Branch:** seogu-Jeong

---

## 1. 목표

CLAUDE.md Phase 2 프론트엔드 요구사항 구현:
- Lightweight Charts 캔들스틱 컴포넌트 분리
- 기술적 지표 서브차트 (RSI, MACD) — 고정 60/20/20 분할
- 실시간 시세 WebSocket 연동 (Mock WebSocket 클래스)
- 종목 상세 정보 (카드형 정보바 — 시가/고가/저가/거래량)

백엔드(hygrenn) 미완성 상태에서 Mock 데이터로 UI 먼저 완성.
백엔드 완성 후 API/WebSocket URL 교체만으로 연동 가능하도록 설계.

---

## 2. 변경 범위

### 신규 파일

| 파일 | 역할 |
|---|---|
| `frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx` | 카드형 종목 정보 (현재가, 시가/고가/저가/거래량) |
| `frontend/src/components/MainPanel/ChartTab/CandleChart.tsx` | 캔들스틱 차트만 (기존 ChartTab에서 분리) |
| `frontend/src/components/MainPanel/ChartTab/RSIChart.tsx` | RSI 서브차트 |
| `frontend/src/components/MainPanel/ChartTab/MACDChart.tsx` | MACD 서브차트 |
| `frontend/src/hooks/useStockWebSocket.ts` | WebSocket 연결/재연결/cleanup 훅 |
| `frontend/src/lib/mockWebSocket.ts` | Mock WebSocket 클래스 (실제 WebSocket API와 동일 인터페이스) |
| `frontend/src/lib/indicators.ts` | RSI, MACD 계산 순수 함수 |

### 수정 파일

| 파일 | 변경 내용 |
|---|---|
| `frontend/src/components/MainPanel/ChartTab/ChartTab.tsx` | 4개 컴포넌트 조립으로 교체 |
| `frontend/src/store/stockStore.ts` | 실시간 가격 업데이트 필드 + action 추가 |
| `frontend/src/lib/mockData.ts` | MOCK_STOCK_DETAIL (시가/고가/저가/거래량) 추가 |

---

## 3. 컴포넌트 설계

### 3-1. ChartTab 레이아웃 (60/20/20 분할)

```
┌─────────────────────────────────┐
│  StockInfoBar                   │  (고정 높이 ~80px)
│  삼성전자 73,400 ▲1.21%         │
│  [시가] [고가] [저가] [거래량]   │
├─────────────────────────────────┤
│                                 │
│  CandleChart (flex:3 = 60%)     │
│                                 │
├─────────────────────────────────┤
│  RSIChart (flex:1 = 20%)        │
├─────────────────────────────────┤
│  MACDChart (flex:1 = 20%)       │
└─────────────────────────────────┘
```

### 3-2. StockInfoBar

```typescript
interface StockInfoBarProps {
  stock: Stock           // 기본 정보 (name, code, price, change_pct)
  detail: StockDetail    // 상세 정보 (open, high, low, volume)
  isLive: boolean        // WebSocket 연결 여부 표시
}
```

- 현재가 크게 표시 + 변동폭 (▲/▼ + %)
- 실시간 연결 상태 표시 (🟢 실시간 / 🔴 연결 끊김)
- 시가/고가/저가/거래량 4칸 카드 그리드

### 3-3. CandleChart

- 기존 ChartTab의 lightweight-charts 로직을 이 파일로 이동
- props: `candles: Candle[]`
- 부모에서 데이터 주입 (ChartTab이 MOCK_CANDLES 전달)
- 리사이즈 처리 자체 관리

### 3-4. RSIChart

- lightweight-charts `addLineSeries`로 RSI 라인 렌더링
- 과매수(70) / 과매도(30) 기준선 표시
- props: `data: { time: string; value: number }[]`
- `indicators.ts`의 `calculateRSI()` 결과를 부모에서 주입

### 3-5. MACDChart

- MACD 라인 + Signal 라인 + 히스토그램 (양수=초록, 음수=빨강)
- props: `data: MACDData[]`
- `indicators.ts`의 `calculateMACD()` 결과를 부모에서 주입

---

## 4. 지표 계산 (indicators.ts)

```typescript
// RSI (14)
calculateRSI(candles: Candle[], period?: number): { time: string; value: number }[]

// MACD (12, 26, 9)
calculateMACD(candles: Candle[]): {
  time: string
  macd: number
  signal: number
  histogram: number
}[]
```

- 순수 함수 — 부수효과 없음
- 입력: `Candle[]`, 출력: 시계열 배열
- Phase 3에서 백엔드 `/ai/{code}/indicators` 연동 시 이 함수 결과를 API 응답으로 교체

---

## 5. WebSocket 설계

### mockWebSocket.ts

```typescript
class MockWebSocket {
  url: string
  onmessage: ((event: { data: string }) => void) | null
  onclose: (() => void) | null
  onopen: (() => void) | null
  readyState: number

  constructor(url: string)  // url 파라미터 받아 실제 WebSocket처럼 동작
  send(data: string): void
  close(): void
}
```

- `constructor` 실행 즉시 `onopen` 호출 (연결 성공 시뮬레이션)
- `setInterval` 2초마다 현재가 ±0.3% 랜덤 변동 → `onmessage` 호출
- 메시지 포맷: `{ code, price, change_pct, volume }`
- `close()` 호출 시 interval 정리 + `onclose` 호출

### useStockWebSocket.ts

```typescript
function useStockWebSocket(stockCode: string): {
  isConnected: boolean
}
```

- 컴포넌트 마운트 시 MockWebSocket 연결
- 메시지 수신 시 `stockStore.updateRealtimePrice()` 호출
- `stockCode` 변경 시 기존 연결 닫고 새 연결
- 언마운트 시 cleanup

### stockStore 추가

```typescript
realtimePrice: number | null
updateRealtimePrice: (code: string, price: number, change_pct: number) => void
```

---

## 6. Mock 데이터 추가

```typescript
// lib/mockData.ts 추가
interface StockDetail {
  open: number
  high: number
  low: number
  volume: number
}

export const MOCK_STOCK_DETAILS: Record<string, StockDetail> = {
  '005930': { open: 72800, high: 74200, low: 72100, volume: 12300000 },
  '000660': { open: 184000, high: 186500, low: 183500, volume: 5200000 },
  // ...
}
```

---

## 7. 백엔드 연동 교체 계획

Phase 2 완료 후 백엔드 준비 시 교체 지점:

| Mock | 실제 API |
|---|---|
| `MOCK_CANDLES` | `GET /stocks/{code}/chart` |
| `MOCK_STOCK_DETAILS` | `GET /stocks/{code}` 응답의 시가/고가/저가/거래량 |
| `MockWebSocket` | `ws://localhost:8000/ws/stocks/{code}` |
| `calculateRSI/MACD` | `GET /ai/{code}/indicators` |

---

## 8. 구현 순서

1. `lib/indicators.ts` — RSI/MACD 계산 함수 + 테스트
2. `lib/mockWebSocket.ts` — Mock WebSocket 클래스 + 테스트
3. `lib/mockData.ts` 업데이트 — StockDetail 추가
4. `store/stockStore.ts` 업데이트 — realtimePrice 필드
5. `hooks/useStockWebSocket.ts` — WebSocket 훅
6. `ChartTab/StockInfoBar.tsx` — 카드형 정보바
7. `ChartTab/CandleChart.tsx` — 캔들스틱 분리
8. `ChartTab/RSIChart.tsx` — RSI 서브차트
9. `ChartTab/MACDChart.tsx` — MACD 서브차트
10. `ChartTab/ChartTab.tsx` 리팩터 — 조립
11. 전체 동작 확인 + dev 머지
