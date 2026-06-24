# StockSenseAI Frontend Phase 4 — Design

**Date:** 2026-06-02
**Author:** seogu-Jeong
**Branch:** seogu-Jeong

---

## 1. 목표

CLAUDE.md Phase 4 프론트엔드 요구사항 구현:
- 주문 모달 + 확인 플로우 (매수/매도)
- 호가창 10단 (우측 패널)
- 포트폴리오 현황 + 수익률 차트
- 투자 시뮬레이터 탭
- 백테스팅 UI
- 리스크 설정 폼

---

## 2. 변경 범위

### 신규 파일

| 파일 | 역할 |
|---|---|
| `frontend/src/components/Trade/OrderBook.tsx` | 호가창 10단 |
| `frontend/src/components/Trade/OrderModal.tsx` | 주문 모달 (매수/매도) |
| `frontend/src/components/Risk/RiskSettingsModal.tsx` | 리스크 설정 모달 |
| `frontend/src/components/MainPanel/PortfolioTab/PortfolioTab.tsx` | 포트폴리오 탭 |
| `frontend/src/components/MainPanel/SimulatorTab/SimulatorTab.tsx` | 시뮬레이터 탭 |
| `frontend/src/components/MainPanel/BacktestTab/BacktestTab.tsx` | 백테스팅 탭 |

### 수정 파일

| 파일 | 변경 내용 |
|---|---|
| `frontend/src/components/Layout/MainLayout.tsx` | 우측 패널에 OrderBook 활성화 |
| `frontend/src/components/MainPanel/ChartTab/StockInfoBar.tsx` | 매수/매도 버튼 추가 |
| `frontend/src/components/MainPanel/MainPanel.tsx` | PortfolioTab/SimulatorTab/BacktestTab 연결 |
| `frontend/src/types/index.ts` | OrderBookEntry, PortfolioMetrics 타입 추가 |
| `frontend/src/lib/mockData.ts` | Mock 호가/포트폴리오/시뮬/백테스트 데이터 추가 |

---

## 3. 컴포넌트 설계

### 3-1. OrderBook (호가창 10단)

```typescript
interface OrderBookEntry {
  price: number
  quantity: number
}
interface OrderBookProps {
  asks: OrderBookEntry[]  // 매도호가 (10개, 높은가격→낮은가격)
  bids: OrderBookEntry[]  // 매수호가 (10개, 높은가격→낮은가격)
  currentPrice: number
}
```

- 매도호가: 빨강, 위에 표시
- 매수호가: 초록, 아래 표시
- 현재가 중간에 표시
- 우측 패널 (`hidden lg:block w-56`)에 배치

### 3-2. OrderModal

StockInfoBar에 매수(초록)/매도(빨강) 버튼 추가 → Dialog 팝업

```typescript
interface OrderModalProps {
  open: boolean
  onClose: () => void
  stock: Stock
  orderType: 'BUY' | 'SELL'
}
```

- MARKET/LIMIT 주문 타입 선택
- 수량/가격 입력
- paper 모드 확인 후 제출 (실제 API 호출 X)
- 제출 시 toast 알림

### 3-3. PortfolioTab

```
PortfolioTab
  ├─ 요약 카드 (총 평가액, 총 수익률, MDD)
  ├─ 보유종목 테이블 (종목/수량/평균가/현재가/수익률)
  └─ 수익률 차트 (Recharts LineChart)
```

### 3-4. SimulatorTab

일시불 시뮬레이션 폼:
- 종목 선택 (드롭다운)
- 매수일 / 매도일 선택
- 투자금액 입력
- 결과: 수익/손실 + 수익률 카드

### 3-5. BacktestTab

백테스팅 UI:
- 종목/기간/전략(MA교차/RSI) 선택
- 실행 버튼 → Mock 결과 즉시 표시
- 결과: 수익률/승률/MDD 카드

### 3-6. RiskSettingsModal

리스크 설정:
- 1회 최대 투자 비중 (%)
- 손절 기준 (%)
- 일일 최대 손실 한도 (원)

Header 오른쪽에 ⚙️ 버튼으로 열기

---

## 4. Mock 데이터 추가

```typescript
MOCK_ORDER_BOOK: { asks: OrderBookEntry[], bids: OrderBookEntry[] }
MOCK_HOLDINGS: Holding[]
MOCK_PORTFOLIO_PERFORMANCE: { date: string; value: number }[]
MOCK_PORTFOLIO_METRICS: { total_value: number; total_return_pct: number; mdd: number }
MOCK_BACKTEST_RESULT: { return_pct: number; win_rate: number; mdd: number; trades: number }
```

---

## 5. shadcn 추가 컴포넌트

```bash
npx shadcn@latest add select toast sonner
```

---

## 6. 구현 순서

1. types/index.ts — OrderBookEntry, PortfolioMetrics 타입
2. mockData.ts — Phase 4 Mock 데이터
3. OrderBook.tsx + OrderModal.tsx
4. StockInfoBar 매수/매도 버튼 + MainLayout 우측패널 활성화
5. PortfolioTab
6. SimulatorTab
7. BacktestTab
8. RiskSettingsModal + Header ⚙️ 버튼
9. MainPanel 탭 연결
10. CLAUDE.md 완료 + dev 머지
