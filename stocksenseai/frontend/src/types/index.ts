// frontend/src/types/index.ts
// 공동 관리 파일 — 수정 전 반드시 hygrenn과 협의 (CLAUDE.md §5)

export interface User {
  id: string
  email: string
  mode: 'demo' | 'paper' | 'real'
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
  time: string | number
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
  eval_amount: number
  profit_loss: number
  return_pct: number
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

export type TabId = 'chart' | 'ai' | 'simulator' | 'portfolio' | 'screener' | 'backtest' | 'recommend' | 'market' | 'autotrade'

export interface Stock {
  code: string
  name: string
  price?: number
  change_pct?: number
}

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
  time: string | number
  macd: number
  signal: number
  histogram: number
}

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

export interface OrderBookEntry {
  price: number
  quantity: number
}

export interface PortfolioMetrics {
  total_trades: number
  win_rate_pct: number
  sharpe_ratio: number
  mdd_pct: number
}

export interface PortfolioResponse {
  holdings: Holding[]
  total_eval: number
  total_cost: number
  total_return_pct: number
  total_asset?: number
  deposit?: number
  holding_source?: string
  performance_source?: string
}

export interface RiskSettings {
  max_per_stock_pct: number
  daily_loss_limit_pct: number
  stop_loss_enabled: boolean
  enforce_hard_stop: boolean
  trading_blocked: boolean
}

// ── 시스템 상태 진단 API (/system/status) ──────────────────────────────────
export interface SystemStatusResponse {
  backend: {
    ok: boolean
    message: string
  }
  auth: {
    logged_in: boolean
    email: string | null
  }
  kis: {
    mode: 'paper' | 'real' | null
    configured: boolean
    account_no: string | null
    message: string
  }
  account: {
    ok: boolean | null
    holdings_count: number | null
    data_source: string | null
    message: string
  }
  portfolio: {
    ok: boolean | null
    holding_source: string | null
    performance_source: string | null
    message: string
  }
  ai: {
    prediction_source: 'uploaded' | 'local' | 'unavailable'
    message: string
  }
  checked_at: string
}
