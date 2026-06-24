import type { User, Candle, TabId, StockDetail, CandlePattern, MultiframeSignal, OrderBookEntry, PortfolioMetrics } from '@/types'

describe('types', () => {
  it('User type has required fields', () => {
    const user: User = {
      id: '1',
      email: 'test@test.com',
      mode: 'demo',
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

  it('StockDetail has OHLV fields', () => {
    const detail: StockDetail = { open: 72800, high: 74200, low: 72100, volume: 12300000 }
    expect(detail.high).toBe(74200)
  })

  it('CandlePattern has name, type, description', () => {
    const p: CandlePattern = { name: '망치형', type: 'bullish', description: '반전 신호' }
    expect(p.type).toBe('bullish')
  })

  it('MultiframeSignal has timeframe, signal, score', () => {
    const s: MultiframeSignal = { timeframe: '1D', signal: 'BUY', score: 78 }
    expect(s.signal).toBe('BUY')
  })

  it('OrderBookEntry has price and quantity', () => {
    const entry: OrderBookEntry = { price: 73400, quantity: 500 }
    expect(entry.price).toBe(73400)
  })

  it('PortfolioMetrics has mdd_pct, win_rate_pct, sharpe_ratio, total_trades', () => {
    const m: PortfolioMetrics = { total_trades: 24, win_rate_pct: 62.5, sharpe_ratio: 1.23, mdd_pct: 3.2 }
    expect(m.win_rate_pct).toBe(62.5)
  })
})
