import { render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import { RecommendTab } from '@/components/MainPanel/RecommendTab/RecommendTab'
import { buildReasonChips } from '@/lib/recommendReasons'

// ── mock stores ──────────────────────────────────────────────────────────────
vi.mock('@/store/stockStore', () => ({
  useStockStore: () => ({
    selectedStock: null,
    setSelectedStock: vi.fn(),
  }),
}))

// ── mock api ─────────────────────────────────────────────────────────────────
vi.mock('@/lib/api', () => ({
  default: {
    get: vi.fn(() =>
      Promise.resolve({
        data: {
          scanned: 100,
          total: 3,
          ranking: [
            {
              code: '005930',
              name: '삼성전자',
              signal: 'BUY',
              signal_score: 75,
              tech_score: 72,
              lstm_score: 65,
              lstm_available: true,
            },
            {
              code: '000660',
              name: 'SK하이닉스',
              signal: 'HOLD',
              signal_score: 55,
              tech_score: 30,
              lstm_score: null,
              lstm_available: false,
            },
            {
              code: '035420',
              name: 'NAVER',
              signal: 'SELL',
              signal_score: 25,
              tech_score: 20,
              lstm_score: 35,
              lstm_available: true,
            },
          ],
        },
      })
    ),
  },
}))

// ── mock panels (not under test) ──────────────────────────────────────────────
vi.mock('@/components/Analysis/ComprehensivePanel', () => ({
  ComprehensivePanel: () => <div data-testid="comprehensive-panel" />,
}))
vi.mock('@/components/Analysis/FundamentalPanel', () => ({
  FundamentalPanel: () => <div data-testid="fundamental-panel" />,
}))

// ─────────────────────────────────────────────────────────────────────────────

describe('buildReasonChips (unit)', () => {
  it('returns AI 점수 우수 when signal_score >= 70', () => {
    const chips = buildReasonChips({ signal: 'BUY', signal_score: 75, tech_score: 50, lstm_score: null, lstm_available: false })
    expect(chips.map((c) => c.label)).toContain('AI 점수 우수')
  })

  it('returns AI 점수 양호 when signal_score 50~69', () => {
    const chips = buildReasonChips({ signal: 'HOLD', signal_score: 55, tech_score: 50, lstm_score: null, lstm_available: false })
    expect(chips.map((c) => c.label)).toContain('AI 점수 양호')
  })

  it('returns 기술 점수 강세 when tech_score >= 70', () => {
    const chips = buildReasonChips({ signal: 'BUY', signal_score: 60, tech_score: 72, lstm_score: null, lstm_available: false })
    expect(chips.map((c) => c.label)).toContain('기술 점수 강세')
  })

  it('returns 기술 점수 약세 when tech_score <= 35', () => {
    const chips = buildReasonChips({ signal: 'SELL', signal_score: 20, tech_score: 30, lstm_score: 20, lstm_available: true })
    expect(chips.map((c) => c.label)).toContain('기술 점수 약세')
  })

  it('returns LSTM 긍정 when lstm_available and lstm_score >= 60', () => {
    const chips = buildReasonChips({ signal: 'BUY', signal_score: 60, tech_score: 60, lstm_score: 65, lstm_available: true })
    expect(chips.map((c) => c.label)).toContain('LSTM 긍정')
  })

  it('returns LSTM 부정 when lstm_available and lstm_score <= 40', () => {
    const chips = buildReasonChips({ signal: 'SELL', signal_score: 30, tech_score: 30, lstm_score: 35, lstm_available: true })
    expect(chips.map((c) => c.label)).toContain('LSTM 부정')
  })

  it('returns LSTM 미사용 when lstm_available is false', () => {
    const chips = buildReasonChips({ signal: 'HOLD', signal_score: 55, tech_score: 50, lstm_score: null, lstm_available: false })
    expect(chips.map((c) => c.label)).toContain('LSTM 미사용')
  })

  it('returns 매수 후보 for BUY signal when few other chips fire', () => {
    // signal_score=45 (no score chip), tech_score=50 (no tech chip), lstm_available=false (LSTM 미사용 only)
    const chips = buildReasonChips({ signal: 'BUY', signal_score: 45, tech_score: 50, lstm_score: null, lstm_available: false })
    expect(chips.map((c) => c.label)).toContain('매수 후보')
  })

  it('returns 주의 후보 for SELL signal when few other chips fire', () => {
    // signal_score=45 (no score chip), tech_score=50 (no tech chip), lstm_available=false (LSTM 미사용 only)
    const chips = buildReasonChips({ signal: 'SELL', signal_score: 45, tech_score: 50, lstm_score: null, lstm_available: false })
    expect(chips.map((c) => c.label)).toContain('주의 후보')
  })

  it('never returns more than 3 chips', () => {
    const chips = buildReasonChips({ signal: 'BUY', signal_score: 80, tech_score: 75, lstm_score: 70, lstm_available: true })
    expect(chips.length).toBeLessThanOrEqual(3)
  })

  it('always returns at least 1 chip', () => {
    const chips = buildReasonChips({ signal: 'HOLD', signal_score: 40, tech_score: 50, lstm_score: null, lstm_available: false })
    expect(chips.length).toBeGreaterThanOrEqual(1)
  })
})

describe('RecommendTab (integration)', () => {
  it('renders ranked stock names', async () => {
    render(<RecommendTab />)
    await waitFor(() => expect(screen.getByText('삼성전자')).toBeInTheDocument())
    expect(screen.getByText('SK하이닉스')).toBeInTheDocument()
    expect(screen.getByText('NAVER')).toBeInTheDocument()
  })

  it('shows reason chips for each ranked stock', async () => {
    render(<RecommendTab />)
    // 삼성전자: BUY, score=75 → 'AI 점수 우수' + '기술 점수 강세' + 'LSTM 긍정'
    await waitFor(() => expect(screen.getByText('AI 점수 우수')).toBeInTheDocument())
    expect(screen.getByText('기술 점수 강세')).toBeInTheDocument()
  })

  it('shows LSTM 미사용 chip for stock without LSTM', async () => {
    render(<RecommendTab />)
    // SK하이닉스: lstm_available=false
    await waitFor(() => expect(screen.getByText('LSTM 미사용')).toBeInTheDocument())
  })

  it('shows 주의 후보 chip for SELL signal stock', async () => {
    render(<RecommendTab />)
    // NAVER: signal=SELL
    await waitFor(() => expect(screen.getByText('주의 후보')).toBeInTheDocument())
  })
})
