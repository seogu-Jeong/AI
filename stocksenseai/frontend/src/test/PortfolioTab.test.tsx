import { render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import { PortfolioTab } from '@/components/MainPanel/PortfolioTab/PortfolioTab'

vi.mock('@/lib/api', () => ({
  default: {
    get: vi.fn((url: string) => {
      if (url === '/portfolio') return Promise.resolve({ data: {
        holdings: [{ stock_code: '005930', stock_name: '삼성전자', quantity: 10, avg_price: 70000, current_price: 73400, eval_amount: 734000, profit_loss: 34000, return_pct: 4.86 }],
        total_eval: 5290000,
        total_cost: 5000000,
        total_return_pct: 5.8,
        holding_source: 'KIS 모의투자 계좌',
        performance_source: '앱 거래 기록 기준',
      }})
      if (url === '/portfolio/metrics') return Promise.resolve({ data: { total_trades: 24, win_rate_pct: 62.5, sharpe_ratio: 1.23, mdd_pct: 3.2 } })
      if (url === '/portfolio/performance') return Promise.resolve({ data: [] })
      return Promise.resolve({ data: {} })
    }),
  },
}))

describe('PortfolioTab', () => {
  it('renders 총 평가액', async () => {
    render(<PortfolioTab />)
    await waitFor(() => expect(screen.getByText('총 평가액')).toBeInTheDocument())
  })

  it('renders 보유종목 heading', async () => {
    render(<PortfolioTab />)
    await waitFor(() => expect(screen.getByText('보유종목')).toBeInTheDocument())
  })

  it('renders holding stock names', async () => {
    render(<PortfolioTab />)
    await waitFor(() => expect(screen.getByText('삼성전자')).toBeInTheDocument())
  })

  it('renders data source labels', async () => {
    render(<PortfolioTab />)
    // text appears in description box (badge is hidden when holdingSource is set)
    await waitFor(() => expect(screen.getAllByText('KIS 모의투자 계좌').length).toBeGreaterThanOrEqual(1))
    expect(screen.getAllByText('앱 거래 기록 기준').length).toBeGreaterThanOrEqual(1)
  })

  it('renders data source description box', async () => {
    render(<PortfolioTab />)
    await waitFor(() => {
      const box = screen.getByTestId('portfolio-description-box')
      expect(box).toBeInTheDocument()
    })
  })

  it('description box contains KIS 계좌 출처 문구', async () => {
    render(<PortfolioTab />)
    await waitFor(() => {
      expect(screen.getByTestId('portfolio-description-box')).toBeInTheDocument()
    })
    expect(screen.getByTestId('portfolio-description-box').textContent).toContain('보유 현황은')
    expect(screen.getByTestId('portfolio-description-box').textContent).toContain('KIS 모의투자 계좌')
  })

  it('description box shows fallback text when holding_source is absent', async () => {
    const { default: api } = await import('@/lib/api')
    vi.mocked(api.get).mockImplementation((url: string) => {
      if (url === '/portfolio') return Promise.resolve({ data: {
        holdings: [],
        total_eval: 0,
        total_cost: 0,
        total_return_pct: 0,
        holding_source: '',
        performance_source: '',
      }})
      if (url === '/portfolio/metrics') return Promise.resolve({ data: { total_trades: 0, win_rate_pct: 0, sharpe_ratio: 0, mdd_pct: 0 } })
      if (url === '/portfolio/performance') return Promise.resolve({ data: [] })
      return Promise.resolve({ data: {} })
    })
    render(<PortfolioTab />)
    await waitFor(() => {
      expect(screen.getByTestId('portfolio-description-box').textContent).toContain(
        'KIS 잔고 조회에 실패해 앱 DB 포트폴리오 기록을 표시합니다.'
      )
    })
  })
})
