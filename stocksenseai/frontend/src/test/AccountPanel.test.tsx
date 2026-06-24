import { render, screen, waitFor } from '@testing-library/react'
import { AccountPanel } from '@/components/Account/AccountPanel'
import api from '@/lib/api'
import { useAuthStore } from '@/store/authStore'

vi.mock('@/lib/api', () => ({
  default: { get: vi.fn() },
}))

describe('AccountPanel', () => {
  beforeEach(() => {
    useAuthStore.setState({
      user: { id: '1', email: 'user@test.com', mode: 'demo', is_verified: true, dark_mode: true },
    })
    vi.mocked(api.get).mockResolvedValue({
      data: {
        mode: 'real',
        account_no: '1234****-01',
        summary: {
          total_asset: 1000000,
          deposit: 200000,
          eval_amount: 800000,
          buy_amount: 700000,
          eval_profit_loss: 100000,
          return_pct: 14.29,
        },
        holdings: [],
        data_source: 'KIS 실계좌 계좌',
      },
    })
  })

  it('loads the configured account without a query mode and shows real account warning', async () => {
    render(<AccountPanel onClose={() => {}} />)

    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/account/balance'))
    expect(await screen.findByText('실계좌')).toBeInTheDocument()
    expect(screen.getByText('실제 주문이 이 계좌로 실행됩니다.')).toBeInTheDocument()
  })

  it('renders account-mode-info element', async () => {
    render(<AccountPanel onClose={() => {}} />)
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/trades?limit=10'))
    const modeInfo = screen.getByTestId('account-mode-info')
    expect(modeInfo).toBeInTheDocument()
  })

  it('account-mode-info shows current mode label from API response', async () => {
    render(<AccountPanel onClose={() => {}} />)
    // API mock returns mode: 'real' → label should be '실계좌'
    await waitFor(() => {
      const modeInfo = screen.getByTestId('account-mode-info')
      expect(modeInfo.textContent).toContain('실계좌')
    })
  })

  it('account-mode-info defaults demo user mode to paper when data is not yet loaded', async () => {
    vi.mocked(api.get).mockImplementation((url: string) => {
      if (url === '/trades?limit=10') return Promise.resolve({ data: [] })
      return new Promise(() => {})
    })
    render(<AccountPanel onClose={() => {}} />)
    expect(await screen.findByText('최근 주문 내역이 없습니다')).toBeInTheDocument()
    const modeInfo = screen.getByTestId('account-mode-info')
    expect(modeInfo).toBeInTheDocument()
    expect(modeInfo.textContent).toContain('모의투자')
  })

  it('account-mode-info shows paper mode label when user mode is paper', async () => {
    useAuthStore.setState({
      user: { id: '1', email: 'user@test.com', mode: 'paper', is_verified: true, dark_mode: true },
    })
    vi.mocked(api.get).mockResolvedValue({
      data: {
        mode: 'paper',
        account_no: '1234****-01',
        summary: {
          total_asset: 1000000,
          deposit: 200000,
          eval_amount: 800000,
          buy_amount: 700000,
          eval_profit_loss: 100000,
          return_pct: 14.29,
        },
        holdings: [],
        data_source: 'KIS 모의투자 계좌',
      },
    })
    render(<AccountPanel onClose={() => {}} />)
    await waitFor(() => {
      const modeInfo = screen.getByTestId('account-mode-info')
      expect(modeInfo.textContent).toContain('모의투자')
    })
  })

  it('account-mode-info shows settings navigation hint text', async () => {
    render(<AccountPanel onClose={() => {}} />)
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/trades?limit=10'))
    const modeInfo = screen.getByTestId('account-mode-info')
    expect(modeInfo.textContent).toContain('실계좌/모의계좌 전환은 리스크/설정 화면에서만 변경합니다.')
  })
})
