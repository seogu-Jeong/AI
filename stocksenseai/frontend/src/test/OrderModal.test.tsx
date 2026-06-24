import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { OrderModal } from '@/components/Trade/OrderModal'
import { MOCK_STOCKS } from '@/lib/mockData'
import api from '@/lib/api'

vi.mock('@/lib/api', () => ({
  default: { get: vi.fn(), post: vi.fn() },
}))

const stock = MOCK_STOCKS[0]

describe('OrderModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.get).mockResolvedValue({
      data: { mode: 'paper', account_no: '1234****-01' },
    })
    vi.mocked(api.post).mockResolvedValue({ data: { status: 'PENDING' } })
  })

  it('renders BUY modal title', () => {
    render(<OrderModal open={true} onClose={() => {}} stock={stock} orderType="BUY" />)
    expect(screen.getByText(/매수/)).toBeInTheDocument()
  })

  it('renders SELL modal title', () => {
    render(<OrderModal open={true} onClose={() => {}} stock={stock} orderType="SELL" />)
    expect(screen.getByText(/매도/)).toBeInTheDocument()
  })

  it('renders stock name', () => {
    render(<OrderModal open={true} onClose={() => {}} stock={stock} orderType="BUY" />)
    expect(screen.getByText('삼성전자')).toBeInTheDocument()
  })

  it('calls onClose when cancelled', () => {
    const fn = vi.fn()
    render(<OrderModal open={true} onClose={fn} stock={stock} orderType="BUY" />)
    fireEvent.click(screen.getByText('취소'))
    expect(fn).toHaveBeenCalled()
  })

  it('requires a second confirmation before a real-account order', async () => {
    vi.mocked(api.get).mockResolvedValue({
      data: { mode: 'real', account_no: '1234****-01' },
    })
    render(<OrderModal open={true} onClose={() => {}} stock={stock} orderType="BUY" />)

    await screen.findByText('실계좌 · 1234****-01')
    fireEvent.click(screen.getByRole('button', { name: '주문' }))

    expect(await screen.findByText('실계좌 주문을 실행하시겠습니까?')).toBeInTheDocument()
    expect(api.post).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: '실계좌 주문 확인' }))
    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(1))
  })
})
