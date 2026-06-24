import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { SystemStatusPanel } from '@/components/System/SystemStatusPanel'
import api from '@/lib/api'
import type { SystemStatusResponse } from '@/types'

vi.mock('@/lib/api', () => ({
  default: { get: vi.fn() },
}))

const mockStatusOk: SystemStatusResponse = {
  backend: { ok: true, message: '백엔드 정상' },
  auth: { logged_in: true, email: 'user@test.com' },
  kis: { mode: 'paper', configured: true, account_no: '1234****-01', message: 'KIS 모의투자 설정됨' },
  account: { ok: true, holdings_count: 2, data_source: 'KIS 모의투자 계좌', message: '잔고 조회 성공 · 보유 2종목' },
  portfolio: { ok: true, holding_source: 'KIS 모의투자 계좌', performance_source: '앱 거래 기록 기준', message: '보유 현황은 KIS 모의투자 계좌 기준입니다.' },
  ai: { prediction_source: 'local', message: '로컬 LSTM 가중치 5종목 사용 가능' },
  checked_at: '2026-06-19T15:30:00+09:00',
}

const mockStatusNoKis: SystemStatusResponse = {
  backend: { ok: true, message: '백엔드 정상' },
  auth: { logged_in: false, email: null },
  kis: { mode: null, configured: false, account_no: null, message: 'KIS API 키 미설정' },
  account: { ok: null, holdings_count: null, data_source: null, message: 'login_required' },
  portfolio: { ok: null, holding_source: null, performance_source: null, message: 'login_required' },
  ai: { prediction_source: 'unavailable', message: 'LSTM 가중치 없음 — AI 예측 비활성' },
  checked_at: '2026-06-19T10:00:00+09:00',
}

const mockStatusEmptyHoldings: SystemStatusResponse = {
  ...mockStatusOk,
  account: { ok: true, holdings_count: 0, data_source: 'KIS 모의투자 계좌', message: '잔고 조회 성공 · 보유 종목 없음' },
}

describe('SystemStatusPanel', () => {
  const onClose = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders panel title', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusOk })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/system/status'))
    expect(screen.getByRole('dialog', { name: '시스템 상태 패널' })).toBeInTheDocument()
    expect(screen.getByText('시스템 상태')).toBeInTheDocument()
  })

  it('shows status rows after successful load', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusOk })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/system/status'))
    // 인증 row
    expect(await screen.findByText('user@test.com')).toBeInTheDocument()
    // KIS row
    expect(screen.getByText('모의투자')).toBeInTheDocument()
  })

  it('shows holdings count when KIS balance ok', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusOk })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
    expect(await screen.findByText('2종목 보유')).toBeInTheDocument()
  })

  it('shows "보유 없음" label when holdings_count is 0 (조회 성공)', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusEmptyHoldings })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
    expect(await screen.findByText('조회 성공 (보유 없음)')).toBeInTheDocument()
  })

  it('shows "키 미설정" when KIS not configured', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusNoKis })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
    expect(await screen.findByText('키 미설정')).toBeInTheDocument()
  })

  it('shows error message when API call fails', async () => {
    vi.mocked(api.get).mockRejectedValue(new Error('network error'))
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
    expect(await screen.findByText(/상태 조회에 실패/)).toBeInTheDocument()
  })

  it('shows checked_at time in footer', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusOk })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
    expect(await screen.findByText(/최근 확인/)).toBeInTheDocument()
  })

  it('calls onClose when close button clicked', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusOk })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/system/status'))
    fireEvent.click(screen.getByRole('button', { name: '닫기' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('re-fetches when refresh button clicked', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusOk })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: '새로고침' }))
    await waitFor(() => expect(api.get).toHaveBeenCalledTimes(2))
  })

  it('shows login_required portfolio status when not logged in', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: mockStatusNoKis })
    render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
    // portfolio ok is null → "로그인 필요" (both 로그인 row and 포트폴리오 row show this)
    const items = await screen.findAllByText('로그인 필요')
    expect(items.length).toBeGreaterThanOrEqual(1)
  })

  it('does not expose KIS keys or secrets in rendered output', async () => {
    const sensitiveData = {
      ...mockStatusOk,
      kis: { ...mockStatusOk.kis, account_no: '1234****-01' },
    }
    vi.mocked(api.get).mockResolvedValue({ data: sensitiveData })
    const { container } = render(<SystemStatusPanel onClose={onClose} />)
    await waitFor(() => expect(api.get).toHaveBeenCalled())
    const html = container.innerHTML
    expect(html).not.toContain('app_key')
    expect(html).not.toContain('APP_KEY')
    expect(html).not.toContain('app_secret')
  })
})
