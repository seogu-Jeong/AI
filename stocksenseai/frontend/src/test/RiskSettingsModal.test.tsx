import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import { RiskSettingsModal } from '@/components/Risk/RiskSettingsModal'

vi.mock('@/lib/api', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: { max_per_stock_pct: 20, daily_loss_limit_pct: 5, stop_loss_enabled: false, enforce_hard_stop: false, trading_blocked: false } }),
    put: vi.fn().mockResolvedValue({ data: { updated: true } }),
  },
}))

describe('RiskSettingsModal', () => {
  it('renders when open', () => {
    render(<RiskSettingsModal open={true} onClose={() => {}} />)
    expect(screen.getByText('리스크 설정')).toBeInTheDocument()
  })

  it('does not render when closed', () => {
    render(<RiskSettingsModal open={false} onClose={() => {}} />)
    expect(screen.queryByText('리스크 설정')).not.toBeInTheDocument()
  })

  it('calls onClose when saved', async () => {
    const fn = vi.fn()
    render(<RiskSettingsModal open={true} onClose={fn} />)
    fireEvent.click(screen.getByText('저장'))
    await waitFor(() => expect(fn).toHaveBeenCalled())
  })
})
