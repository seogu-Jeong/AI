import { render, screen } from '@testing-library/react'
import { SignalCard } from '@/components/MainPanel/AITab/SignalCard'

describe('SignalCard', () => {
  it('renders BUY signal', () => {
    render(<SignalCard signal="BUY" signal_score={78} confidence={0.81} />)
    expect(screen.getByText('BUY')).toBeInTheDocument()
  })

  it('renders score', () => {
    render(<SignalCard signal="BUY" signal_score={78} confidence={0.81} />)
    expect(screen.getByText(/78/)).toBeInTheDocument()
  })

  it('renders SELL signal with red style', () => {
    const { container } = render(<SignalCard signal="SELL" signal_score={20} confidence={0.7} />)
    expect(container.querySelector('.text-red-400')).toBeInTheDocument()
  })

  it('renders HOLD signal', () => {
    render(<SignalCard signal="HOLD" signal_score={50} confidence={0.6} />)
    expect(screen.getByText('HOLD')).toBeInTheDocument()
  })
})
