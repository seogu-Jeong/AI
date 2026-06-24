import { render, screen } from '@testing-library/react'
import { SimulatorTab } from '@/components/MainPanel/SimulatorTab/SimulatorTab'

describe('SimulatorTab', () => {
  it('renders 투자 시뮬레이터 heading', () => {
    render(<SimulatorTab />)
    expect(screen.getByText('투자 시뮬레이터')).toBeInTheDocument()
  })

  it('renders 투자금액 input label', () => {
    render(<SimulatorTab />)
    expect(screen.getByText('투자금액')).toBeInTheDocument()
  })

  it('renders simulate button', () => {
    render(<SimulatorTab />)
    expect(screen.getByRole('button', { name: /시뮬레이션/ })).toBeInTheDocument()
  })
})
