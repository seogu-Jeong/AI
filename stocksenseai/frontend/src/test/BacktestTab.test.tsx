import { render, screen } from '@testing-library/react'
import { BacktestTab } from '@/components/MainPanel/BacktestTab/BacktestTab'

describe('BacktestTab', () => {
  it('renders 백테스팅 heading', () => {
    render(<BacktestTab />)
    expect(screen.getByText('백테스팅')).toBeInTheDocument()
  })

  it('renders run button', () => {
    render(<BacktestTab />)
    expect(screen.getByRole('button', { name: /백테스트 실행/ })).toBeInTheDocument()
  })
})
