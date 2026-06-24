import { render, screen } from '@testing-library/react'
import { AITab } from '@/components/MainPanel/AITab/AITab'

describe('AITab', () => {
  it('renders signal (BUY/HOLD/SELL)', () => {
    render(<AITab />)
    expect(
      screen.getAllByText(/BUY|HOLD|SELL/).length
    ).toBeGreaterThan(0)
  })

  it('renders 점수 분해 section', () => {
    render(<AITab />)
    expect(screen.getByText('점수 분해')).toBeInTheDocument()
  })

  it('renders 멀티 타임프레임 section', () => {
    render(<AITab />)
    expect(screen.getByText('멀티 타임프레임')).toBeInTheDocument()
  })
})
