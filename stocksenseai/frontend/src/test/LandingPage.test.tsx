import { render, screen, fireEvent } from '@testing-library/react'
import { LandingPage } from '@/pages/LandingPage'

describe('LandingPage', () => {
  it('renders title', () => {
    render(<LandingPage onEnter={() => {}} />)
    expect(screen.getByText('SenseAI')).toBeInTheDocument()
  })

  it('calls onEnter when 둘러보기 clicked', () => {
    const fn = vi.fn()
    render(<LandingPage onEnter={fn} />)
    fireEvent.click(screen.getByText('둘러보기 (데모)'))
    expect(fn).toHaveBeenCalled()
  })

  it('opens login modal on 로그인 click', () => {
    render(<LandingPage onEnter={() => {}} />)
    fireEvent.click(screen.getByText('로그인'))
    expect(screen.getAllByText('로그인').length).toBeGreaterThan(1)
  })
})
