import { render, screen, fireEvent } from '@testing-library/react'
import { Header } from '@/components/Layout/Header'

describe('Header', () => {
  it('renders logo', () => {
    render(<Header onLoginClick={() => {}} />)
    expect(screen.getByText('StockSenseAI')).toBeInTheDocument()
  })

  it('renders search input', () => {
    render(<Header onLoginClick={() => {}} />)
    expect(screen.getByPlaceholderText('종목명 또는 코드')).toBeInTheDocument()
  })

  it('calls onLoginClick when login button clicked', () => {
    const fn = vi.fn()
    render(<Header onLoginClick={fn} />)
    fireEvent.click(screen.getByText('로그인'))
    expect(fn).toHaveBeenCalledTimes(1)
  })

  it('shows search results when typing', async () => {
    render(<Header onLoginClick={() => {}} />)
    const input = screen.getByPlaceholderText('종목명 또는 코드')
    fireEvent.change(input, { target: { value: '삼성' } })
    expect(await screen.findByText('삼성전자')).toBeInTheDocument()
  })
})
