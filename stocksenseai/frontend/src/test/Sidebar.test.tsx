import { render, screen, fireEvent } from '@testing-library/react'
import { Sidebar } from '@/components/Sidebar/Sidebar'

describe('Sidebar', () => {
  it('renders 관심종목 group', () => {
    render(<Sidebar />)
    expect(screen.getByText('관심종목')).toBeInTheDocument()
  })

  it('renders 전체종목 group', () => {
    render(<Sidebar />)
    expect(screen.getByText('전체종목')).toBeInTheDocument()
  })

  it('collapses group on click', () => {
    render(<Sidebar />)
    const btn = screen.getByText('관심종목')
    fireEvent.click(btn)
    expect(screen.queryAllByText('삼성전자').length).toBeLessThan(2)
  })
})
