import { render, screen } from '@testing-library/react'
import { WatchlistPanel } from '@/components/WatchlistPanel/WatchlistPanel'

describe('WatchlistPanel', () => {
  it('renders watchlist stocks', () => {
    render(<WatchlistPanel />)
    expect(screen.getByText('삼성전자')).toBeInTheDocument()
  })
})
