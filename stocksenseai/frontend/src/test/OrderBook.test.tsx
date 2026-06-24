import { render, screen } from '@testing-library/react'
import { OrderBook } from '@/components/Trade/OrderBook'
import { MOCK_ORDER_BOOK } from '@/lib/mockData'

describe('OrderBook', () => {
  it('renders 매도/매수 labels', () => {
    render(<OrderBook asks={MOCK_ORDER_BOOK.asks} bids={MOCK_ORDER_BOOK.bids} currentPrice={73400} />)
    expect(screen.getByText('매도')).toBeInTheDocument()
    expect(screen.getByText('매수')).toBeInTheDocument()
  })

  it('renders 10 ask rows', () => {
    render(<OrderBook asks={MOCK_ORDER_BOOK.asks} bids={MOCK_ORDER_BOOK.bids} currentPrice={73400} />)
    const rows = screen.getAllByText(/74,200|74,100|74,000/)
    expect(rows.length).toBeGreaterThan(0)
  })

  it('renders current price', () => {
    render(<OrderBook asks={MOCK_ORDER_BOOK.asks} bids={MOCK_ORDER_BOOK.bids} currentPrice={73400} />)
    expect(screen.getAllByText(/73,400/).length).toBeGreaterThan(0)
  })
})
