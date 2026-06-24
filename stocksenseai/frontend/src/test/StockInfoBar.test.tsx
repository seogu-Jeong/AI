import { render, screen } from '@testing-library/react'
import { StockInfoBar } from '@/components/MainPanel/ChartTab/StockInfoBar'
import type { Stock, StockDetail } from '@/types'

const mockStock: Stock = { code: '005930', name: '삼성전자', price: 73400, change_pct: 1.2 }
const mockDetail: StockDetail = { open: 72800, high: 74200, low: 72100, volume: 12300000 }

describe('StockInfoBar', () => {
  it('renders stock name and code', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText('삼성전자')).toBeInTheDocument()
    expect(screen.getByText('005930')).toBeInTheDocument()
  })

  it('renders current price', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText(/73,400/)).toBeInTheDocument()
  })

  it('renders OHLV detail cards', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText('시가')).toBeInTheDocument()
    expect(screen.getByText('고가')).toBeInTheDocument()
    expect(screen.getByText('저가')).toBeInTheDocument()
    expect(screen.getByText('거래량')).toBeInTheDocument()
  })

  it('shows live indicator when isLive is true', () => {
    render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={true} />)
    expect(screen.getByText('실시간')).toBeInTheDocument()
  })

  it('shows green color for positive change', () => {
    const { container } = render(<StockInfoBar stock={mockStock} detail={mockDetail} isLive={false} />)
    expect(container.querySelector('.text-green-500')).toBeInTheDocument()
  })
})
