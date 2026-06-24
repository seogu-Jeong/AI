import { render, screen } from '@testing-library/react'
import { MultiframePanel } from '@/components/MainPanel/AITab/MultiframePanel'
import type { MultiframeSignal } from '@/types'

const mockData: MultiframeSignal[] = [
  { timeframe: '1D', signal: 'BUY', score: 78 },
  { timeframe: '1W', signal: 'HOLD', score: 52 },
  { timeframe: '1M', signal: 'BUY', score: 65 },
]

describe('MultiframePanel', () => {
  it('renders all timeframes', () => {
    render(<MultiframePanel signals={mockData} />)
    expect(screen.getByText('1D')).toBeInTheDocument()
    expect(screen.getByText('1W')).toBeInTheDocument()
    expect(screen.getByText('1M')).toBeInTheDocument()
  })

  it('renders signals for each timeframe', () => {
    render(<MultiframePanel signals={mockData} />)
    expect(screen.getAllByText('BUY').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('HOLD')).toBeInTheDocument()
  })

  it('renders scores', () => {
    render(<MultiframePanel signals={mockData} />)
    expect(screen.getByText(/78/)).toBeInTheDocument()
    expect(screen.getByText(/52/)).toBeInTheDocument()
  })
})
