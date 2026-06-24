import { render, screen } from '@testing-library/react'
import { PatternBadges } from '@/components/MainPanel/ChartTab/PatternBadges'
import type { CandlePattern } from '@/types'

const mockPatterns: CandlePattern[] = [
  { name: '망치형', type: 'bullish', description: '반전 신호' },
  { name: '도지', type: 'neutral', description: '불확실성' },
  { name: '상승장악형', type: 'bearish', description: '하락 신호' },
]

describe('PatternBadges', () => {
  it('renders pattern names', () => {
    render(<PatternBadges patterns={mockPatterns} />)
    expect(screen.getByText('망치형')).toBeInTheDocument()
    expect(screen.getByText('도지')).toBeInTheDocument()
  })

  it('renders nothing when patterns is empty', () => {
    const { container } = render(<PatternBadges patterns={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('bullish badge has green styling', () => {
    const { container } = render(<PatternBadges patterns={[mockPatterns[0]]} />)
    expect(container.querySelector('.text-green-400')).toBeInTheDocument()
  })

  it('bearish badge has red styling', () => {
    const { container } = render(<PatternBadges patterns={[mockPatterns[2]]} />)
    expect(container.querySelector('.text-red-400')).toBeInTheDocument()
  })
})
