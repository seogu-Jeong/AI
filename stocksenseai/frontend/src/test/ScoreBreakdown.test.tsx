import { render, screen } from '@testing-library/react'
import { ScoreBreakdown } from '@/components/MainPanel/AITab/ScoreBreakdown'

describe('ScoreBreakdown', () => {
  it('renders 기술적 지표 label', () => {
    render(<ScoreBreakdown tech_score={72} lstm_score={84} confidence={0.81} />)
    expect(screen.getByText('기술적 지표')).toBeInTheDocument()
  })

  it('renders LSTM 예측 label', () => {
    render(<ScoreBreakdown tech_score={72} lstm_score={84} confidence={0.81} />)
    expect(screen.getByText('LSTM 예측')).toBeInTheDocument()
  })

  it('renders confidence as percentage', () => {
    render(<ScoreBreakdown tech_score={72} lstm_score={84} confidence={0.81} />)
    expect(screen.getByText(/81%/)).toBeInTheDocument()
  })
})
