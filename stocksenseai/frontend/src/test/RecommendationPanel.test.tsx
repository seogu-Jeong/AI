import { render, screen, waitFor } from '@testing-library/react'
import { RecommendationPanel } from '@/components/Analysis/RecommendationPanel'
import api from '@/lib/api'

vi.mock('@/lib/api', () => ({ default: { get: vi.fn() } }))
const mockedGet = api.get as unknown as ReturnType<typeof vi.fn>

describe('RecommendationPanel', () => {
  it('BUY 추천 종목과 스캔 요약을 표시한다', async () => {
    mockedGet.mockResolvedValue({
      data: {
        picks: [
          { code: '069960', name: '현대백화점', signal: 'BUY', signal_score: 86.1, financial_score: 3.7, financial_grade: '양호', market_caution: false },
        ],
        scanned: 99, buy_count: 1, market_trend: 'up', market_caution: false,
      },
    })
    render(<RecommendationPanel />)
    await waitFor(() => expect(screen.getByText('현대백화점')).toBeInTheDocument())
    expect(screen.getByText(/99종목 스캔/)).toBeInTheDocument()
    expect(screen.getByText(/BUY 86/)).toBeInTheDocument()
  })

  it('추천이 없으면 안내 문구를 표시한다', async () => {
    mockedGet.mockResolvedValue({
      data: { picks: [], scanned: 99, buy_count: 0, market_trend: 'neutral', market_caution: false },
    })
    render(<RecommendationPanel />)
    await waitFor(() => expect(screen.getByText(/추천 종목이 없습니다/)).toBeInTheDocument())
  })
})
