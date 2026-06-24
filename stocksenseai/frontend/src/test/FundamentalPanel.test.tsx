import { render, screen, waitFor } from '@testing-library/react'
import { FundamentalPanel } from '@/components/Analysis/FundamentalPanel'
import { useStockStore } from '@/store/stockStore'
import api from '@/lib/api'

vi.mock('@/lib/api', () => ({ default: { get: vi.fn() } }))
const mockedGet = api.get as unknown as ReturnType<typeof vi.fn>

describe('FundamentalPanel', () => {
  beforeEach(() => {
    useStockStore.setState({ selectedStock: { code: '005930', name: '삼성전자' } })
  })

  it('재무 점수를 5.0 만점 소수 1자리로 표시한다', async () => {
    mockedGet.mockResolvedValue({
      data: {
        code: '005930', available: true, score: 3.7, max_score: 5.0,
        grade: '양호', risk: false, risk_threshold: 2.5,
        reasons: ['PER 12.0 적정'], metrics: { per: 12, pbr: 1.5, eps: 5000, bps: 70000, dividend_yield: 2 },
      },
    })
    render(<FundamentalPanel />)
    await waitFor(() => expect(screen.getByText('3.7')).toBeInTheDocument())
    expect(screen.getByText('/ 5.0')).toBeInTheDocument()
  })

  it('기준 미만이면 위험 경고를 표시한다', async () => {
    mockedGet.mockResolvedValue({
      data: {
        code: '005930', available: true, score: 2.0, max_score: 5.0,
        grade: '위험', risk: true, risk_threshold: 2.5,
        reasons: ['PER 26.0 고평가'], metrics: { per: 26, pbr: 4.5, eps: 12000, bps: 71000, dividend_yield: null },
      },
    })
    render(<FundamentalPanel />)
    // '위험'은 등급 배지와 경고 문구 두 곳에 나타난다.
    await waitFor(() => expect(screen.getAllByText(/위험/).length).toBeGreaterThan(0))
  })
})
