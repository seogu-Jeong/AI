import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { cn } from '@/lib/utils'
import api from '@/lib/api'
import type { Holding, PortfolioMetrics, PortfolioResponse } from '@/types'

export function PortfolioTab() {
  const [holdings, setHoldings] = useState<Holding[]>([])
  const [totalEval, setTotalEval] = useState(0)
  const [totalReturnPct, setTotalReturnPct] = useState(0)
  const [holdingSource, setHoldingSource] = useState('')
  const [performanceSource, setPerformanceSource] = useState('')
  const [metrics, setMetrics] = useState<PortfolioMetrics | null>(null)
  const [chartData, setChartData] = useState<{ date: string; value: number }[]>([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [holdingsLoaded, setHoldingsLoaded] = useState(false)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [portfolioRes, metricsRes, perfRes] = await Promise.all([
          api.get('/portfolio'),
          api.get('/portfolio/metrics'),
          api.get('/portfolio/performance'),
        ])
        const p = portfolioRes.data as PortfolioResponse
        setHoldings(p.holdings ?? [])
        setTotalEval(p.total_eval ?? 0)
        setTotalReturnPct(p.total_return_pct ?? 0)
        setHoldingSource(p.holding_source ?? '')
        setPerformanceSource(p.performance_source ?? '')
        setMetrics(metricsRes.data)
        setHoldingsLoaded(true)
        setFetchError(null)
        // Convert daily PNL list to cumulative chart data
        const perf: { date: string; pnl: number }[] = perfRes.data ?? []
        let cum = 0
        setChartData(perf.map((d) => { cum += d.pnl; return { date: d.date, value: cum } }))
      } catch (e: unknown) {
        const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        setFetchError(msg ?? '포트폴리오 데이터를 불러오지 못했습니다.')
        setHoldingsLoaded(false)
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
  }, [])

  if (loading) {
    return <div className="h-full flex items-center justify-center text-muted-foreground text-sm">로딩 중...</div>
  }

  if (fetchError) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 px-6 text-center">
        <span className="text-sm text-destructive">포트폴리오 조회 실패</span>
        <span className="text-xs text-muted-foreground">{fetchError}</span>
        <span className="text-xs text-muted-foreground mt-1">로그인 여부와 서버 상태를 확인해 주세요.</span>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* 데이터 출처 설명 박스 */}
      <div
        data-testid="portfolio-description-box"
        className="text-xs text-muted-foreground bg-muted/40 border border-border rounded-lg px-3 py-2.5 space-y-1"
      >
        {holdingSource ? (
          <>
            <p>
              보유 현황은{' '}
              <span className="text-foreground font-medium">{holdingSource}</span>
              에서 조회합니다.
            </p>
            <p>
              수익 추이, 승률, MDD는{' '}
              {performanceSource
                ? <span className="text-foreground font-medium">{performanceSource}</span>
                : '앱에서 발생한 체결 기록'}
              을 기준으로 계산합니다.
            </p>
          </>
        ) : (
          <p className="text-yellow-400">
            KIS 잔고 조회에 실패해 앱 DB 포트폴리오 기록을 표시합니다.
          </p>
        )}
      </div>

      {/* 소스 배지 — holdingSource가 없을 때만 표시 (description box와 중복 방지) */}
      {!holdingSource && performanceSource && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <div className="text-xs text-muted-foreground bg-card border border-border rounded-lg px-3 py-2">
            성과 지표: <span className="text-foreground">{performanceSource}</span>
          </div>
        </div>
      )}

      {/* 요약 카드 */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <div className="text-xs text-muted-foreground mb-1">총 평가액</div>
          <div className="text-sm font-bold">{(totalEval / 10000).toFixed(0)}만원</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <div className="text-xs text-muted-foreground mb-1">수익률</div>
          <div className={cn('text-sm font-bold', totalReturnPct >= 0 ? 'text-green-400' : 'text-red-400')}>
            {totalReturnPct >= 0 ? '+' : ''}{totalReturnPct.toFixed(1)}%
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <div className="text-xs text-muted-foreground mb-1">MDD</div>
          <div className="text-sm font-bold text-red-400">
            {metrics ? `-${metrics.mdd_pct.toFixed(1)}%` : '-'}
          </div>
        </div>
      </div>

      {/* 성과 지표 */}
      {metrics && (
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-card border border-border rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">승률</div>
            <div className="text-sm font-bold">{metrics.win_rate_pct.toFixed(1)}%</div>
          </div>
          <div className="bg-card border border-border rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">샤프비율</div>
            <div className="text-sm font-bold">{metrics.sharpe_ratio.toFixed(2)}</div>
          </div>
          <div className="bg-card border border-border rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">총 거래수</div>
            <div className="text-sm font-bold">{metrics.total_trades}</div>
          </div>
        </div>
      )}

      {/* 수익 추이 차트 */}
      {chartData.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-3">
          <div className="text-sm font-semibold mb-3">누적 손익 추이</div>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={(v) => v.slice(5)} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={(v) => `${(v / 10000).toFixed(0)}만`} />
              <Tooltip
                formatter={(v) => [`${Number(v ?? 0).toLocaleString('ko-KR')}원`, '누적 손익']}
                contentStyle={{ background: '#161b22', border: '1px solid #30363d', fontSize: 11 }}
              />
              <Line type="monotone" dataKey="value" stroke="#58a6ff" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 보유종목 */}
      <div className="bg-card border border-border rounded-lg p-3">
        <div className="text-sm font-semibold mb-3">보유종목</div>
        {holdingsLoaded && holdings.length === 0 ? (
          <div className="text-xs text-muted-foreground text-center py-4">
            조회 성공 · 현재 보유 중인 종목이 없습니다
          </div>
        ) : (
          <div className="space-y-2">
            {holdings.map((h) => (
              <div key={h.stock_code} className="flex items-center justify-between text-sm py-1.5 border-b border-border last:border-0">
                <div>
                  <div className="font-medium">{h.stock_name}</div>
                  <div className="text-xs text-muted-foreground">{h.quantity}주 · 평균 {h.avg_price.toLocaleString('ko-KR')}원</div>
                </div>
                <div className="text-right">
                  <div>{h.current_price.toLocaleString('ko-KR')}원</div>
                  <div className={cn('text-xs', h.return_pct >= 0 ? 'text-green-400' : 'text-red-400')}>
                    {h.return_pct >= 0 ? '+' : ''}{h.return_pct.toFixed(2)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
