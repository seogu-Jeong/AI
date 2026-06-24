import { useEffect, useState } from 'react'
import { useStockStore } from '@/store/stockStore'
import { cn } from '@/lib/utils'
import api from '@/lib/api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

interface InvestorDay {
  date: string
  foreign_net: number
  institution_net: number
  individual_net: number
}

interface InvestorSummary {
  foreign_5d: number
  institution_5d: number
  foreign_net_buy: boolean
  institution_net_buy: boolean
}

interface InvestorResponse {
  code: string
  available: boolean
  trend: InvestorDay[]
  summary: InvestorSummary
}

function fmt(n: number): string {
  const abs = Math.abs(n)
  if (abs >= 1_000_000_000_000) return `${(n / 1_000_000_000_000).toFixed(1)}조`
  if (abs >= 100_000_000)       return `${(n / 100_000_000).toFixed(0)}억`
  if (abs >= 10_000)            return `${(n / 10_000).toFixed(0)}만`
  return n.toLocaleString()
}

export function InvestorPanel() {
  const { selectedStock } = useStockStore()
  const [data, setData] = useState<InvestorResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [tab, setTab] = useState<'foreign' | 'institution'>('foreign')

  useEffect(() => {
    if (!selectedStock?.code) return
    const code = selectedStock.code
    let cancelled = false

    async function load() {
      setLoading(true)
      setData(null)
      try {
        const { data: d } = await api.get<InvestorResponse>(`/analysis/investor/${code}`)
        if (!cancelled) setData(d)
      } catch {
        if (!cancelled) setData(null)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => { cancelled = true }
  }, [selectedStock?.code])

  if (!selectedStock) return null

  const trend = data?.trend ?? []
  const summary = data?.summary

  // 차트 데이터: 날짜 short format
  const chartData = trend.map((d) => ({
    date: d.date.slice(5),  // MM-DD
    foreign: Math.round(d.foreign_net / 100_000_000),      // 억원
    institution: Math.round(d.institution_net / 100_000_000),
  }))

  const activeKey = tab === 'foreign' ? 'foreign' : 'institution'
  const activeColor = tab === 'foreign' ? '#60a5fa' : '#a78bfa'
  const activeLabel = tab === 'foreign' ? '외국인' : '기관'
  const activeSummary = tab === 'foreign' ? summary?.foreign_5d : summary?.institution_5d
  const activeNetBuy = tab === 'foreign' ? summary?.foreign_net_buy : summary?.institution_net_buy

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold">투자자 매매동향</h3>
        {loading && <span className="text-[11px] text-muted-foreground">로딩 중…</span>}
      </div>

      {!loading && data && !data.available && (
        <div className="text-xs text-muted-foreground text-center py-3">
          투자자 데이터를 가져올 수 없습니다.
        </div>
      )}

      {!loading && data && data.available && (
        <>
          {/* 탭 */}
          <div className="flex gap-1 mb-2">
            {(['foreign', 'institution'] as const).map((t) => {
              const label = t === 'foreign' ? '외국인' : '기관'
              const isNetBuy = t === 'foreign' ? summary?.foreign_net_buy : summary?.institution_net_buy
              return (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={cn(
                    'flex-1 text-[11px] py-0.5 rounded transition-colors',
                    tab === t
                      ? t === 'foreign'
                        ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                        : 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                      : 'text-muted-foreground hover:text-foreground',
                  )}
                >
                  {label}
                  {isNetBuy !== undefined && (
                    <span className={cn('ml-1', isNetBuy ? 'text-green-400' : 'text-red-400')}>
                      {isNetBuy ? '▲' : '▼'}
                    </span>
                  )}
                </button>
              )
            })}
          </div>

          {/* 5일 순매수 요약 */}
          {activeSummary !== undefined && (
            <div className={cn(
              'text-xs text-center mb-2 py-1 rounded',
              activeNetBuy ? 'text-green-400 bg-green-400/10' : 'text-red-400 bg-red-400/10'
            )}>
              {activeLabel} 5일 순매수: {activeSummary >= 0 ? '+' : ''}{fmt(activeSummary)}
            </div>
          )}

          {/* 바 차트 */}
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.4)' }} />
                <YAxis
                  tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.4)' }}
                  tickFormatter={(v: number) => `${v}억`}
                  width={36}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', fontSize: 10 }}
                  formatter={(v) => [`${Number(v ?? 0)}억원`, activeLabel]}
                />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
                <Bar
                  dataKey={activeKey}
                  fill={activeColor}
                  radius={[2, 2, 0, 0]}
                  // 음수 막대: 빨간색
                  // Recharts doesn't support per-bar colors easily via fill prop,
                  // so we use a fixed color and rely on the reference line
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-xs text-muted-foreground text-center py-3">
              차트 데이터가 없습니다.
            </div>
          )}
        </>
      )}
    </div>
  )
}
