import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { useStockStore } from '@/store/stockStore'
import api from '@/lib/api'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { Stock } from '@/types'

// ─── types ────────────────────────────────────────────────────────────────────

interface SingleResult {
  total_return_pct: number
  win_rate_pct: number
  mdd_pct: number
  total_trades: number
  sharpe_ratio: number
}

interface PortfolioItem {
  code: string
  name: string
  weight_pct: number
}

interface PortfolioResult {
  portfolio_metrics: {
    total_return_pct: number
    mdd_pct: number
    sharpe_ratio: number
    win_rate_pct: number
  }
  per_stock: {
    code: string
    name: string
    weight_pct: number
    allocated_cash: number
    total_return_pct: number
    total_trades: number
  }[]
  equity_curve: { date: string; equity: number }[]
  period_start: string
  period_end: string
  initial_cash: number
  stock_count: number
}

// ─── helpers ──────────────────────────────────────────────────────────────────

function MetricCard({
  label, value, positive,
}: { label: string; value: string; positive: boolean | null }) {
  return (
    <div className="bg-background border border-border rounded p-2.5 text-center">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className={cn('text-sm font-bold',
        positive === true ? 'text-green-400'
          : positive === false ? 'text-red-400'
          : 'text-foreground',
      )}>
        {value}
      </div>
    </div>
  )
}

function fmt(n: number, digits = 2) {
  return n.toFixed(digits)
}

// ─── main component ───────────────────────────────────────────────────────────

export function BacktestTab() {
  const { selectedStock, watchlist, stockList, searchStocks } = useStockStore()

  // shared date/cash
  const [startDate, setStartDate] = useState('2025-01-02')
  const [endDate, setEndDate] = useState('2026-01-01')
  const [initialCash, setInitialCash] = useState('10000000')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // mode
  const [mode, setMode] = useState<'single' | 'portfolio'>('single')

  // single mode
  const [singleResult, setSingleResult] = useState<SingleResult | null>(null)

  // portfolio mode
  const [portfolioStocks, setPortfolioStocks] = useState<PortfolioItem[]>([])
  const [portfolioResult, setPortfolioResult] = useState<PortfolioResult | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Stock[]>([])
  const [searchOpen, setSearchOpen] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // debounced search
  const handleSearchInput = (q: string) => {
    setSearchQuery(q)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (!q.trim()) { setSearchResults([]); setSearchOpen(false); return }
    debounceRef.current = setTimeout(async () => {
      const res = await searchStocks(q)
      setSearchResults(res.slice(0, 8))
      setSearchOpen(res.length > 0)
    }, 300)
  }

  const addStock = (s: Stock) => {
    if (portfolioStocks.length >= 10) return
    if (portfolioStocks.find((p) => p.code === s.code)) return
    setPortfolioStocks((prev) => [...prev, { code: s.code, name: s.name, weight_pct: 0 }])
    setSearchQuery('')
    setSearchResults([])
    setSearchOpen(false)
  }

  const loadWatchlist = () => {
    const toAdd = watchlist
      .filter((code) => !portfolioStocks.find((p) => p.code === code))
      .slice(0, 10 - portfolioStocks.length)
    const newStocks = toAdd.map((code) => {
      const found = stockList.find((s) => s.code === code)
      return { code, name: found?.name ?? code, weight_pct: 0 }
    })
    setPortfolioStocks((prev) => [...prev, ...newStocks])
  }

  const removeStock = (code: string) => {
    setPortfolioStocks((prev) => prev.filter((p) => p.code !== code))
  }

  const setWeight = (code: string, val: string) => {
    const n = parseFloat(val)
    setPortfolioStocks((prev) =>
      prev.map((p) => p.code === code ? { ...p, weight_pct: isNaN(n) ? 0 : n } : p)
    )
  }

  const totalWeight = portfolioStocks.reduce((s, p) => s + (p.weight_pct || 0), 0)
  const weightOk = Math.abs(totalWeight - 100) < 0.1

  // ── run single ──────────────────────────────────────────────────────────────

  const handleRunSingle = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedStock?.code) return
    setError(null)
    setLoading(true)
    try {
      const { data } = await api.post('/backtest/run', {
        code: selectedStock.code,
        start_date: startDate,
        end_date: endDate,
        initial_cash: Number(initialCash),
      })
      setSingleResult(data)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '백테스트 실행 중 오류가 발생했습니다')
    } finally {
      setLoading(false)
    }
  }

  // ── run portfolio ────────────────────────────────────────────────────────────

  const handleRunPortfolio = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const { data } = await api.post('/backtest/portfolio-run', {
        stocks: portfolioStocks,
        start_date: startDate,
        end_date: endDate,
        initial_cash: Number(initialCash),
      })
      setPortfolioResult(data)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '포트폴리오 백테스트 실행 중 오류가 발생했습니다')
    } finally {
      setLoading(false)
    }
  }

  // ── render ───────────────────────────────────────────────────────────────────

  const pm = portfolioResult?.portfolio_metrics

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">백테스팅</h2>
        {/* mode toggle */}
        <div className="flex rounded-md border border-border overflow-hidden text-xs">
          {(['single', 'portfolio'] as const).map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); setError(null) }}
              className={cn('px-3 py-1.5 transition-colors',
                mode === m ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {m === 'single' ? '단일 종목' : '포트폴리오'}
            </button>
          ))}
        </div>
      </div>

      {/* ── SINGLE MODE ─────────────────────────────────────────────────────── */}
      {mode === 'single' && (
        <>
          {selectedStock && (
            <div className="text-xs text-muted-foreground">
              종목: <span className="text-foreground font-medium">{selectedStock.name} ({selectedStock.code})</span>
            </div>
          )}
          <form onSubmit={handleRunSingle} className="space-y-3 bg-card border border-border rounded-lg p-4">
            <div>
              <label className="text-xs text-muted-foreground">시작일</label>
              <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="mt-1" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">종료일</label>
              <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="mt-1" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">초기 자본 (원)</label>
              <Input type="number" value={initialCash} onChange={(e) => setInitialCash(e.target.value)} className="mt-1" />
            </div>
            {error && <div className="text-xs text-red-400">{error}</div>}
            <Button type="submit" className="w-full" disabled={loading || !selectedStock}>
              {loading ? '실행 중...' : '백테스트 실행'}
            </Button>
          </form>

          {singleResult && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-sm font-semibold mb-3">결과</div>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="수익률" value={`${singleResult.total_return_pct >= 0 ? '+' : ''}${fmt(singleResult.total_return_pct)}%`} positive={singleResult.total_return_pct >= 0} />
                <MetricCard label="승률" value={`${fmt(singleResult.win_rate_pct, 1)}%`} positive={null} />
                <MetricCard label="MDD" value={`-${fmt(singleResult.mdd_pct, 1)}%`} positive={false} />
                <MetricCard label="거래 횟수" value={`${singleResult.total_trades}회`} positive={null} />
                <MetricCard label="샤프비율" value={fmt(singleResult.sharpe_ratio)} positive={singleResult.sharpe_ratio > 0} />
              </div>
            </div>
          )}
        </>
      )}

      {/* ── PORTFOLIO MODE ──────────────────────────────────────────────────── */}
      {mode === 'portfolio' && (
        <>
          <div className="bg-card border border-border rounded-lg p-4 space-y-3">
            <div className="text-sm font-semibold">종목 구성</div>

            {/* 검색 */}
            <div ref={searchRef} className="relative">
              <Input
                placeholder="종목명 또는 코드 검색 (최대 10종목)"
                value={searchQuery}
                onChange={(e) => handleSearchInput(e.target.value)}
                className="text-xs"
              />
              {searchOpen && searchResults.length > 0 && (
                <div className="absolute z-50 w-full mt-1 bg-card border border-border rounded-md shadow-lg max-h-48 overflow-y-auto">
                  {searchResults.map((s) => (
                    <button
                      key={s.code}
                      type="button"
                      onClick={() => addStock(s)}
                      className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-accent transition-colors"
                    >
                      <span className="text-xs font-medium">{s.name}</span>
                      <span className="text-[11px] text-muted-foreground">{s.code}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* 관심종목 불러오기 */}
            <button
              type="button"
              onClick={loadWatchlist}
              disabled={watchlist.length === 0 || portfolioStocks.length >= 10}
              className="text-xs text-primary border border-primary/30 rounded px-2 py-1 hover:bg-primary/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              관심종목 불러오기 ({watchlist.length}개)
            </button>

            {/* 종목 리스트 */}
            {portfolioStocks.length > 0 && (
              <div className="space-y-1.5">
                {portfolioStocks.map((p) => (
                  <div key={p.code} className="flex items-center gap-2 bg-background rounded border border-border px-2 py-1.5">
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium truncate">{p.name}</div>
                      <div className="text-[10px] text-muted-foreground">{p.code}</div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <input
                        type="number"
                        min={0}
                        max={100}
                        step={1}
                        value={p.weight_pct || ''}
                        onChange={(e) => setWeight(p.code, e.target.value)}
                        placeholder="0"
                        className="w-14 text-xs text-right bg-muted border border-border rounded px-1.5 py-0.5 focus:outline-none focus:border-primary"
                      />
                      <span className="text-xs text-muted-foreground">%</span>
                      <button
                        type="button"
                        onClick={() => removeStock(p.code)}
                        className="text-muted-foreground hover:text-red-400 transition-colors ml-1 text-xs"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))}

                {/* 비중 합계 */}
                <div className={cn(
                  'text-xs text-right font-medium pt-1',
                  weightOk ? 'text-green-400' : 'text-red-400'
                )}>
                  합계: {totalWeight.toFixed(1)}% {weightOk ? '✓' : '(100%가 되어야 합니다)'}
                </div>
              </div>
            )}
          </div>

          {/* 설정 폼 */}
          <form onSubmit={handleRunPortfolio} className="space-y-3 bg-card border border-border rounded-lg p-4">
            <div>
              <label className="text-xs text-muted-foreground">시작일</label>
              <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="mt-1" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">종료일</label>
              <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="mt-1" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">초기 자본 (원)</label>
              <Input type="number" value={initialCash} onChange={(e) => setInitialCash(e.target.value)} className="mt-1" />
            </div>
            {error && <div className="text-xs text-red-400">{error}</div>}
            <Button
              type="submit"
              className="w-full"
              disabled={loading || portfolioStocks.length === 0 || !weightOk}
            >
              {loading ? '시뮬레이션 중...' : '포트폴리오 백테스트 실행'}
            </Button>
          </form>

          {/* 결과 */}
          {portfolioResult && pm && (
            <div className="space-y-4">
              {/* 지표 카드 */}
              <div className="bg-card border border-border rounded-lg p-4">
                <div className="text-sm font-semibold mb-3">
                  포트폴리오 결과 ({portfolioResult.stock_count}종목)
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <MetricCard label="수익률" value={`${pm.total_return_pct >= 0 ? '+' : ''}${fmt(pm.total_return_pct)}%`} positive={pm.total_return_pct >= 0} />
                  <MetricCard label="승률" value={`${fmt(pm.win_rate_pct, 1)}%`} positive={null} />
                  <MetricCard label="MDD" value={`-${fmt(pm.mdd_pct, 1)}%`} positive={false} />
                  <MetricCard label="샤프비율" value={fmt(pm.sharpe_ratio)} positive={pm.sharpe_ratio > 0} />
                </div>
              </div>

              {/* 자산 추이 차트 */}
              {portfolioResult.equity_curve.length > 0 && (
                <div className="bg-card border border-border rounded-lg p-4">
                  <div className="text-sm font-semibold mb-3">자산 추이</div>
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={portfolioResult.equity_curve} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
                        tickFormatter={(v: string) => v.slice(2, 7)}
                        interval="preserveStartEnd"
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
                        tickFormatter={(v: number) => `${(v / 1_000_000).toFixed(0)}M`}
                        width={42}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', fontSize: 11 }}
                        formatter={(v) => [`${Number(v ?? 0).toLocaleString()}원`, '자산']}
                      />
                      <Line type="monotone" dataKey="equity" stroke="#58a6ff" dot={false} strokeWidth={1.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* 종목별 수익률 테이블 */}
              <div className="bg-card border border-border rounded-lg p-4">
                <div className="text-sm font-semibold mb-2">종목별 결과</div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      <th className="text-left py-1.5 font-medium">종목</th>
                      <th className="text-right py-1.5 font-medium w-12">비중</th>
                      <th className="text-right py-1.5 font-medium w-16">수익률</th>
                      <th className="text-right py-1.5 font-medium w-12">거래</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolioResult.per_stock.map((s) => (
                      <tr key={s.code} className="border-b border-border/50">
                        <td className="py-1.5">
                          <div className="font-medium">{s.name}</div>
                          <div className="text-[10px] text-muted-foreground">{s.code}</div>
                        </td>
                        <td className="py-1.5 text-right text-muted-foreground">{s.weight_pct}%</td>
                        <td className={cn('py-1.5 text-right font-semibold',
                          s.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'
                        )}>
                          {s.total_return_pct >= 0 ? '+' : ''}{fmt(s.total_return_pct)}%
                        </td>
                        <td className="py-1.5 text-right text-muted-foreground">{s.total_trades}회</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
