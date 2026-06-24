import { useEffect, useState } from 'react'
import { useStockStore } from '@/store/stockStore'
import api from '@/lib/api'

interface ComprehensiveResponse {
  code: string
  overall_score: number
  overall_grade: string
  recommendation: string
  components: {
    ai: {
      signal: string | null
      signal_score: number | null
      component_score: number
      reason: string
    }
    financial: {
      score: number | null
      grade: string | null
      component_score: number
      reason: string
      risk: boolean | null
    }
    market: {
      trend: string | null
      avg_change: number | null
      component_score: number
      reason: string
    }
  }
  factors: string[]
}

function GradeChip({ grade }: { grade: string }) {
  const cls: Record<string, string> = {
    강력매수: 'bg-green-500/20 text-green-400 border-green-500/40',
    매수:    'bg-green-400/15 text-green-400 border-green-400/30',
    중립:    'bg-yellow-400/15 text-yellow-400 border-yellow-400/30',
    매도주의:'bg-orange-400/15 text-orange-400 border-orange-400/30',
    매도:    'bg-red-500/20 text-red-400 border-red-500/40',
  }
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${cls[grade] ?? 'border-border text-muted-foreground'}`}>
      {grade}
    </span>
  )
}

function ScoreBar({ value, max = 1, label }: { value: number; max?: number; label: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  const color = pct >= 65 ? 'bg-green-500' : pct >= 40 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-14 text-muted-foreground shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 text-right font-medium">{(value * 100).toFixed(0)}%</span>
    </div>
  )
}

export function ComprehensivePanel() {
  const selectedStock = useStockStore((s) => s.selectedStock)
  const [data, setData] = useState<ComprehensiveResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!selectedStock?.code) return
    const code = selectedStock.code
    let cancelled = false

    async function load() {
      setLoading(true)
      setData(null)
      try {
        const { data } = await api.get<ComprehensiveResponse>(`/analysis/comprehensive/${code}`)
        if (!cancelled) setData(data)
      } catch {
        if (!cancelled) setData(null)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => { cancelled = true }
  }, [selectedStock?.code])

  const score = data?.overall_score
  const grade = data?.overall_grade
  const isRisk = data?.components.financial.risk === true
  const scoreColor =
    score == null ? 'text-muted-foreground'
    : score >= 6.5 ? 'text-green-400'
    : score >= 5.0 ? 'text-yellow-400'
    : 'text-red-400'

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold">종합 판단</h3>
        {grade && <GradeChip grade={grade} />}
      </div>

      {loading && (
        <div className="text-xs text-muted-foreground py-3 text-center">분석 중…</div>
      )}

      {!loading && !data && (
        <div className="text-xs text-muted-foreground py-3 text-center">종목을 선택하세요.</div>
      )}

      {!loading && data && (
        <>
          {/* 종합 점수 */}
          <div className="flex items-end gap-1 mb-2">
            <span className={`text-3xl font-bold ${scoreColor}`}>{score?.toFixed(1)}</span>
            <span className="text-sm text-muted-foreground mb-1">/ 10.0</span>
          </div>

          <p className="text-[11px] text-muted-foreground mb-3">{data.recommendation}</p>

          {isRisk && (
            <div className="text-xs font-semibold text-red-400 mb-2">
              ⚠ 재무 위험 종목 — 신중한 판단 필요
            </div>
          )}

          {/* 컴포넌트 바 */}
          <div className="space-y-1.5 mb-3">
            <ScoreBar value={data.components.ai.component_score} label="AI" />
            <ScoreBar value={data.components.financial.component_score} label="재무" />
            <ScoreBar value={data.components.market.component_score} label="시장" />
          </div>

          {/* 근거 목록 */}
          <div className="border-t border-border pt-2 space-y-0.5">
            {data.factors.map((f, i) => (
              <p key={i} className="text-[11px] text-muted-foreground">· {f}</p>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
