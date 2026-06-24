import { useEffect, useState } from 'react'
import { useStockStore } from '@/store/stockStore'
import api from '@/lib/api'

// 백엔드 /analysis/fundamental/{code} 응답
interface FundamentalResponse {
  code: string
  available: boolean
  score: number | null
  max_score?: number
  grade: string | null
  risk: boolean | null
  risk_threshold?: number
  reasons?: string[]
  metrics?: {
    per: number | null
    pbr: number | null
    eps: number | null
    bps: number | null
    dividend_yield: number | null
  }
}

function Metric({ label, value, suffix = '' }: { label: string; value: number | null | undefined; suffix?: string }) {
  return (
    <div className="flex justify-between text-xs py-0.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value != null ? `${value.toLocaleString()}${suffix}` : '-'}</span>
    </div>
  )
}

export function FundamentalPanel() {
  const selectedStock = useStockStore((s) => s.selectedStock)
  const [data, setData] = useState<FundamentalResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!selectedStock?.code) return
    const code = selectedStock.code
    let cancelled = false

    async function load() {
      setLoading(true)
      setData(null)
      try {
        const { data } = await api.get<FundamentalResponse>(`/analysis/fundamental/${code}`)
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

  const score = data?.score
  const isRisk = data?.risk === true
  // 점수 색상: 위험(빨강) / 보통(노랑) / 양호·우수(초록)
  const color = isRisk
    ? 'text-red-400'
    : score != null && score >= 3.0
      ? 'text-green-400'
      : 'text-yellow-400'

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold">재무 평가</h3>
        {data?.grade && (
          <span className={`text-xs px-2 py-0.5 rounded-full border ${isRisk ? 'border-red-400/40 text-red-400' : 'border-border text-muted-foreground'}`}>
            {data.grade}
          </span>
        )}
      </div>

      {loading && <div className="text-xs text-muted-foreground py-3 text-center">재무 분석 중…</div>}

      {!loading && data && !data.available && (
        <div className="text-xs text-muted-foreground py-3 text-center">재무 데이터가 없습니다.</div>
      )}

      {!loading && data?.available && (
        <>
          <div className="flex items-end gap-1 mb-1">
            <span className={`text-3xl font-bold ${color}`}>{score?.toFixed(1)}</span>
            <span className="text-sm text-muted-foreground mb-1">/ {data.max_score?.toFixed(1) ?? '5.0'}</span>
          </div>

          {isRisk && (
            <div className="text-xs font-semibold text-red-400 mb-2">
              ⚠ 위험 — 재무 기준({data.risk_threshold?.toFixed(1)}점) 미만
            </div>
          )}

          <div className="border-t border-border mt-2 pt-2">
            <Metric label="PER" value={data.metrics?.per} suffix="배" />
            <Metric label="PBR" value={data.metrics?.pbr} suffix="배" />
            <Metric label="EPS" value={data.metrics?.eps} suffix="원" />
            <Metric label="BPS" value={data.metrics?.bps} suffix="원" />
          </div>

          {data.reasons && data.reasons.length > 0 && (
            <ul className="mt-2 space-y-0.5">
              {data.reasons.map((r, i) => (
                <li key={i} className="text-[11px] text-muted-foreground">· {r}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  )
}
