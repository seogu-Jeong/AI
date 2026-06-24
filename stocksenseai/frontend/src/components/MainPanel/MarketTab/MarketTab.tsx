import { useEffect, useState } from 'react'
import { useStockStore } from '@/store/stockStore'
import { cn } from '@/lib/utils'
import api from '@/lib/api'
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts'

// ─── types ────────────────────────────────────────────────────────────────────

interface IndexInfo {
  name: string
  value: number | null
  change_rate: number | null
}

interface TopStock {
  code: string
  name: string
  change_pct: number
  mktcap: number
}

interface Sector {
  sector: string
  total_mktcap: number
  change_pct: number
  up_count: number
  down_count: number
  flat_count: number
  total_stocks: number
  top_stocks: TopStock[]
}

interface SectorResponse {
  date: string
  available: boolean
  sectors: Sector[]
}

interface IndicesResponse {
  indices: IndexInfo[]
  trend: string
  avg_change: number
}

// ─── helpers ──────────────────────────────────────────────────────────────────

function changePctColor(pct: number): string {
  if (pct >= 2)   return '#16a34a'
  if (pct >= 1)   return '#22c55e'
  if (pct >= 0.3) return '#4ade80'
  if (pct >= 0)   return '#86efac'
  if (pct >= -0.3) return '#fca5a5'
  if (pct >= -1)  return '#f87171'
  if (pct >= -2)  return '#ef4444'
  return '#dc2626'
}

function fmtMktcap(n: number): string {
  if (n >= 1_000_000_000_000) return `${(n / 1_000_000_000_000).toFixed(0)}조`
  if (n >= 100_000_000)       return `${(n / 100_000_000).toFixed(0)}억`
  return n.toLocaleString()
}

// ─── Treemap custom content ───────────────────────────────────────────────────

interface TreemapContentProps {
  x?: number
  y?: number
  width?: number
  height?: number
  name?: string
  change_pct?: number
  depth?: number
}

function SectorBlock(props: TreemapContentProps) {
  const { x = 0, y = 0, width = 0, height = 0, name = '', change_pct = 0 } = props
  if (width < 20 || height < 20) return null
  const bg = changePctColor(change_pct)
  const sign = change_pct >= 0 ? '+' : ''
  const showLabel = width > 50 && height > 30

  return (
    <g>
      <rect
        x={x + 1} y={y + 1}
        width={width - 2} height={height - 2}
        fill={bg}
        fillOpacity={0.85}
        rx={3}
      />
      {showLabel && (
        <>
          <text
            x={x + width / 2} y={y + height / 2 - 6}
            textAnchor="middle"
            fill="#fff"
            fontSize={Math.min(12, width / 6)}
            fontWeight="600"
          >
            {name}
          </text>
          <text
            x={x + width / 2} y={y + height / 2 + 10}
            textAnchor="middle"
            fill="rgba(255,255,255,0.9)"
            fontSize={Math.min(11, width / 7)}
          >
            {sign}{change_pct.toFixed(2)}%
          </text>
        </>
      )}
    </g>
  )
}

// ─── main component ───────────────────────────────────────────────────────────

export function MarketTab() {
  const { setSelectedStock } = useStockStore()
  const [indices, setIndices] = useState<IndexInfo[]>([])
  const [sectors, setSectors] = useState<Sector[]>([])
  const [selectedSector, setSelectedSector] = useState<Sector | null>(null)
  const [loading, setLoading] = useState(false)
  const [trend, setTrend] = useState<string>('neutral')

  useEffect(() => {
    let cancelled = false

    async function load() {
      setLoading(true)
      try {
        const [idxRes, secRes] = await Promise.all([
          api.get<IndicesResponse>('/analysis/indices'),
          api.get<SectorResponse>('/analysis/sector'),
        ])
        if (!cancelled) {
          setIndices(idxRes.data.indices ?? [])
          setTrend(idxRes.data.trend ?? 'neutral')
          if (secRes.data.available) {
            setSectors(secRes.data.sectors)
            setSelectedSector(secRes.data.sectors[0] ?? null)
          }
        }
      } catch (_e) {
        if (import.meta.env.DEV) console.warn('[MarketTab] 시장 데이터 fetch 실패:', _e)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => { cancelled = true }
  }, [])

  // total_stocks를 크기로 사용 (시총 대체)
  const treemapData = sectors.map((s) => ({
    name: s.sector,
    value: Math.max(s.total_stocks || s.total_mktcap || 1, 1),
    change_pct: s.change_pct,
  }))

  const trendColor = trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-yellow-400'
  const trendLabel = trend === 'up' ? '상승' : trend === 'down' ? '하락' : '중립'

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">시장 현황</h2>
        {!loading && trend && (
          <span className={cn('text-xs font-medium px-2 py-0.5 rounded border', trendColor,
            trend === 'up' ? 'border-green-400/30 bg-green-400/10'
              : trend === 'down' ? 'border-red-400/30 bg-red-400/10'
              : 'border-yellow-400/30 bg-yellow-400/10'
          )}>
            시장 {trendLabel}
          </span>
        )}
      </div>

      {loading && (
        <div className="text-xs text-muted-foreground text-center py-8">
          시장 데이터 로딩 중…
        </div>
      )}

      {!loading && (
        <>
          {/* 지수 요약 */}
          <div className="grid grid-cols-3 gap-2">
            {indices.map((idx) => {
              const chg = idx.change_rate ?? 0
              const pos = chg >= 0
              return (
                <div key={idx.name} className="bg-card border border-border rounded-lg p-3 text-center">
                  <div className="text-[11px] text-muted-foreground mb-0.5">{idx.name}</div>
                  <div className="text-sm font-bold">{idx.value?.toLocaleString() ?? '-'}</div>
                  <div className={cn('text-xs font-medium', pos ? 'text-green-400' : 'text-red-400')}>
                    {pos ? '+' : ''}{chg.toFixed(2)}%
                  </div>
                </div>
              )
            })}
          </div>

          {/* 섹터 히트맵 */}
          {sectors.length > 0 ? (
            <div className="bg-card border border-border rounded-lg p-3">
              <div className="text-sm font-semibold mb-2">업종 히트맵</div>
              <div className="text-[11px] text-muted-foreground mb-2">
                크기 = 종목수 · 색상 = 등락률 · 클릭하면 업종 선택
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <Treemap
                  data={treemapData}
                  dataKey="value"
                  content={<SectorBlock />}
                  onClick={(node) => {
                    const found = sectors.find((s) => s.sector === node.name)
                    if (found) setSelectedSector(found)
                  }}
                >
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const d = payload[0].payload
                      return (
                        <div className="bg-card border border-border rounded px-2 py-1.5 text-xs shadow-lg">
                          <div className="font-semibold">{d.name}</div>
                          <div className={d.change_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {d.change_pct >= 0 ? '+' : ''}{d.change_pct?.toFixed(2)}%
                          </div>
                          <div className="text-muted-foreground">{fmtMktcap(d.value)}</div>
                        </div>
                      )
                    }}
                  />
                </Treemap>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="bg-card border border-border rounded-lg p-4 text-center text-xs text-muted-foreground">
              섹터 데이터를 가져올 수 없습니다. (pykrx 연결 확인 필요)
            </div>
          )}

          {/* 선택된 업종 정보 */}
          {selectedSector && (
            <div className="bg-card border border-border rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-semibold">{selectedSector.sector}</div>
                <div className={cn(
                  'text-sm font-bold px-2 py-0.5 rounded',
                  selectedSector.change_pct >= 0
                    ? 'text-green-400 bg-green-400/10'
                    : 'text-red-400 bg-red-400/10'
                )}>
                  {selectedSector.change_pct >= 0 ? '+' : ''}{selectedSector.change_pct.toFixed(2)}%
                </div>
              </div>
              {(selectedSector.total_stocks || selectedSector.total_mktcap) > 0 && (
                <div className="flex gap-3 text-[11px] text-muted-foreground mt-1">
                  <span>전체 {selectedSector.total_stocks ?? selectedSector.total_mktcap}종목</span>
                  <span className="text-green-400">▲ {selectedSector.up_count}</span>
                  <span className="text-muted-foreground">— {selectedSector.flat_count}</span>
                  <span className="text-red-400">▼ {selectedSector.down_count}</span>
                </div>
              )}
              {/* top_stocks 는 KRX 로그인 없이 제공 불가 — 히트맵 클릭 시 업종명만 표시 */}
              {selectedSector.top_stocks.length > 0 ? (
                <div className="mt-2 space-y-1">
                  {selectedSector.top_stocks.map((s) => (
                    <button
                      key={s.code}
                      onClick={() => setSelectedStock({ code: s.code, name: s.name })}
                      className="w-full flex items-center justify-between px-2 py-1.5 rounded hover:bg-accent transition-colors text-left"
                    >
                      <div>
                        <div className="text-xs font-medium">{s.name}</div>
                        <div className="text-[10px] text-muted-foreground">{s.code}</div>
                      </div>
                      <div className="text-right">
                        <div className={cn('text-xs font-semibold',
                          s.change_pct >= 0 ? 'text-green-400' : 'text-red-400'
                        )}>
                          {s.change_pct >= 0 ? '+' : ''}{s.change_pct.toFixed(2)}%
                        </div>
                        <div className="text-[10px] text-muted-foreground">{fmtMktcap(s.mktcap)}</div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="mt-2 text-[11px] text-muted-foreground">
                  히트맵에서 업종을 클릭하면 등락률을 확인할 수 있습니다.
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
