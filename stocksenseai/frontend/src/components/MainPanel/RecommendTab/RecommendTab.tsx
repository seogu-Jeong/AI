import { useEffect, useState, useCallback } from 'react'
import { useStockStore } from '@/store/stockStore'
import { ComprehensivePanel } from '@/components/Analysis/ComprehensivePanel'
import { FundamentalPanel } from '@/components/Analysis/FundamentalPanel'
import api from '@/lib/api'
import { AlertTriangle } from 'lucide-react'
import { buildReasonChips, type ReasonChip } from '@/lib/recommendReasons'

interface RankItem {
  code: string
  name: string
  signal: 'BUY' | 'HOLD' | 'SELL'
  signal_score: number
  tech_score: number | null
  lstm_score: number | null
  lstm_available: boolean
}

interface RankingResponse {
  ranking: RankItem[]
  scanned: number
  total: number
}

const SIGNAL_STYLE: Record<string, string> = {
  BUY:  'text-green-400 bg-green-400/10 border border-green-400/30',
  SELL: 'text-red-400 bg-red-400/10 border border-red-400/30',
  HOLD: 'text-yellow-400 bg-yellow-400/10 border border-yellow-400/30',
}

const CHIP_COLOR: Record<ReasonChip['color'], string> = {
  green: 'text-green-400 bg-green-400/10 border border-green-400/30',
  yellow: 'text-yellow-400 bg-yellow-400/10 border border-yellow-400/30',
  red: 'text-red-400 bg-red-400/10 border border-red-400/30',
  gray: 'text-muted-foreground bg-muted border border-border',
  blue: 'text-blue-400 bg-blue-400/10 border border-blue-400/30',
}

function ReasonChips({ item }: { item: RankItem }) {
  const chips = buildReasonChips(item)
  return (
    <div className="flex flex-wrap gap-1 mt-1">
      {chips.map((chip) => (
        <span
          key={chip.label}
          className={`px-1 py-0.5 rounded text-[10px] font-medium ${CHIP_COLOR[chip.color]}`}
        >
          {chip.label}
        </span>
      ))}
    </div>
  )
}

function ScoreBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value))
  const color = pct >= 65 ? 'bg-green-500' : pct >= 35 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-medium w-7 text-right">{value.toFixed(0)}</span>
    </div>
  )
}

export function RecommendTab() {
  const { selectedStock, setSelectedStock } = useStockStore()
  const [data, setData] = useState<RankingResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [focusedCode, setFocusedCode] = useState<string | null>(null)

  const fetchRanking = useCallback(async () => {
    await Promise.resolve()
    setLoading(true)
    setFetchError(null)
    try {
      const { data: res } = await api.get<RankingResponse>('/analysis/ai-ranking?limit=50')
      setData(res)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setFetchError(msg ?? 'AI 추천 데이터를 불러오지 못했습니다.')
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function fetchOnMount() {
      setLoading(true)
      setFetchError(null)
      try {
        const { data: res } = await api.get<RankingResponse>('/analysis/ai-ranking?limit=50')
        if (!cancelled) setData(res)
      } catch (e: unknown) {
        const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        if (!cancelled) {
          setFetchError(msg ?? 'AI 추천 데이터를 불러오지 못했습니다.')
          setData(null)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void fetchOnMount()
    return () => { cancelled = true }
  }, [])

  const handleSelect = (item: RankItem) => {
    setFocusedCode(item.code)
    setSelectedStock({ code: item.code, name: item.name })
  }

  return (
    <div className="h-full overflow-hidden flex flex-col">
      {/* 헤더 */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <div>
          <span className="text-sm font-semibold">AI 추천 종목 순위</span>
          {data && (
            <span className="ml-2 text-xs text-muted-foreground">
              {data.scanned}종목 스캔 · 총 {data.total}개
            </span>
          )}
        </div>
        <button
          onClick={fetchRanking}
          disabled={loading}
          className="text-xs px-3 py-1 rounded border border-border text-muted-foreground hover:text-foreground hover:border-foreground/40 transition-colors disabled:opacity-50"
        >
          {loading ? '스캔 중…' : '새로고침'}
        </button>
      </div>

      {/* 본문 — 좌: 순위 / 우: 선택 종목 상세 */}
      <div className="flex-1 min-h-0 overflow-hidden flex gap-0">
        {/* 순위 테이블 */}
        <div className="flex-1 min-w-0 overflow-y-auto">
          {loading && (
            <div className="text-xs text-muted-foreground text-center py-8">
              AI 점수 분석 중… (최대 1분 소요)
              <div className="text-[10px] mt-1 text-muted-foreground/70">top100 종목 전체를 스캔합니다</div>
            </div>
          )}
          {!loading && fetchError && (
            <div className="flex flex-col items-center gap-2 py-8 px-4 text-center">
              <AlertTriangle className="h-4 w-4 text-yellow-400" />
              <span className="text-xs text-destructive">추천 API 조회 실패</span>
              <span className="text-[10px] text-muted-foreground">{fetchError}</span>
              <span className="text-[10px] text-muted-foreground">서버 상태를 확인하거나 잠시 후 새로고침해 주세요.</span>
            </div>
          )}
          {!loading && !fetchError && data && data.ranking.length === 0 && (
            <div className="text-xs text-muted-foreground text-center py-8">
              추천 결과가 없습니다. 시장 데이터를 다시 스캔해 주세요.
            </div>
          )}
          {!loading && !fetchError && data && data.ranking.length > 0 && (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card border-b border-border">
                <tr>
                  <th className="px-3 py-2 text-left text-muted-foreground font-medium w-8">#</th>
                  <th className="px-3 py-2 text-left text-muted-foreground font-medium">종목</th>
                  <th className="px-3 py-2 text-center text-muted-foreground font-medium w-16">신호</th>
                  <th className="px-3 py-2 text-left text-muted-foreground font-medium">AI 점수</th>
                  <th className="px-3 py-2 text-right text-muted-foreground font-medium w-12">기술</th>
                  <th className="px-3 py-2 text-right text-muted-foreground font-medium w-12">LSTM</th>
                </tr>
              </thead>
              <tbody>
                {data.ranking.map((item, idx) => {
                  const isSelected = focusedCode === item.code || selectedStock?.code === item.code
                  return (
                    <tr
                      key={item.code}
                      onClick={() => handleSelect(item)}
                      className={`border-b border-border/50 cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-primary/10'
                          : 'hover:bg-accent/50'
                      }`}
                    >
                      <td className="px-3 py-2 text-muted-foreground">{idx + 1}</td>
                      <td className="px-3 py-2">
                        <div className="font-medium">{item.name}</div>
                        <div className="text-muted-foreground text-[10px]">{item.code}</div>
                        <ReasonChips item={item} />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`px-1.5 py-0.5 rounded text-[11px] font-semibold ${SIGNAL_STYLE[item.signal] ?? ''}`}>
                          {item.signal}
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        <ScoreBar value={item.signal_score} />
                      </td>
                      <td className="px-3 py-2 text-right text-muted-foreground">
                        {item.tech_score != null ? item.tech_score.toFixed(0) : '-'}
                      </td>
                      <td className="px-3 py-2 text-right text-muted-foreground">
                        {item.lstm_available && item.lstm_score != null ? item.lstm_score.toFixed(0) : '-'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* 우측 상세 패널 — 종합 판단 + 재무 평가 */}
        <div className="w-72 shrink-0 border-l border-border overflow-y-auto p-3 space-y-3">
          {!selectedStock ? (
            <p className="text-xs text-muted-foreground text-center mt-8">
              왼쪽 종목을 클릭하면<br />종합 판단과 재무 평가가 표시됩니다.
            </p>
          ) : (
            <>
              <div className="text-xs font-semibold text-muted-foreground border-b border-border pb-1 mb-1">
                {selectedStock.name} ({selectedStock.code})
              </div>
              <ComprehensivePanel />
              <FundamentalPanel />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
