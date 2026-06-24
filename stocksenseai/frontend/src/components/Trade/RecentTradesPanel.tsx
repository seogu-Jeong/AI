// frontend/src/components/Trade/RecentTradesPanel.tsx
import { useEffect, useState, useCallback } from 'react'
import { RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import api from '@/lib/api'

type TradeStatus = 'PENDING' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED' | 'UNKNOWN'

interface TradeItem {
  id: string
  stock_code: string
  order_type: 'BUY' | 'SELL'
  quantity: number
  order_price: number | null
  executed_price: number | null
  status: TradeStatus
  mode: string
  created_at: string | null
}

const RECENT_TRADES_LIMIT = 10

const STATUS_LABEL: Record<TradeStatus, string> = {
  PENDING: '주문 접수됨, 체결 확인 중',
  PARTIALLY_FILLED: '일부 체결',
  FILLED: '체결 완료',
  CANCELLED: '주문 취소',
  UNKNOWN: '체결 확인 필요',
}

const STATUS_BADGE_CLASS: Record<TradeStatus, string> = {
  PENDING: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30',
  PARTIALLY_FILLED: 'bg-blue-500/20 text-blue-300 border border-blue-500/30',
  FILLED: 'bg-green-500/20 text-green-300 border border-green-500/30',
  CANCELLED: 'bg-muted/60 text-muted-foreground border border-border',
  UNKNOWN: 'bg-red-500/20 text-red-300 border border-red-500/30',
}

function StatusBadge({ status }: { status: TradeStatus }) {
  const label = STATUS_LABEL[status] ?? status
  const cls = STATUS_BADGE_CLASS[status] ?? 'bg-muted/60 text-muted-foreground border border-border'
  return (
    <span className={`inline-block text-xs rounded px-1.5 py-0.5 ${cls}`}>
      {label}
    </span>
  )
}

function formatTime(iso: string | null) {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return iso
  }
}

interface RecentTradesPanelProps {
  /** 새 주문 직후 강제 갱신을 트리거하는 신호 값. 바뀔 때마다 목록을 다시 fetch한다. */
  refreshSignal?: number
}

export function RecentTradesPanel({ refreshSignal }: RecentTradesPanelProps) {
  const [trades, setTrades] = useState<TradeItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchTrades = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await api.get<TradeItem[]>(`/trades?limit=${RECENT_TRADES_LIMIT}`)
      setTrades(data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '주문 목록 조회 실패')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function fetchOnMount() {
      setLoading(true)
      setError(null)
      try {
        const { data } = await api.get<TradeItem[]>(`/trades?limit=${RECENT_TRADES_LIMIT}`)
        if (!cancelled) setTrades(data)
      } catch (e: unknown) {
        const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        if (!cancelled) setError(msg ?? '주문 목록 조회 실패')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void fetchOnMount()
    return () => { cancelled = true }
  }, [refreshSignal])

  return (
    <div className="px-4 py-3 border-t border-border">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground">최근 주문</span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={() => void fetchTrades()}
          disabled={loading}
        >
          <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {error && (
        <div className="text-xs text-destructive bg-destructive/10 rounded p-2 mb-2">{error}</div>
      )}

      {!loading && trades.length === 0 && !error && (
        <div className="text-xs text-muted-foreground text-center py-4">최근 주문 내역이 없습니다</div>
      )}

      {trades.length > 0 && (
        <div className="space-y-2">
          {trades.map((t) => {
            const isBuy = t.order_type === 'BUY'
            const safeStatus: TradeStatus = (
              ['PENDING', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'UNKNOWN'] as TradeStatus[]
            ).includes(t.status as TradeStatus)
              ? (t.status as TradeStatus)
              : 'UNKNOWN'

            const displayPrice = t.executed_price ?? t.order_price
            return (
              <div key={t.id} className="bg-muted/40 rounded-lg p-2.5 space-y-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className={`text-xs font-semibold ${isBuy ? 'text-red-400' : 'text-blue-400'}`}>
                      {isBuy ? '매수' : '매도'}
                    </span>
                    <span className="text-xs font-medium">{t.stock_code}</span>
                    <span className="text-xs text-muted-foreground">{t.quantity}주</span>
                  </div>
                  <span className="text-xs text-muted-foreground">{formatTime(t.created_at)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <StatusBadge status={safeStatus} />
                  {displayPrice != null && (
                    <span className="text-xs text-muted-foreground">
                      {displayPrice.toLocaleString('ko-KR')}원
                    </span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
