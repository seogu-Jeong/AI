import type { OrderBookEntry } from '@/types'

interface OrderBookProps {
  asks: OrderBookEntry[]
  bids: OrderBookEntry[]
  currentPrice: number
}

export function OrderBook({ asks, bids, currentPrice }: OrderBookProps) {
  const maxQty = Math.max(...asks.map((a) => a.quantity), ...bids.map((b) => b.quantity))

  return (
    <div className="flex flex-col h-full text-xs overflow-hidden">
      <div className="px-2 py-1.5 border-b border-border font-semibold text-xs">호가</div>

      {/* 매도호가 - 위에서 아래로 (높은가격→낮은가격) */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="text-center text-red-400 text-xs py-0.5 bg-red-400/5">매도</div>
        <div className="flex-1 overflow-y-auto">
          {[...asks].reverse().map((ask, i) => (
            <div key={i} className="relative flex items-center justify-between px-2 py-0.5 hover:bg-accent/50">
              <div
                className="absolute right-0 top-0 bottom-0 bg-red-400/10"
                style={{ width: `${(ask.quantity / maxQty) * 100}%` }}
              />
              <span className="text-red-400 font-medium relative z-10">{ask.price.toLocaleString('ko-KR')}</span>
              <span className="text-muted-foreground relative z-10">{ask.quantity.toLocaleString('ko-KR')}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 현재가 */}
      <div className="text-center py-1 border-y border-border bg-card font-bold text-sm">
        {currentPrice.toLocaleString('ko-KR')}
      </div>

      {/* 매수호가 */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto">
          {bids.map((bid, i) => (
            <div key={i} className="relative flex items-center justify-between px-2 py-0.5 hover:bg-accent/50">
              <div
                className="absolute right-0 top-0 bottom-0 bg-green-400/10"
                style={{ width: `${(bid.quantity / maxQty) * 100}%` }}
              />
              <span className="text-green-400 font-medium relative z-10">{bid.price.toLocaleString('ko-KR')}</span>
              <span className="text-muted-foreground relative z-10">{bid.quantity.toLocaleString('ko-KR')}</span>
            </div>
          ))}
        </div>
        <div className="text-center text-green-400 text-xs py-0.5 bg-green-400/5">매수</div>
      </div>
    </div>
  )
}
