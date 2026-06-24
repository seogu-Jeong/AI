import { useState } from 'react'
import type { Stock, StockDetail } from '@/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { OrderModal } from '@/components/Trade/OrderModal'

interface StockInfoBarProps {
  stock: Stock
  detail: StockDetail | null
  isLive: boolean
  realtimePrice?: number | null
  realtimeChangePct?: number | null
}

export function StockInfoBar({ stock, detail, isLive, realtimePrice, realtimeChangePct }: StockInfoBarProps) {
  const price = realtimePrice ?? stock.price ?? 0
  const changePct = realtimeChangePct ?? stock.change_pct ?? 0
  const isPositive = changePct >= 0

  const [orderType, setOrderType] = useState<'BUY' | 'SELL' | null>(null)

  return (
    <div className="bg-card border-b border-border px-4 py-2 shrink-0">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-bold text-base">{stock.name}</span>
          <span className="text-muted-foreground text-sm">{stock.code}</span>
          {isLive && (
            <span className="text-xs bg-green-500/20 text-green-500 px-1.5 py-0.5 rounded">
              실시간
            </span>
          )}
        </div>
        <div className="text-right">
          <span className={cn('font-bold text-xl', isPositive ? 'text-green-500' : 'text-red-500')}>
            {price.toLocaleString('ko-KR')}
          </span>
          <span className={cn('ml-2 text-sm', isPositive ? 'text-green-500' : 'text-red-500')}>
            {isPositive ? '▲' : '▼'}{Math.abs(changePct).toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2">
        {[
          { label: '시가', value: detail?.open, color: 'text-foreground' },
          { label: '고가', value: detail?.high, color: 'text-green-500' },
          { label: '저가', value: detail?.low, color: 'text-red-500' },
          { label: '거래량', value: detail?.volume, color: 'text-foreground', isVolume: true },
        ].map(({ label, value, color, isVolume }) => (
          <div key={label} className="bg-background rounded px-2 py-1 text-center">
            <div className="text-muted-foreground text-xs mb-0.5">{label}</div>
            <div className={cn('text-xs font-semibold', color)}>
              {value == null ? '-' : isVolume ? `${(value / 1000000).toFixed(1)}M` : value.toLocaleString('ko-KR')}
            </div>
          </div>
        ))}
      </div>

      {/* 매수/매도 버튼 */}
      <div className="flex gap-2 mt-2">
        <Button
          size="sm"
          className="flex-1 bg-green-500 hover:bg-green-600 text-white"
          onClick={() => setOrderType('BUY')}
        >
          매수
        </Button>
        <Button
          size="sm"
          className="flex-1 bg-red-500 hover:bg-red-600 text-white"
          onClick={() => setOrderType('SELL')}
        >
          매도
        </Button>
      </div>
      {orderType && (
        <OrderModal
          open={true}
          onClose={() => setOrderType(null)}
          stock={stock}
          orderType={orderType}
        />
      )}
    </div>
  )
}
