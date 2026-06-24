// frontend/src/components/Sidebar/StockList.tsx
import type { Stock } from '@/types'
import { useStockStore } from '@/store/stockStore'
import { cn } from '@/lib/utils'

interface StockListProps {
  stocks: Stock[]
}

export function StockList({ stocks }: StockListProps) {
  const { selectedStock, setSelectedStock } = useStockStore()

  return (
    <ul>
      {stocks.map((stock) => (
        <li
          key={stock.code}
          onClick={() => setSelectedStock(stock)}
          className={cn(
            'flex items-center justify-between px-4 py-1.5 cursor-pointer text-sm hover:bg-accent',
            selectedStock?.code === stock.code && 'bg-accent text-accent-foreground'
          )}
        >
          <span className="truncate">{stock.name}</span>
          {stock.price && (
            <div className="text-right ml-2 shrink-0">
              <div className="text-xs font-medium">{stock.price.toLocaleString('ko-KR')}</div>
              {stock.change_pct !== undefined && (
                <div
                  className={cn(
                    'text-xs',
                    stock.change_pct >= 0 ? 'text-green-500' : 'text-red-500'
                  )}
                >
                  {stock.change_pct >= 0 ? '+' : ''}{stock.change_pct.toFixed(1)}%
                </div>
              )}
            </div>
          )}
        </li>
      ))}
    </ul>
  )
}
