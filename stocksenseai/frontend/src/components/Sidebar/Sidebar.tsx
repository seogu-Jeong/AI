// frontend/src/components/Sidebar/Sidebar.tsx
import { useStockStore } from '@/store/stockStore'
import { StockGroup } from './StockGroup'
import { StockList } from './StockList'

export function Sidebar() {
  const { stockList, watchlist } = useStockStore()
  const watchlistStocks = stockList.filter((s) => watchlist.includes(s.code))

  return (
    <aside className="w-52 flex-1 min-h-0 shrink-0 bg-card border-r border-border overflow-y-auto">
      <StockGroup name="관심종목">
        <StockList stocks={watchlistStocks} />
      </StockGroup>
      <StockGroup name="전체종목">
        <StockList stocks={stockList} />
      </StockGroup>
    </aside>
  )
}
