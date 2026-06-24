// frontend/src/components/WatchlistPanel/WatchlistPanel.tsx
import { useStockStore } from '@/store/stockStore'
import { cn } from '@/lib/utils'

export function WatchlistPanel() {
  const { stockList, watchlist, selectedStock, setSelectedStock, addToWatchlist, removeFromWatchlist } = useStockStore()
  const stocks = stockList.filter((s) => watchlist.includes(s.code))

  const canAdd = selectedStock && !watchlist.includes(selectedStock.code)

  return (
    <div className="h-8 flex items-center gap-1 px-2 bg-card border-t border-border overflow-x-auto shrink-0">
      {/* + 버튼: 현재 선택 종목을 관심목록에 추가 */}
      <button
        onClick={() => selectedStock && addToWatchlist(selectedStock.code)}
        disabled={!canAdd}
        title={canAdd ? `${selectedStock?.name} 관심목록 추가` : '이미 추가됐거나 종목 미선택'}
        className={cn(
          'shrink-0 w-6 h-6 flex items-center justify-center rounded text-xs border transition-colors',
          canAdd
            ? 'border-primary/50 text-primary hover:bg-primary/10'
            : 'border-border text-muted-foreground/40 cursor-default'
        )}
      >
        +
      </button>

      <div className="w-px h-4 bg-border shrink-0 mx-1" />

      {stocks.length === 0 && (
        <span className="text-xs text-muted-foreground">관심 종목이 없습니다. + 버튼으로 추가하세요.</span>
      )}

      {stocks.map((stock) => (
        <div key={stock.code} className="flex items-center gap-1 group shrink-0">
          <button
            onClick={() => setSelectedStock(stock)}
            className={cn(
              'flex items-center gap-1.5 text-xs whitespace-nowrap hover:text-foreground py-0.5',
              selectedStock?.code === stock.code ? 'text-foreground' : 'text-muted-foreground'
            )}
          >
            <span className="font-medium">{stock.name}</span>
            {stock.price && (
              <span>{stock.price.toLocaleString('ko-KR')}</span>
            )}
            {stock.change_pct !== undefined && (
              <span className={stock.change_pct >= 0 ? 'text-green-500' : 'text-red-500'}>
                {stock.change_pct >= 0 ? '▲' : '▼'}{Math.abs(stock.change_pct).toFixed(1)}%
              </span>
            )}
          </button>
          {/* × 삭제 버튼 — hover 시 표시 */}
          <button
            onClick={() => removeFromWatchlist(stock.code)}
            title="관심목록에서 제거"
            className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-red-400 text-[10px] w-3.5 h-3.5 flex items-center justify-center rounded transition-all"
          >
            ×
          </button>
          <div className="w-px h-3 bg-border mx-1" />
        </div>
      ))}
    </div>
  )
}
