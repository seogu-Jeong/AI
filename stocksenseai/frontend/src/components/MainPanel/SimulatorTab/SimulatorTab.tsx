import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { useStockStore } from '@/store/stockStore'
import api from '@/lib/api'
import type { LumpsumResult } from '@/types'

export function SimulatorTab() {
  const selectedStock = useStockStore((s) => s.selectedStock)
  const [buyDate, setBuyDate] = useState('2025-01-02')
  const [sellDate, setSellDate] = useState('2026-01-01')
  const [amount, setAmount] = useState('1000000')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<LumpsumResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSimulate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedStock?.code) return
    setError(null)
    setLoading(true)
    try {
      const { data } = await api.post('/simulate/lumpsum', {
        tickers: [selectedStock.code],
        buy_date: buyDate,
        sell_date: sellDate,
        amount_krw: Number(amount),
      })
      const results: LumpsumResult[] = data.results ?? data ?? []
      if (results.length > 0) setResult(results[0])
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '시뮬레이션 중 오류가 발생했습니다')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      <h2 className="text-base font-semibold mb-4">투자 시뮬레이터</h2>

      <form onSubmit={handleSimulate} className="space-y-3 bg-card border border-border rounded-lg p-4">
        <div>
          <label className="text-xs text-muted-foreground">종목</label>
          <div className="mt-1 text-sm font-medium">
            {selectedStock ? `${selectedStock.name} (${selectedStock.code})` : '종목을 선택해주세요'}
          </div>
        </div>
        <div>
          <label className="text-xs text-muted-foreground">매수일</label>
          <Input type="date" value={buyDate} onChange={(e) => setBuyDate(e.target.value)} className="mt-1" />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">매도일</label>
          <Input type="date" value={sellDate} onChange={(e) => setSellDate(e.target.value)} className="mt-1" />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">투자금액</label>
          <Input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} className="mt-1" />
        </div>
        {error && <div className="text-xs text-red-400">{error}</div>}
        <Button type="submit" className="w-full" disabled={loading || !selectedStock}>
          {loading ? '계산 중...' : '시뮬레이션 실행'}
        </Button>
      </form>

      {result && (
        <div className="mt-4 bg-card border border-border rounded-lg p-4 space-y-2">
          <div className="text-sm font-semibold">{result.name} 시뮬레이션 결과</div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">매수가</span>
            <span className="font-medium">{result.buy_price.toLocaleString('ko-KR')}원</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">매도가</span>
            <span className="font-medium">{result.sell_price.toLocaleString('ko-KR')}원</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">수익/손실</span>
            <span className={cn('font-medium', result.profit_krw >= 0 ? 'text-green-400' : 'text-red-400')}>
              {result.profit_krw >= 0 ? '+' : ''}{result.profit_krw.toLocaleString('ko-KR')}원
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">수익률</span>
            <span className={cn('font-medium', result.return_pct >= 0 ? 'text-green-400' : 'text-red-400')}>
              {result.return_pct >= 0 ? '+' : ''}{result.return_pct.toFixed(2)}%
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
