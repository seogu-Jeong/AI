import { useState, useEffect } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import api from '@/lib/api'

interface RiskSettingsModalProps {
  open: boolean
  onClose: () => void
}

export function RiskSettingsModal({ open, onClose }: RiskSettingsModalProps) {
  const [maxPerStockPct, setMaxPerStockPct] = useState('20')
  const [dailyLossLimitPct, setDailyLossLimitPct] = useState('5')
  const [stopLossEnabled, setStopLossEnabled] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    api.get('/risk/settings').then(({ data }) => {
      setMaxPerStockPct(String(data.max_per_stock_pct))
      setDailyLossLimitPct(String(data.daily_loss_limit_pct))
      setStopLossEnabled(data.stop_loss_enabled ?? false)
    }).catch((_e) => { if (import.meta.env.DEV) console.warn('[RiskSettings] 설정 로드 실패:', _e) })
  }, [open])

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await api.put('/risk/settings', {
        max_per_stock_pct: Number(maxPerStockPct),
        daily_loss_limit_pct: Number(dailyLossLimitPct),
        stop_loss_enabled: stopLossEnabled,
      })
      onClose()
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '저장 중 오류가 발생했습니다')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>리스크 설정</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSave} className="space-y-3">
          <div>
            <label className="text-xs text-muted-foreground">종목당 최대 투자 비중 (%)</label>
            <Input type="number" value={maxPerStockPct} onChange={(e) => setMaxPerStockPct(e.target.value)} className="mt-1" min="0" max="100" step="0.1" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">일일 최대 손실 한도 (%)</label>
            <Input type="number" value={dailyLossLimitPct} onChange={(e) => setDailyLossLimitPct(e.target.value)} className="mt-1" min="0" max="100" step="0.1" />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="stopLoss"
              checked={stopLossEnabled}
              onChange={(e) => setStopLossEnabled(e.target.checked)}
              className="w-4 h-4"
            />
            <label htmlFor="stopLoss" className="text-xs text-muted-foreground cursor-pointer">자동 손절 활성화</label>
          </div>
          {error && <div className="text-xs text-red-400">{error}</div>}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? '저장 중...' : '저장'}
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  )
}
