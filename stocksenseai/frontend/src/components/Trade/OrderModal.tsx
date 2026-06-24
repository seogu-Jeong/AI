import { useEffect, useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { Stock } from '@/types'
import { cn } from '@/lib/utils'
import api from '@/lib/api'
import { useUIStore } from '@/store/uiStore'

interface OrderModalProps {
  open: boolean
  onClose: () => void
  stock: Stock
  orderType: 'BUY' | 'SELL'
}

export function OrderModal({ open, onClose, stock, orderType }: OrderModalProps) {
  const [priceType, setPriceType] = useState<'MARKET' | 'LIMIT'>('LIMIT')
  const [quantity, setQuantity] = useState('1')
  const [price, setPrice] = useState(String(stock.price ?? 0))
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [account, setAccount] = useState<{ mode: 'paper' | 'real'; account_no: string } | null>(null)
  const [confirmingReal, setConfirmingReal] = useState(false)
  const notifyOrderPlaced = useUIStore((s) => s.notifyOrderPlaced)

  const isBuy = orderType === 'BUY'
  const total = Number(quantity) * Number(price)

  useEffect(() => {
    if (!open) return
    api.get('/account/config')
      .then(({ data }) => setAccount(data))
      .catch(() => setAccount(null))
  }, [open])

  const closeModal = () => {
    setConfirmingReal(false)
    setAccount(null)
    onClose()
  }

  const submitOrder = async () => {
    setError(null)
    setSubmitting(true)
    setSubmitted(false)
    try {
      await api.post('/trades/order', {
        stock_code: stock.code,
        order_type: orderType,
        price_type: priceType,
        quantity: Number(quantity),
        price: priceType === 'LIMIT' ? Number(price) : undefined,
      })
      notifyOrderPlaced()
      setSubmitted(true)
      setTimeout(() => { setSubmitted(false); closeModal() }, 1500)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '주문 처리 중 오류가 발생했습니다')
    } finally {
      setSubmitting(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (account?.mode === 'real' && !confirmingReal) {
      setConfirmingReal(true)
      return
    }
    await submitOrder()
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && closeModal()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle className={cn(isBuy ? 'text-green-400' : 'text-red-400')}>
            <span>{stock.name}</span>
            {' '}
            <span>{isBuy ? '매수' : '매도'}</span>
          </DialogTitle>
        </DialogHeader>

        {submitted ? (
          <div className="text-center py-6 space-y-2">
            <div className={cn('text-xl font-bold', isBuy ? 'text-green-400' : 'text-red-400')}>
              {isBuy ? '매수 접수' : '매도 접수'}
            </div>
            <div className="text-sm font-medium text-foreground">주문 접수됨, 체결 확인 중</div>
            <div className="text-xs text-muted-foreground">
              계좌 패널의 최근 주문에서 체결 상태를 확인하세요.
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-3">
            {account && (
              <div className={`text-xs rounded p-2 ${
                account.mode === 'real'
                  ? 'text-red-300 bg-red-500/10 border border-red-500/20'
                  : 'text-blue-300 bg-blue-500/10 border border-blue-500/20'
              }`}>
                {account.mode === 'real' ? '실계좌' : '모의투자'} · {account.account_no}
              </div>
            )}
            {confirmingReal && (
              <div className="text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded p-3">
                실계좌 주문을 실행하시겠습니까?
              </div>
            )}
            <div className="flex gap-2">
              {(['LIMIT', 'MARKET'] as const).map((type) => (
                <Button
                  key={type}
                  type="button"
                  variant={priceType === type ? 'default' : 'outline'}
                  size="sm"
                  className="flex-1"
                  onClick={() => setPriceType(type)}
                >
                  {type === 'LIMIT' ? '지정가' : '시장가'}
                </Button>
              ))}
            </div>

            {priceType === 'LIMIT' && (
              <div>
                <label className="text-xs text-muted-foreground">주문가격</label>
                <Input
                  type="number"
                  disabled={submitting}
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  className="mt-1"
                />
              </div>
            )}

            <div>
              <label className="text-xs text-muted-foreground">수량</label>
              <Input
                type="number"
                min="1"
                disabled={submitting}
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                className="mt-1"
              />
            </div>

            {priceType === 'LIMIT' && (
              <div className="text-sm text-right text-muted-foreground">
                주문금액: <span className="text-foreground font-medium">{total.toLocaleString('ko-KR')}원</span>
              </div>
            )}

            {error && (
              <div className="text-xs text-red-400">{error}</div>
            )}

            <div className="flex gap-2 pt-1">
              <Button type="button" variant="outline" className="flex-1" onClick={closeModal} disabled={submitting}>취소</Button>
              <Button
                type="submit"
                disabled={submitting}
                className={cn('flex-1', isBuy ? 'bg-green-500 hover:bg-green-600' : 'bg-red-500 hover:bg-red-600')}
              >
                {submitting ? '주문 요청 중...' : confirmingReal ? '실계좌 주문 확인' : '주문'}
              </Button>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}
