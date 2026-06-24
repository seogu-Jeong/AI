// frontend/src/components/Account/AccountPanel.tsx
import { useEffect, useState, useCallback } from 'react'
import { X, RefreshCw, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAuthStore } from '@/store/authStore'
import { useUIStore } from '@/store/uiStore'
import api from '@/lib/api'
import { RecentTradesPanel } from '@/components/Trade/RecentTradesPanel'

interface AccountSummary {
  total_asset: number
  deposit: number
  eval_amount: number
  buy_amount: number
  eval_profit_loss: number
  return_pct: number
}

interface HoldingItem {
  stock_code: string
  stock_name: string
  quantity: number
  avg_price: number
  current_price: number
  eval_amount: number
  profit_loss: number
  return_pct: number
}

interface AccountConfig {
  mode: 'paper' | 'real'
  account_no: string
}

interface AccountData {
  mode: 'paper' | 'real'
  account_no: string
  summary: AccountSummary
  holdings: HoldingItem[]
  data_source: string
}

function formatKRW(n: number) {
  if (Math.abs(n) >= 100_000_000) return `${(n / 100_000_000).toFixed(1)}억`
  if (Math.abs(n) >= 10_000) return `${Math.floor(n / 10_000).toLocaleString()}만`
  return n.toLocaleString()
}

function ProfitText({ value, pct }: { value: number; pct: number }) {
  const color = value > 0 ? 'text-red-400' : value < 0 ? 'text-blue-400' : 'text-muted-foreground'
  const sign = value > 0 ? '+' : ''
  return (
    <span className={color}>
      {sign}{formatKRW(value)}원 ({sign}{pct.toFixed(2)}%)
    </span>
  )
}

interface AccountPanelProps {
  onClose: () => void
}

export function AccountPanel({ onClose }: AccountPanelProps) {
  const { user } = useAuthStore()
  const lastOrderAt = useUIStore((s) => s.lastOrderAt)
  const [accountConfig, setAccountConfig] = useState<AccountConfig | null>(null)
  const [data, setData] = useState<AccountData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchAccountConfig = useCallback(async () => {
    if (!user) return
    try {
      const { data: res } = await api.get<AccountConfig>('/account/config')
      setAccountConfig(res)
    } catch {
      setAccountConfig(null)
    }
  }, [user])

  const fetchBalance = useCallback(async () => {
    if (!user) return
    setLoading(true)
    setError(null)
    try {
      const { data: res } = await api.get<AccountData>('/account/balance')
      setData(res)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '잔고 조회에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }, [user])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void fetchAccountConfig()
      void fetchBalance()
    }, 0)
    return () => window.clearTimeout(timer)
  }, [fetchAccountConfig, fetchBalance])

  const displayMode = data?.mode ?? accountConfig?.mode ?? (user?.mode === 'real' ? 'real' : 'paper')

  return (
    <div className="fixed inset-y-0 right-0 w-80 bg-card border-l border-border shadow-2xl z-50 flex flex-col">
      {/* 헤더 */}
      <div className="flex items-center justify-between px-4 h-12 border-b border-border shrink-0">
        <div>
          <span className="font-semibold text-sm">내 계좌</span>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => { void fetchAccountConfig(); void fetchBalance() }} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {!user && (
          <div className="flex flex-col items-center justify-center h-40 text-sm text-muted-foreground gap-2">
            <span>로그인이 필요합니다.</span>
          </div>
        )}

        {user && loading && !data && (
          <div className="flex items-center justify-center h-40 text-sm text-muted-foreground">
            <RefreshCw className="h-4 w-4 animate-spin mr-2" /> 불러오는 중...
          </div>
        )}

        {user && error && (
          <div className="p-4">
            <div className="text-xs text-destructive bg-destructive/10 rounded p-3">{error}</div>
            <Button variant="outline" size="sm" className="mt-3 w-full" onClick={fetchBalance}>
              다시 시도
            </Button>
          </div>
        )}

        {/* 조회 모드 안내 — data null(잔고 조회 실패)이어도 항상 표시 */}
        {user && (
          <div
            data-testid="account-mode-info"
            className="mx-4 mt-3 text-xs text-muted-foreground bg-muted/40 border border-border rounded p-2 space-y-1"
          >
            <p>
              현재 조회 모드:{' '}
              <span className="font-medium text-foreground">
                {displayMode === 'paper' ? '모의투자' : '실계좌'}
              </span>
            </p>
            <p>실계좌/모의계좌 전환은 리스크/설정 화면에서만 변경합니다.</p>
          </div>
        )}

        {data && (
          <>
            {/* 계좌 요약 */}
            <div className="p-4 border-b border-border">
              <div className={`text-xs rounded p-2 mb-3 ${
                data.mode === 'real'
                  ? 'text-red-300 bg-red-500/10 border border-red-500/20'
                  : 'text-blue-300 bg-blue-500/10 border border-blue-500/20'
              }`}>
                {data.mode === 'real'
                  ? '실제 주문이 이 계좌로 실행됩니다.'
                  : '주문은 모의투자 계좌로 실행됩니다.'}
              </div>

              <div className="text-xs text-muted-foreground mb-1">{data.account_no}</div>
              <div className="text-xl font-bold mb-0.5">{data.summary.total_asset.toLocaleString()}원</div>
              <div className="text-sm">
                <ProfitText value={data.summary.eval_profit_loss} pct={data.summary.return_pct} />
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                <div>
                  <div className="text-muted-foreground">예수금</div>
                  <div className="font-medium">{formatKRW(data.summary.deposit)}원</div>
                </div>
                <div>
                  <div className="text-muted-foreground">평가금액</div>
                  <div className="font-medium">{formatKRW(data.summary.eval_amount)}원</div>
                </div>
                <div>
                  <div className="text-muted-foreground">매입금액</div>
                  <div className="font-medium">{formatKRW(data.summary.buy_amount)}원</div>
                </div>
              </div>
            </div>

            {/* 보유 종목 */}
            <div className="p-4">
              <div className="text-xs font-medium text-muted-foreground mb-3">
                보유 종목 ({data.holdings.length}개)
              </div>
              {data.holdings.length === 0 ? (
                <div className="text-xs text-muted-foreground text-center py-6">
                  KIS 계좌 조회 성공 · 현재 보유 중인 종목이 없습니다
                </div>
              ) : (
                <div className="space-y-3">
                  {data.holdings.map((h) => {
                    const Icon = h.profit_loss > 0 ? TrendingUp : h.profit_loss < 0 ? TrendingDown : Minus
                    const color = h.profit_loss > 0 ? 'text-red-400' : h.profit_loss < 0 ? 'text-blue-400' : 'text-muted-foreground'
                    const sign = h.profit_loss > 0 ? '+' : ''
                    return (
                      <div key={h.stock_code} className="bg-muted/40 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-1">
                          <div>
                            <span className="text-sm font-medium">{h.stock_name}</span>
                            <span className="ml-1.5 text-xs text-muted-foreground">{h.stock_code}</span>
                          </div>
                          <Icon className={`h-3.5 w-3.5 ${color}`} />
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">{h.quantity}주</span>
                          <span className="font-medium">{h.current_price.toLocaleString()}원</span>
                        </div>
                        <div className="flex justify-between text-xs mt-0.5">
                          <span className="text-muted-foreground">평균 {h.avg_price.toLocaleString()}원</span>
                          <span className={color}>
                            {sign}{formatKRW(h.profit_loss)} ({sign}{h.return_pct.toFixed(2)}%)
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </>
        )}

        {/* 최근 주문 패널 — 계좌/보유 종목과 같은 맥락에 배치 */}
        {user && <RecentTradesPanel refreshSignal={lastOrderAt} />}
      </div>

      {data && (
        <div className="px-4 py-2 border-t border-border shrink-0">
          <div className="text-xs text-muted-foreground text-center">{data.data_source}</div>
        </div>
      )}
    </div>
  )
}
