import { useEffect, useState, useCallback, useRef } from 'react'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import api from '@/lib/api'
import { useStockStore } from '@/store/stockStore'
import {
  Bot, Play, Square, RefreshCw, TrendingUp, TrendingDown,
  ChevronDown, ChevronUp, Clock, BarChart2, Trash2,
} from 'lucide-react'

interface AutoTradeConfig {
  id: string
  enabled: boolean
  mode: 'paper' | 'real'
  total_budget: number
  stop_loss_pct: number
  take_profit_pct: number
}

interface AutoTradeLog {
  id: string
  stock_code: string
  stock_name: string
  action: string
  quantity: number
  price: number
  total_amount: number
  reason: string
  signal_score: number
  mode: string
  created_at: string
}

interface ScanStock {
  code: string
  name: string
  signal: 'BUY' | 'HOLD' | 'SELL'
  score: number
  rsi: number
}

const POLL_INTERVAL = 5 * 60  // 5분 (초)

function formatKRW(n: number) {
  if (n >= 100000000) return `${(n / 100000000).toFixed(1)}억원`
  if (n >= 10000) return `${Math.floor(n / 10000)}만원`
  return `${n.toLocaleString()}원`
}

function formatCountdown(sec: number) {
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return m > 0 ? `${m}분 ${s}초` : `${s}초`
}

function SignalBadge({ signal, score }: { signal: string; score: number }) {
  if (signal === 'BUY') return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-green-500/20 text-green-400">
      ▲ BUY {score > 0 ? `${score.toFixed(0)}점` : ''}
    </span>
  )
  if (signal === 'SELL') return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-red-500/20 text-red-400">
      ▼ SELL {score > 0 ? `${score.toFixed(0)}점` : ''}
    </span>
  )
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-muted text-muted-foreground">
      — HOLD
    </span>
  )
}

export function AutoTradePanel() {
  const { watchlist } = useStockStore()

  const [config, setConfig] = useState<AutoTradeConfig>({
    id: '', enabled: false, mode: 'paper',
    total_budget: 1000000, stop_loss_pct: 5, take_profit_pct: 10,
  })
  const [logs, setLogs] = useState<AutoTradeLog[]>([])
  const [scanStocks, setScanStocks] = useState<ScanStock[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [running, setRunning] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [runResult, setRunResult] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showScan, setShowScan] = useState(true)
  const [countdown, setCountdown] = useState(POLL_INTERVAL)
  const [lastRunAt, setLastRunAt] = useState<Date | null>(null)

  const configRef = useRef(config)
  useEffect(() => {
    configRef.current = config
  }, [config])

  const watchlistRef = useRef(watchlist)
  useEffect(() => {
    watchlistRef.current = watchlist
  }, [watchlist])

  const fetchLogs = useCallback(async () => {
    const res = await api.get('/auto-trade/logs?limit=50')
    setLogs(res.data.logs || [])
  }, [])

  const fetchScan = useCallback(async (codes?: string[]) => {
    setScanning(true)
    try {
      const res = await api.post('/auto-trade/scan', { codes: codes ?? watchlistRef.current })
      setScanStocks(res.data.stocks || [])
    } catch {
      // scan failure는 무시
    } finally {
      setScanning(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [configRes, logsRes] = await Promise.all([
          api.get('/auto-trade/config'),
          api.get('/auto-trade/logs?limit=50'),
        ])
        if (!cancelled) {
          setConfig(configRes.data)
          setLogs(logsRes.data.logs || [])
          fetchScan(watchlistRef.current)
        }
      } catch {
        if (!cancelled) setError('데이터를 불러오지 못했습니다.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [fetchScan])

  // 5분 자동 실행 (enabled일 때)
  useEffect(() => {
    if (!configRef.current.enabled) {
      setCountdown(POLL_INTERVAL)
      return
    }

    setCountdown(POLL_INTERVAL)

    // 카운트다운 타이머
    const tick = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) return POLL_INTERVAL
        return prev - 1
      })
    }, 1000)

    // 5분마다 자동 실행
    const runner = setInterval(async () => {
      if (!configRef.current.enabled) return
      try {
        const res = await api.post('/auto-trade/run', { extra_codes: watchlistRef.current })
        setLastRunAt(new Date())
        const { executed, scanned } = res.data
        if (executed > 0) {
          setRunResult(`자동 실행 ${executed}건 완료 (${scanned}종목 분석)`)
          await fetchLogs()
        }
        // 스캔 결과 갱신
        fetchScan(watchlistRef.current)
      } catch {
        // 자동 실행 에러는 조용히 무시
      }
    }, POLL_INTERVAL * 1000)

    return () => {
      clearInterval(tick)
      clearInterval(runner)
    }
  }, [config.enabled, fetchLogs, fetchScan])

  const handleToggle = async (enabled: boolean) => {
    if (enabled && config.mode === 'real') {
      if (!window.confirm('실거래 모드입니다. 실제 자금이 사용됩니다. 계속하시겠습니까?')) return
    }
    setError(null)
    try {
      const res = await api.put('/auto-trade/config', { enabled })
      setConfig(res.data)
      if (enabled) setCountdown(POLL_INTERVAL)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err.response?.data?.detail || '변경 실패')
    }
  }

  const handleSaveBudget = async () => {
    setSaving(true)
    setError(null)
    try {
      const res = await api.put('/auto-trade/config', {
        total_budget: config.total_budget,
        stop_loss_pct: config.stop_loss_pct,
        take_profit_pct: config.take_profit_pct,
        mode: config.mode,
      })
      setConfig(res.data)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err.response?.data?.detail || '저장 실패')
    } finally {
      setSaving(false)
    }
  }

  const handleRunNow = async () => {
    setRunning(true)
    setRunResult(null)
    setError(null)
    try {
      const res = await api.post('/auto-trade/run', { extra_codes: watchlist })
      if (res.data.skipped) {
        const reason = res.data.reason
        if (reason === 'already_running') {
          setRunResult('분석 중입니다. 잠시 후 결과가 업데이트됩니다.')
        } else if (reason === 'not_enabled') {
          setError('자동매매를 먼저 활성화해 주세요.')
        } else {
          setError(res.data.message || '실행을 건너뛰었습니다.')
        }
      } else {
        const { executed, scanned, held_count, no_trade_reason } = res.data
        if (executed > 0) {
          setRunResult(`${executed}건 실행 완료 (${scanned}종목 분석)`)
        } else {
          const detail = no_trade_reason || '조건 미충족'
          const scanInfo = scanned > 0 ? `${scanned}종목 분석` : '종목 분석 중'
          const heldInfo = held_count > 0 ? ` · 보유 ${held_count}종목` : ''
          setRunResult(`매매 없음 — ${scanInfo}${heldInfo} → ${detail}`)
        }
        setLastRunAt(new Date())
        if (config.enabled) setCountdown(POLL_INTERVAL)
        await fetchLogs()
        fetchScan(watchlist)
      }
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string }; status?: number } }
      if (err.response?.status === 429) {
        setError('요청이 너무 많습니다. 잠시 후 다시 시도해주세요.')
      } else {
        setError(err.response?.data?.detail || '실행 실패')
      }
    } finally {
      setRunning(false)
    }
  }

  const handleStop = async () => {
    if (!window.confirm('자동매매를 즉시 중지합니다. 보유 포지션은 그대로 유지됩니다.')) return
    try {
      await api.post('/auto-trade/stop')
      setConfig(prev => ({ ...prev, enabled: false }))
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err.response?.data?.detail || '중지 실패')
    }
  }

  const handleReset = async () => {
    if (!window.confirm('모의매매 포지션과 거래 기록을 전부 초기화합니다.\n자동매매도 중지됩니다. 계속하시겠습니까?')) return
    try {
      await api.post('/auto-trade/reset')
      setConfig(prev => ({ ...prev, enabled: false }))
      setLogs([])
      setScanStocks([])
      setRunResult(null)
      setError(null)
      fetchScan(watchlist)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } }
      setError(err.response?.data?.detail || '초기화 실패')
    }
  }

  const totalBuy = logs.filter(l => l.action === 'BUY').reduce((s, l) => s + l.total_amount, 0)
  const totalSell = logs.filter(l => l.action === 'SELL').reduce((s, l) => s + l.total_amount, 0)
  const invested = totalBuy - totalSell

  const buyCount = scanStocks.filter(s => s.signal === 'BUY').length
  const sellCount = scanStocks.filter(s => s.signal === 'SELL').length

  if (loading) return (
    <div className="flex items-center justify-center h-full text-muted-foreground text-sm">불러오는 중...</div>
  )

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto p-4 space-y-3">

        {/* 메인 카드 */}
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          {/* 상태 헤더 */}
          <div className={`px-5 py-4 ${config.enabled ? 'bg-green-500/10 border-b border-green-500/20' : 'border-b border-border'}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-9 h-9 rounded-full flex items-center justify-center ${config.enabled ? 'bg-green-500/20' : 'bg-muted'}`}>
                  <Bot className={`w-4 h-4 ${config.enabled ? 'text-green-400' : 'text-muted-foreground'}`} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm">AI 자동매매</span>
                    <Badge variant={config.enabled ? 'default' : 'outline'} className="text-xs py-0">
                      {config.enabled ? '● 실행 중' : '중지됨'}
                    </Badge>
                    <Badge variant="outline" className="text-xs py-0">{config.mode === 'paper' ? '모의' : '실거래'}</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {config.enabled
                      ? config.mode === 'real'
                        ? 'KIS 예수금 기준 실거래 · 5분마다 자동 스캔'
                        : `${formatKRW(config.total_budget)} 모의 운용 · 5분마다 자동 스캔`
                      : config.mode === 'real'
                        ? 'KIS 계좌 예수금 자동 적용 · 실거래 모드'
                        : '종가 기준 모의 자동매매 · 5분 주기'}
                  </p>
                </div>
              </div>
              <Switch checked={config.enabled} onCheckedChange={handleToggle} />
            </div>

            {/* 다음 실행 카운트다운 */}
            {config.enabled && (
              <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  다음 실행: <span className="text-green-400 font-medium ml-1">{formatCountdown(countdown)}</span>
                </span>
                {lastRunAt && (
                  <span>마지막 실행: {lastRunAt.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}</span>
                )}
              </div>
            )}
          </div>

          {/* 예산 입력 */}
          <div className="px-5 py-4 space-y-3">
            {config.mode === 'real' ? (
              <div className="flex items-center gap-3 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <div className="flex-1">
                  <p className="text-xs font-medium text-blue-400">실거래 모드 — KIS 예수금 자동 적용</p>
                  <p className="text-xs text-muted-foreground mt-0.5">예산 별도 설정 불필요. KIS 계좌의 실시간 예수금 기준으로 매매합니다.</p>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <label className="text-xs text-muted-foreground block mb-1">AI에게 맡길 금액 (모의)</label>
                  <div className="relative">
                    <Input
                      type="number"
                      value={config.total_budget}
                      onChange={e => setConfig(prev => ({ ...prev, total_budget: Number(e.target.value) }))}
                      min={10000} step={100000}
                      className="h-10 text-sm pr-8"
                      placeholder="1000000"
                    />
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">원</span>
                  </div>
                </div>
                <Button onClick={handleSaveBudget} disabled={saving} className="mt-5 h-10 px-4">
                  {saving ? '저장...' : '설정'}
                </Button>
              </div>
            )}

            {/* 고급 설정 토글 */}
            <button
              onClick={() => setShowAdvanced(v => !v)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              안전장치 설정
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-3 gap-3 pt-1">
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">거래 모드</label>
                  <div className="flex gap-1">
                    {(['paper', 'real'] as const).map(m => (
                      <button key={m} onClick={() => setConfig(prev => ({ ...prev, mode: m }))}
                        className={`flex-1 py-1 text-xs rounded border transition-colors ${config.mode === m ? 'bg-primary text-primary-foreground border-primary' : 'border-border text-muted-foreground'}`}>
                        {m === 'paper' ? '모의' : '실거래'}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">손절 기준</label>
                  <div className="relative">
                    <Input type="number" value={config.stop_loss_pct}
                      onChange={e => setConfig(prev => ({ ...prev, stop_loss_pct: Number(e.target.value) }))}
                      min={1} max={30} step={0.5} className="h-8 text-xs pr-6" />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-red-400">%</span>
                  </div>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">익절 기준</label>
                  <div className="relative">
                    <Input type="number" value={config.take_profit_pct}
                      onChange={e => setConfig(prev => ({ ...prev, take_profit_pct: Number(e.target.value) }))}
                      min={1} max={100} step={0.5} className="h-8 text-xs pr-6" />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-green-400">%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* 액션 버튼 */}
          <div className="px-5 pb-4 flex gap-2">
            <Button variant="outline" onClick={handleRunNow} disabled={running} className="flex-1 h-9">
              {running ? <><RefreshCw className="w-3.5 h-3.5 mr-2 animate-spin" />분석 중...</> : <><Play className="w-3.5 h-3.5 mr-2" />지금 실행</>}
            </Button>
            {config.enabled && (
              <Button variant="destructive" onClick={handleStop} className="h-9 px-4">
                <Square className="w-3.5 h-3.5 mr-1.5" />긴급 정지
              </Button>
            )}
            {config.mode === 'paper' && !config.enabled && (
              <Button
                variant="outline"
                onClick={handleReset}
                className="h-9 px-3 text-muted-foreground hover:text-destructive hover:border-destructive"
                title="모의매매 초기화"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        </div>

        {/* 알림 */}
        {error && <div className="bg-destructive/10 border border-destructive/30 rounded-lg px-4 py-2.5 text-sm text-destructive">{error}</div>}
        {runResult && (
          <div className={`border rounded-lg px-4 py-2.5 text-sm ${
            runResult.startsWith('매매 없음')
              ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400'
              : 'bg-green-500/10 border-green-500/30 text-green-400'
          }`}>
            {runResult.startsWith('매매 없음') ? '⚠️' : '✅'} {runResult}
          </div>
        )}

        {/* AI 분석 결과 */}
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <button
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-muted/20 transition-colors"
            onClick={() => setShowScan(v => !v)}
          >
            <div className="flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium">AI 종목 분석</span>
              {scanStocks.length > 0 && (
                <div className="flex items-center gap-1">
                  {buyCount > 0 && <span className="text-xs px-1.5 py-0.5 rounded-full bg-green-500/20 text-green-400">BUY {buyCount}</span>}
                  {sellCount > 0 && <span className="text-xs px-1.5 py-0.5 rounded-full bg-red-500/20 text-red-400">SELL {sellCount}</span>}
                  <span className="text-xs text-muted-foreground">{scanStocks.length}종목</span>
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={e => { e.stopPropagation(); fetchScan(watchlist) }}
                className="p-1 hover:text-foreground text-muted-foreground"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${scanning ? 'animate-spin' : ''}`} />
              </button>
              {showScan ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
            </div>
          </button>

          {showScan && (
            <div className="border-t border-border">
              {scanStocks.length === 0 ? (
                <div className="flex flex-col items-center py-8 text-muted-foreground">
                  <BarChart2 className="w-6 h-6 mb-2 opacity-20" />
                  <p className="text-xs">분석 데이터 없음</p>
                </div>
              ) : (
                <div className="divide-y divide-border/50">
                  {/* 헤더 */}
                  <div className="grid grid-cols-[1fr_auto_auto] gap-3 px-4 py-2 text-xs text-muted-foreground">
                    <span>종목</span>
                    <span className="text-right">RSI</span>
                    <span className="text-right w-24">신호</span>
                  </div>
                  {scanStocks.map(stock => (
                    <div
                      key={stock.code}
                      className={`grid grid-cols-[1fr_auto_auto] gap-3 items-center px-4 py-2.5 text-sm ${
                        stock.signal === 'BUY' ? 'bg-green-500/5' : stock.signal === 'SELL' ? 'bg-red-500/5' : ''
                      }`}
                    >
                      <div>
                        <span className="font-medium">{stock.name}</span>
                        <span className="text-xs text-muted-foreground ml-2">{stock.code}</span>
                      </div>
                      <span className={`text-right text-xs tabular-nums ${
                        stock.rsi > 70 ? 'text-red-400' : stock.rsi < 30 ? 'text-green-400' : 'text-muted-foreground'
                      }`}>
                        {stock.rsi > 0 ? stock.rsi.toFixed(1) : '-'}
                      </span>
                      <div className="flex justify-end w-24">
                        <SignalBadge signal={stock.signal} score={stock.score} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 운용 현황 */}
        {logs.length > 0 && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: '총 매수', value: formatKRW(totalBuy) },
              { label: '총 매도', value: formatKRW(totalSell) },
              { label: '현재 투자', value: formatKRW(Math.max(0, invested)), highlight: invested > 0 },
            ].map(item => (
              <div key={item.label} className="bg-card border border-border rounded-lg p-3 text-center">
                <p className="text-xs text-muted-foreground">{item.label}</p>
                <p className={`text-sm font-semibold mt-1 ${item.highlight ? 'text-primary' : 'text-foreground'}`}>{item.value}</p>
              </div>
            ))}
          </div>
        )}

        {/* 거래 기록 */}
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium">거래 기록</h3>
            <button onClick={fetchLogs} className="text-muted-foreground hover:text-foreground">
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          {logs.length === 0 ? (
            <div className="flex flex-col items-center py-10 text-muted-foreground">
              <Bot className="w-8 h-8 mb-2 opacity-20" />
              <p className="text-sm">아직 거래 기록이 없습니다</p>
              <p className="text-xs mt-1 opacity-70">금액 설정 후 "지금 실행"을 눌러보세요</p>
            </div>
          ) : (
            <div className="space-y-1">
              {logs.map(log => (
                <div key={log.id} className="flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-muted/30 transition-colors">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${log.action === 'BUY' ? 'bg-green-500/15' : 'bg-red-500/15'}`}>
                    {log.action === 'BUY'
                      ? <TrendingUp className="w-3 h-3 text-green-400" />
                      : <TrendingDown className="w-3 h-3 text-red-400" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className="text-sm font-medium">{log.stock_name || log.stock_code}</span>
                    <span className="text-xs text-muted-foreground ml-2">{log.reason}</span>
                  </div>
                  <div className="text-right shrink-0">
                    <p className={`text-sm font-medium ${log.action === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                      {log.action === 'BUY' ? '-' : '+'}{formatKRW(log.total_amount)}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {log.created_at ? new Date(log.created_at).toLocaleString('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : ''}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
