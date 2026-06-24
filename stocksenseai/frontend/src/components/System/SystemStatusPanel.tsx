// frontend/src/components/System/SystemStatusPanel.tsx
import { useEffect, useState, useCallback, useRef } from 'react'
import { X, RefreshCw, CheckCircle2, AlertTriangle, XCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import api from '@/lib/api'
import type { SystemStatusResponse } from '@/types'

interface SystemStatusPanelProps {
  onClose: () => void
}

type ItemStatus = 'ok' | 'warn' | 'error' | 'loading'

function StatusIcon({ status }: { status: ItemStatus }) {
  if (status === 'loading') return <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
  if (status === 'ok') return <CheckCircle2 className="h-3.5 w-3.5 text-green-400" />
  if (status === 'warn') return <AlertTriangle className="h-3.5 w-3.5 text-yellow-400" />
  return <XCircle className="h-3.5 w-3.5 text-red-400" />
}

interface StatusRowProps {
  label: string
  value: string
  status: ItemStatus
  sub?: string
}

function StatusRow({ label, value, status, sub }: StatusRowProps) {
  return (
    <div className="flex items-start justify-between py-2.5 border-b border-border last:border-0">
      <div className="text-xs text-muted-foreground shrink-0 w-20">{label}</div>
      <div className="flex-1 text-right">
        <div className="flex items-center justify-end gap-1.5">
          <span className="text-xs font-medium">{value}</span>
          <StatusIcon status={status} />
        </div>
        {sub && <div className="text-[10px] text-muted-foreground mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}

function deriveKisStatus(data: SystemStatusResponse): ItemStatus {
  if (!data.kis.configured) return 'error'
  return 'ok'
}

function deriveAccountStatus(data: SystemStatusResponse): ItemStatus {
  if (data.account.ok === null) return 'warn'
  return data.account.ok ? 'ok' : 'error'
}

function derivePortfolioStatus(data: SystemStatusResponse): ItemStatus {
  if (data.portfolio.ok === null) return 'warn'
  return data.portfolio.ok ? 'ok' : 'warn'
}

function deriveAiStatus(data: SystemStatusResponse): ItemStatus {
  if (data.ai.prediction_source === 'unavailable') return 'warn'
  return 'ok'
}

function formatCheckedAt(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
  } catch {
    return iso
  }
}

export function SystemStatusPanel({ onClose }: SystemStatusPanelProps) {
  const [data, setData] = useState<SystemStatusResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    setError(null)
    let failed = false
    try {
      const res = await api.get<SystemStatusResponse>('/system/status')
      if (mountedRef.current) setData(res.data)
    } catch {
      failed = true
      if (mountedRef.current) setError('상태 조회에 실패했습니다. 백엔드 서버를 확인해 주세요.')
    }
    if (mountedRef.current) setLoading(false)
    void failed
  }, [])

  useEffect(() => {
    let cancelled = false

    async function fetchOnMount() {
      setLoading(true)
      setError(null)
      try {
        const res = await api.get<SystemStatusResponse>('/system/status')
        if (!cancelled) setData(res.data)
      } catch {
        if (!cancelled) setError('상태 조회에 실패했습니다. 백엔드 서버를 확인해 주세요.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void fetchOnMount()
    return () => { cancelled = true }
  }, [])

  return (
    <div
      className="fixed inset-y-0 right-0 w-80 bg-card border-l border-border shadow-2xl z-50 flex flex-col"
      role="dialog"
      aria-label="시스템 상태 패널"
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between px-4 h-12 border-b border-border shrink-0">
        <span className="font-semibold text-sm">시스템 상태</span>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={fetchStatus}
            disabled={loading}
            aria-label="새로고침"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={onClose}
            aria-label="닫기"
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* 본문 */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading && !data && (
          <div className="flex items-center justify-center h-32 gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            상태 확인 중…
          </div>
        )}

        {error && (
          <div className="text-xs text-destructive bg-destructive/10 rounded-lg p-3 mb-3">
            {error}
          </div>
        )}

        {data && (
          <div className="bg-muted/30 rounded-lg px-3">
            <StatusRow
              label="로그인"
              value={data.auth.logged_in ? (data.auth.email ?? '로그인됨') : '로그인 필요'}
              status={data.auth.logged_in ? 'ok' : 'warn'}
            />
            <StatusRow
              label="백엔드"
              value={data.backend.ok ? '정상' : '응답 없음'}
              status={data.backend.ok ? 'ok' : 'error'}
              sub={data.backend.message}
            />
            <StatusRow
              label="KIS 설정"
              value={
                data.kis.configured
                  ? (data.kis.mode === 'paper' ? '모의투자' : '실계좌')
                  : '키 미설정'
              }
              status={deriveKisStatus(data)}
              sub={data.kis.account_no ? `계좌 ${data.kis.account_no}` : data.kis.message}
            />
            <StatusRow
              label="계좌 잔고"
              value={
                data.account.ok === null
                  ? 'KIS 미설정'
                  : data.account.ok
                    ? (data.account.holdings_count === 0 ? '조회 성공 (보유 없음)' : `${data.account.holdings_count}종목 보유`)
                    : '조회 실패'
              }
              status={deriveAccountStatus(data)}
              sub={data.account.ok ? data.account.data_source ?? undefined : data.account.message}
            />
            <StatusRow
              label="포트폴리오"
              value={
                data.portfolio.ok === null
                  ? '로그인 필요'
                  : data.portfolio.ok
                    ? (data.portfolio.holding_source ?? '조회 성공')
                    : '앱 DB fallback'
              }
              status={derivePortfolioStatus(data)}
              sub={data.portfolio.performance_source ?? undefined}
            />
            <StatusRow
              label="AI 예측"
              value={
                data.ai.prediction_source === 'uploaded'
                  ? '업로드 예측'
                  : data.ai.prediction_source === 'local'
                    ? '로컬 추론'
                    : '없음'
              }
              status={deriveAiStatus(data)}
              sub={data.ai.message}
            />
          </div>
        )}
      </div>

      {/* 푸터 — 최근 확인 시각 */}
      {data && (
        <div className="px-4 py-2 border-t border-border shrink-0 text-center">
          <span className="text-[10px] text-muted-foreground">
            최근 확인: {formatCheckedAt(data.checked_at)}
          </span>
        </div>
      )}
    </div>
  )
}
