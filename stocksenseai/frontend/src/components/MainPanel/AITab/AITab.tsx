import { useEffect, useState } from 'react'
import { useStockStore } from '@/store/stockStore'
import type { AISignal, MultiframeSignal } from '@/types'
import { MOCK_AI_SIGNAL, MOCK_MULTIFRAME } from '@/lib/mockData'
import api from '@/lib/api'
import { SignalCard } from './SignalCard'
import { ScoreBreakdown } from './ScoreBreakdown'
import { MultiframePanel } from './MultiframePanel'
import { FundamentalPanel } from '@/components/Analysis/FundamentalPanel'
import { RecommendationPanel } from '@/components/Analysis/RecommendationPanel'
import { ComprehensivePanel } from '@/components/Analysis/ComprehensivePanel'
import { InvestorPanel } from '@/components/Analysis/InvestorPanel'

// 백엔드 /ai/{code}/signal 응답 — signal_breakdown으로 점수를 감싸서 반환한다.
interface SignalResponse {
  signal: AISignal['signal']
  signal_score: number
  signal_breakdown?: {
    technical_score: number
    lstm_score: number
  }
  lstm_available?: boolean
}

// 백엔드 /ai/{code}/multiframe 응답 — timeframes 객체(daily/weekly/monthly).
interface MultiframeResponse {
  timeframes?: Record<string, { signal: MultiframeSignal['signal']; score: number }>
}

const TF_LABELS: [string, MultiframeSignal['timeframe']][] = [
  ['daily', '1D'],
  ['weekly', '1W'],
  ['monthly', '1M'],
]

export function AITab() {
  const { selectedStock } = useStockStore()
  const [signal, setSignal] = useState<AISignal>(MOCK_AI_SIGNAL)
  const [multiframe, setMultiframe] = useState<MultiframeSignal[]>(MOCK_MULTIFRAME)
  const [loading, setLoading] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)

  useEffect(() => {
    if (!selectedStock?.code) return
    const code = selectedStock.code
    const fetchData = async () => {
      setFetchError(null)
      setLoading(true)
      try {
        const [sigRes, mfRes] = await Promise.all([
          api.get<SignalResponse>(`/ai/${code}/signal`),
          api.get<MultiframeResponse>(`/ai/${code}/multiframe`),
        ])

        const sd = sigRes.data
        if (sd?.signal) {
          setSignal((prev) => ({
            ...prev,
            signal: sd.signal,
            signal_score: sd.signal_score,
            tech_score: sd.signal_breakdown?.technical_score ?? sd.signal_score,
            lstm_score: sd.signal_breakdown?.lstm_score ?? 50,
            confidence: Math.min(1, Math.abs(sd.signal_score - 50) / 50),
          }))
        }

        const tf = mfRes.data?.timeframes
        if (tf) {
          const frames = TF_LABELS.filter(([key]) => tf[key]).map(([key, label]) => ({
            timeframe: label,
            signal: tf[key].signal,
            score: tf[key].score,
          }))
          if (frames.length > 0) setMultiframe(frames)
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        setFetchError(msg)
        console.error('[AITab] fetch 실패:', e)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [selectedStock?.code])

  const { signal: sig, signal_score, tech_score, lstm_score, confidence } = signal

  return (
    <div className="h-full overflow-y-auto p-4">
      {/* 좌: 기존 AI 점수 / 우: 재무 평가 + 추천 종목 (실데이터) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          {loading && (
            <div className="text-xs text-muted-foreground text-center py-2">AI 분석 중...</div>
          )}
          {fetchError && (
            <div className="text-xs text-red-400 text-center py-2 border border-red-400/30 rounded px-3">
              API 오류: {fetchError}
            </div>
          )}
          <SignalCard
            signal={sig}
            signal_score={signal_score}
            confidence={confidence}
          />
          <ScoreBreakdown
            tech_score={tech_score}
            lstm_score={lstm_score}
            confidence={confidence}
          />
          <MultiframePanel signals={multiframe} />
        </div>
        <div className="space-y-4">
          <ComprehensivePanel />
          <FundamentalPanel />
          <InvestorPanel />
          <RecommendationPanel />
        </div>
      </div>
    </div>
  )
}
