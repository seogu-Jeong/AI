/**
 * 이상 감지 점수 차트.
 * RSIChart 방식과 동일하게 독립 차트로 하단에 렌더링한다.
 * threshold 초과 구간을 강조하여 이상 패턴 감지 타이밍을 시각화한다.
 */
import { useEffect, useRef } from 'react'
import { createChart, ColorType, LineSeries, HistogramSeries } from 'lightweight-charts'
import type { Time } from 'lightweight-charts'

interface ScorePoint { date: string; score: number }
interface AnomalyData {
  scores: ScorePoint[]
  threshold: number
  available: boolean
}

interface Props { data: AnomalyData | null }

export function AnomalyChart({ data }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current || !data?.available || data.scores.length === 0) return

    const chart = createChart(ref.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      timeScale: { timeVisible: false },
      rightPriceScale: { borderVisible: false },
      width:  ref.current.clientWidth,
      height: ref.current.clientHeight,
    })

    // 이상 점수 히스토그램
    const histogram = chart.addSeries(HistogramSeries, {
      color: '#3b82f6',
      priceLineVisible: false,
    })
    histogram.setData(
      data.scores.map((s) => ({
        time:  s.date as Time,
        value: s.score,
        color: s.score > data.threshold ? '#f97316' : '#3b82f6',
      }))
    )

    // 임계값 수평선
    const threshLine = chart.addSeries(LineSeries, {
      color: '#f97316',
      lineWidth: 1,
      lineStyle: 2,
      title: '임계값',
      lastValueVisible: true,
      priceLineVisible: false,
    })
    const first = data.scores[0].date as Time
    const last  = data.scores[data.scores.length - 1].date as Time
    threshLine.setData([
      { time: first, value: data.threshold },
      { time: last,  value: data.threshold },
    ])

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth })
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data])

  if (!data?.available) return null

  return (
    <div className="w-full h-full relative">
      <span className="absolute top-0.5 left-1 text-[10px] text-muted-foreground z-10 pointer-events-none">
        이상 점수 (주황=이상 구간)
      </span>
      <div ref={ref} className="w-full h-full" />
    </div>
  )
}
