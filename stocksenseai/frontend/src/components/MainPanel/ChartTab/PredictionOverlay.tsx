import { useEffect, useRef } from 'react'
import { LineSeries } from 'lightweight-charts'
import type { IChartApi, ISeriesApi, SeriesType } from 'lightweight-charts'

interface PredictionData {
  bullish: number[]
  base: number[]
  bearish: number[]
  confidence: number
}

interface PredictionOverlayProps {
  chart: IChartApi | null
  prediction: PredictionData
  lastCandleTime: string
}

function addBusinessDays(dateStr: string, days: number): string {
  const date = new Date(dateStr)
  let added = 0
  while (added < days) {
    date.setDate(date.getDate() + 1)
    const day = date.getDay()
    if (day !== 0 && day !== 6) added++
  }
  return date.toISOString().split('T')[0]
}

export function PredictionOverlay({ chart, prediction, lastCandleTime }: PredictionOverlayProps) {
  const seriesRef = useRef<ISeriesApi<SeriesType>[]>([])

  useEffect(() => {
    if (!chart) return

    seriesRef.current.forEach((s) => {
      try { chart.removeSeries(s) } catch { /* series already removed */ }
    })
    seriesRef.current = []

    const configs = [
      { values: prediction.bullish, color: '#58a6ff' },
      { values: prediction.base,    color: '#c9d1d9' },
      { values: prediction.bearish, color: '#f85149' },
    ]

    configs.forEach(({ values, color }) => {
      if (values.length === 0) return
      const series = chart.addSeries(LineSeries, {
        color,
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      })

      const data = values.map((value, i) => ({
        time: addBusinessDays(lastCandleTime, i + 1),
        value,
      }))

      series.setData(data)
      seriesRef.current.push(series)
    })

    return () => {
      seriesRef.current.forEach((s) => {
        try { chart.removeSeries(s) } catch { /* series already removed */ }
      })
      seriesRef.current = []
    }
  }, [chart, prediction, lastCandleTime])

  return null
}
