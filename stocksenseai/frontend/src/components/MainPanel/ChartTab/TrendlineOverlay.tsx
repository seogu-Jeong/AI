/**
 * 추세선 오버레이.
 * 고점 연결(저항 추세선)·저점 연결(지지 추세선)을 LightweightCharts에 추가한다.
 */
import { useEffect, useRef } from 'react'
import { LineSeries } from 'lightweight-charts'
import type { IChartApi, ISeriesApi, Time } from 'lightweight-charts'

interface TrendPoint { date: string; price: number }
interface TrendLine {
  label: string
  color: string
  r2: number
  points: TrendPoint[]
  pivots: TrendPoint[]
}
interface TrendlineData { lines: TrendLine[]; available: boolean }

interface Props {
  chart: IChartApi | null
  data: TrendlineData | null
}

export function TrendlineOverlay({ chart, data }: Props) {
  const seriesRefs = useRef<ISeriesApi<'Line'>[]>([])

  useEffect(() => {
    if (!chart || !data?.available) return

    seriesRefs.current.forEach((s) => { try { chart.removeSeries(s) } catch { /* 무시 */ } })
    seriesRefs.current = []

    data.lines.forEach((line) => {
      if (line.points.length < 2) return
      try {
        const s = chart.addSeries(LineSeries, {
          color: line.color,
          lineWidth: 2,
          lineStyle: 0,   // Solid
          title: line.label,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        })
        s.setData(
          line.points.map((p) => ({ time: p.date as Time, value: p.price }))
        )
        seriesRefs.current.push(s)
      } catch (e) {
        console.warn('[TrendlineOverlay] addSeries 실패:', line.label, e)
      }
    })

    return () => {
      seriesRefs.current.forEach((s) => { try { chart.removeSeries(s) } catch { /* 무시 */ } })
      seriesRefs.current = []
    }
  }, [chart, data])

  return null
}
