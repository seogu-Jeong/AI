/**
 * 지지/저항 레벨 오버레이.
 * LightweightCharts 캔들 차트에 수평 점선을 추가한다.
 */
import { useEffect, useRef } from 'react'
import { LineSeries } from 'lightweight-charts'
import type { IChartApi, ISeriesApi, Time } from 'lightweight-charts'
import type { Candle } from '@/types'

interface Level {
  price: number
  strength: number
}

interface SRData {
  support: Level[]
  resistance: Level[]
}

interface Props {
  chart: IChartApi | null
  candles: Candle[]
  data: SRData | null
}

export function SupportResistanceOverlay({ chart, candles, data }: Props) {
  const seriesRefs = useRef<ISeriesApi<'Line'>[]>([])

  useEffect(() => {
    if (!chart || !data || candles.length < 2) return

    // 기존 S/R 시리즈 제거
    seriesRefs.current.forEach((s) => { try { chart.removeSeries(s) } catch { /* 이미 제거됨 */ } })
    seriesRefs.current = []

    const firstTime = candles[0].time as Time
    const lastTime  = candles[candles.length - 1].time as Time

    const addLine = (price: number, color: string, label: string) => {
      const s = chart.addSeries(LineSeries, {
        color,
        lineWidth: 1,
        lineStyle: 2,   // Dashed
        title: label,
        lastValueVisible: true,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      })
      s.setData([
        { time: firstTime, value: price },
        { time: lastTime,  value: price },
      ])
      seriesRefs.current.push(s)
    }

    data.support.forEach(({ price }, i) =>
      addLine(price, '#22c55e', i === 0 ? `지지 ${price.toLocaleString()}` : '')
    )
    data.resistance.forEach(({ price }, i) =>
      addLine(price, '#ef4444', i === 0 ? `저항 ${price.toLocaleString()}` : '')
    )

    return () => {
      seriesRefs.current.forEach((s) => { try { chart.removeSeries(s) } catch { /* 무시 */ } })
      seriesRefs.current = []
    }
  }, [chart, candles, data])

  return null
}
