import { useEffect, useRef } from 'react'
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts'
import type { IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts'
import type { Candle } from '@/types'

interface CandleChartProps {
  candles: Candle[]
  onChartReady?: (chart: IChartApi) => void
}

export function CandleChart({ candles, onChartReady }: CandleChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  // 차트는 마운트 시 한 번만 생성
  useEffect(() => {
    if (!chartRef.current) return

    const fmtDate = (ts: number) => {
      const d = new Date(ts * 1000)
      const yy = String(d.getFullYear()).slice(2)
      const mm = String(d.getMonth() + 1).padStart(2, '0')
      const dd = String(d.getDate()).padStart(2, '0')
      return `${yy}/${mm}/${dd}`
    }
    const fmtTime = (ts: number) => {
      const d = new Date(ts * 1000)
      return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
    }

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      localization: {
        timeFormatter: (t: unknown) => {
          if (typeof t === 'number') return `${fmtDate(t)} ${fmtTime(t)}`
          if (typeof t === 'string') {
            const [y, m, d] = t.split('-')
            return y && m && d ? `${y.slice(2)}/${m}/${d}` : t
          }
          if (t && typeof t === 'object') {
            const bd = t as Record<string, number>
            if (bd.year !== undefined)
              return `${String(bd.year).slice(2)}/${String(bd.month).padStart(2,'0')}/${String(bd.day).padStart(2,'0')}`
          }
          return String(t)
        },
      },
      timeScale: {
        timeVisible: false,
        secondsVisible: false,
        tickMarkFormatter: (time: number | string | { year: number; month: number; day: number }, tickMarkType: number) => {
          if (typeof time === 'number') {
            return tickMarkType >= 3 ? fmtTime(time) : fmtDate(time)
          }
          if (typeof time === 'string') {
            const [y, m, d] = time.split('-')
            return `${y.slice(2)}/${m}/${d}`
          }
          const bd = time as { year: number; month: number; day: number }
          return `${String(bd.year).slice(2)}/${String(bd.month).padStart(2,'0')}/${String(bd.day).padStart(2,'0')}`
        },
      },
      width: chartRef.current.clientWidth,
      height: chartRef.current.clientHeight,
    })

    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    chartInstanceRef.current = chart
    seriesRef.current = series
    onChartReady?.(chart)

    const handleResize = () => {
      if (chartRef.current && chartInstanceRef.current) {
        chartInstanceRef.current.applyOptions({ width: chartRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartInstanceRef.current = null
      seriesRef.current = null
    }
  }, [onChartReady])

  // 데이터만 별도로 업데이트
  useEffect(() => {
    if (!seriesRef.current) return
    if (candles.length === 0) {
      seriesRef.current.setData([])
      return
    }
    const isIntraday = typeof candles[0].time === 'number'
    chartInstanceRef.current?.applyOptions({ timeScale: { timeVisible: isIntraday } })
    seriesRef.current.setData(candles as CandlestickData<Time>[])
    chartInstanceRef.current?.timeScale().fitContent()
  }, [candles])

  return <div ref={chartRef} className="w-full h-full" />
}
