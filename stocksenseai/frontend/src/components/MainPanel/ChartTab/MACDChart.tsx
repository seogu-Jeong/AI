import { useEffect, useRef } from 'react'
import { createChart, ColorType, HistogramSeries, LineSeries } from 'lightweight-charts'
import type { HistogramData, LineData, Time } from 'lightweight-charts'
import type { MACDPoint } from '@/types'

interface MACDChartProps {
  data: MACDPoint[]
}

export function MACDChart({ data }: MACDChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || data.length === 0) return

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: chartRef.current.clientWidth,
      height: chartRef.current.clientHeight,
      timeScale: { visible: false },
    })

    const histogram = chart.addSeries(HistogramSeries, {
      color: '#3fb950',
      priceFormat: { type: 'price', precision: 0 },
    })
    histogram.setData(
      data.map((d) => ({
        time: d.time as Time,
        value: d.histogram,
        color: d.histogram >= 0 ? '#3fb950' : '#ef4444',
      })) as HistogramData<Time>[]
    )

    const macdLine = chart.addSeries(LineSeries, { color: '#58a6ff', lineWidth: 1 })
    macdLine.setData(data.map((d) => ({ time: d.time as Time, value: d.macd })) as LineData<Time>[])

    const signalLine = chart.addSeries(LineSeries, { color: '#e3b341', lineWidth: 1 })
    signalLine.setData(data.map((d) => ({ time: d.time as Time, value: d.signal })) as LineData<Time>[])

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth })
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data])

  return (
    <div className="w-full h-full relative">
      <span className="absolute top-1 left-2 text-xs text-blue-400 font-semibold z-10">MACD</span>
      <div ref={chartRef} className="w-full h-full" />
    </div>
  )
}
