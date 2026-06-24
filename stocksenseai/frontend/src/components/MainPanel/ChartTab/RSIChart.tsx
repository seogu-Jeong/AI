import { useEffect, useRef } from 'react'
import { createChart, ColorType, LineSeries } from 'lightweight-charts'
import type { LineData, Time } from 'lightweight-charts'

interface RSIChartProps {
  data: { time: string | number; value: number }[]
}

export function RSIChart({ data }: RSIChartProps) {
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
      rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.1 } },
      timeScale: { visible: false },
    })

    const series = chart.addSeries(LineSeries, { color: '#58a6ff', lineWidth: 1 })
    series.setData(data as LineData<Time>[])

    const overbought = chart.addSeries(LineSeries, { color: '#f85149', lineWidth: 1, lineStyle: 2 })
    const oversold = chart.addSeries(LineSeries, { color: '#3fb950', lineWidth: 1, lineStyle: 2 })
    if (data.length > 0) {
      overbought.setData([
        { time: data[0].time as Time, value: 70 },
        { time: data[data.length - 1].time as Time, value: 70 },
      ])
      oversold.setData([
        { time: data[0].time as Time, value: 30 },
        { time: data[data.length - 1].time as Time, value: 30 },
      ])
    }

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
      <span className="absolute top-1 left-2 text-xs text-blue-400 font-semibold z-10">RSI (14)</span>
      <div ref={chartRef} className="w-full h-full" />
    </div>
  )
}
