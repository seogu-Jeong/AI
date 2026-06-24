import { useEffect, useState } from 'react'
import { useStockStore } from '@/store/stockStore'

const API_BASE = import.meta.env.VITE_API_BASE ?? null

export function useStockWebSocket(stockCode: string): { isConnected: boolean } {
  const [isConnected, setIsConnected] = useState(false)
  const updateRealtimePrice = useStockStore((s) => s.updateRealtimePrice)

  useEffect(() => {
    if (!stockCode) return

    if (!API_BASE || typeof EventSource === 'undefined') {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setIsConnected(true)
      const timer = setInterval(() => {
        const mockPrice = 70000 + Math.round(Math.random() * 10000)
        updateRealtimePrice({ code: stockCode, price: mockPrice, change_pct: +(Math.random() * 4 - 2).toFixed(2) })
      }, 3000)
      return () => { clearInterval(timer); setIsConnected(false) }
    }

    const es = new EventSource(`${API_BASE}/ws/stocks/${stockCode}`)
    es.onopen = () => setIsConnected(true)
    es.onerror = () => setIsConnected(false)
    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        updateRealtimePrice({ code: stockCode, price: data.price, change_pct: data.change_pct })
      } catch { /* malformed SSE message, ignore */ }
    }

    return () => { es.close(); setIsConnected(false) }
  }, [stockCode, updateRealtimePrice])

  return { isConnected }
}
