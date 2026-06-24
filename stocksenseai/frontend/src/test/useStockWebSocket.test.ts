import { renderHook, act } from '@testing-library/react'
import { useStockWebSocket } from '@/hooks/useStockWebSocket'
import { useStockStore } from '@/store/stockStore'

describe('useStockWebSocket', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    useStockStore.setState({ realtimePrice: null })
  })
  afterEach(() => vi.useRealTimers())

  it('returns isConnected true after mount', () => {
    const { result } = renderHook(() => useStockWebSocket('005930'))
    act(() => vi.advanceTimersByTime(1))
    expect(result.current.isConnected).toBe(true)
  })

  it('updates realtimePrice in store after interval', () => {
    renderHook(() => useStockWebSocket('005930'))
    act(() => vi.advanceTimersByTime(3000))
    const price = useStockStore.getState().realtimePrice
    expect(price).not.toBeNull()
    expect(price?.price).toBeGreaterThan(0)
  })

  it('reconnects when stockCode changes — new connection becomes live', () => {
    const { result, rerender } = renderHook(({ code }) => useStockWebSocket(code), {
      initialProps: { code: '005930' },
    })
    act(() => vi.advanceTimersByTime(1))
    expect(result.current.isConnected).toBe(true)
    rerender({ code: '000660' })
    act(() => vi.advanceTimersByTime(1))
    expect(result.current.isConnected).toBe(true)
  })
})
