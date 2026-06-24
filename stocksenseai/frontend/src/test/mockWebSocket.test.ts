import { MockWebSocket } from '@/lib/mockWebSocket'

describe('MockWebSocket', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('calls onopen after construction', () => {
    const onopen = vi.fn()
    const ws = new MockWebSocket('ws://localhost/test')
    ws.onopen = onopen
    vi.advanceTimersByTime(1)  // fire setTimeout(0) only
    ws.close()                 // stop interval to prevent infinite tick
    expect(onopen).toHaveBeenCalled()
  })

  it('readyState is 1 (OPEN) after connection', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    vi.advanceTimersByTime(1)  // fire setTimeout(0) only
    expect(ws.readyState).toBe(1)
    ws.close()  // cleanup interval
  })

  it('calls onmessage with price data on interval', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    const onmessage = vi.fn()
    ws.onmessage = onmessage
    vi.advanceTimersByTime(2100)
    expect(onmessage).toHaveBeenCalled()
    const parsed = JSON.parse(onmessage.mock.calls[0][0].data)
    expect(parsed).toHaveProperty('price')
    expect(parsed).toHaveProperty('change_pct')
    ws.close()
  })

  it('close() sets readyState to 3 and calls onclose', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    const onclose = vi.fn()
    ws.onclose = onclose
    ws.close()
    expect(ws.readyState).toBe(3)
    expect(onclose).toHaveBeenCalled()
  })

  it('does not call onmessage after close()', () => {
    const ws = new MockWebSocket('ws://localhost/test')
    const onmessage = vi.fn()
    ws.onmessage = onmessage
    ws.close()
    vi.advanceTimersByTime(5000)
    expect(onmessage).not.toHaveBeenCalled()
  })
})
