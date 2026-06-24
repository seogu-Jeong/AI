import { calculateRSI, calculateMACD } from '@/lib/indicators'
import type { Candle } from '@/types'

function makeCandlesAscending(count: number): Candle[] {
  return Array.from({ length: count }, (_, i) => ({
    time: `2026-0${Math.floor(i / 28) + 1}-${String((i % 28) + 1).padStart(2, '0')}`,
    open: 1000 + i * 10,
    high: 1010 + i * 10,
    low: 990 + i * 10,
    close: 1005 + i * 10,
    volume: 1000000,
  }))
}

function makeCandlesDescending(count: number): Candle[] {
  return Array.from({ length: count }, (_, i) => ({
    time: `2026-0${Math.floor(i / 28) + 1}-${String((i % 28) + 1).padStart(2, '0')}`,
    open: 2000 - i * 10,
    high: 2010 - i * 10,
    low: 1990 - i * 10,
    close: 1995 - i * 10,
    volume: 1000000,
  }))
}

describe('calculateRSI', () => {
  it('returns empty array for fewer than 15 candles', () => {
    const candles = makeCandlesAscending(10)
    expect(calculateRSI(candles)).toHaveLength(0)
  })

  it('returns array of length candles.length - 14 for sufficient data', () => {
    const candles = makeCandlesAscending(30)
    expect(calculateRSI(candles)).toHaveLength(30 - 14)
  })

  it('RSI is above 70 for consistently rising prices', () => {
    const candles = makeCandlesAscending(40)
    const result = calculateRSI(candles)
    const last = result[result.length - 1]
    expect(last.value).toBeGreaterThan(70)
  })

  it('RSI is below 30 for consistently falling prices', () => {
    const candles = makeCandlesDescending(40)
    const result = calculateRSI(candles)
    const last = result[result.length - 1]
    expect(last.value).toBeLessThan(30)
  })

  it('each result has time and value fields', () => {
    const candles = makeCandlesAscending(20)
    const result = calculateRSI(candles)
    expect(result[0]).toHaveProperty('time')
    expect(result[0]).toHaveProperty('value')
  })
})

describe('calculateMACD', () => {
  it('returns empty array for fewer than 27 candles', () => {
    const candles = makeCandlesAscending(20)
    expect(calculateMACD(candles)).toHaveLength(0)
  })

  it('each result has time, macd, signal, histogram', () => {
    const candles = makeCandlesAscending(50)
    const result = calculateMACD(candles)
    expect(result[0]).toHaveProperty('time')
    expect(result[0]).toHaveProperty('macd')
    expect(result[0]).toHaveProperty('signal')
    expect(result[0]).toHaveProperty('histogram')
  })

  it('histogram equals macd minus signal', () => {
    const candles = makeCandlesAscending(50)
    const result = calculateMACD(candles)
    result.forEach((r) => {
      expect(Math.abs(r.histogram - (r.macd - r.signal))).toBeLessThan(0.0001)
    })
  })
})
