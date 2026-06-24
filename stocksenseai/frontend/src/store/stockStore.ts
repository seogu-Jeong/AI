// frontend/src/store/stockStore.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Stock, RealtimePrice } from '@/types'
import { MOCK_STOCKS, MOCK_WATCHLIST } from '@/lib/mockData'
import api from '@/lib/api'

interface StockState {
  selectedStock: Stock | null
  watchlist: string[]
  stockList: Stock[]
  realtimePrice: RealtimePrice | null
  setSelectedStock: (stock: Stock) => void
  addToWatchlist: (code: string) => void
  removeFromWatchlist: (code: string) => void
  updateRealtimePrice: (data: RealtimePrice) => void
  loadStocks: () => Promise<void>
  searchStocks: (q: string) => Promise<Stock[]>
}

export const useStockStore = create<StockState>()(
  persist(
    (set) => ({
  selectedStock: MOCK_STOCKS[0],
  watchlist: MOCK_WATCHLIST,
  stockList: MOCK_STOCKS,
  realtimePrice: null,

  setSelectedStock: (stock) => set({ selectedStock: stock, realtimePrice: null }),

  addToWatchlist: (code) =>
    set((state) => ({
      watchlist: state.watchlist.includes(code)
        ? state.watchlist
        : [...state.watchlist, code],
    })),

  removeFromWatchlist: (code) =>
    set((state) => ({
      watchlist: state.watchlist.filter((c) => c !== code),
    })),

  updateRealtimePrice: (data) => set({ realtimePrice: data }),

  loadStocks: async () => {
    try {
      const { data } = await api.get('/stocks', { params: { limit: 100 } })
      const stocks: Stock[] = (data.stocks ?? data ?? []).map((s: { code: string; name: string; close?: number; change_pct?: number }) => ({
        code: s.code,
        name: s.name,
        price: s.close,
        change_pct: s.change_pct,
      }))
      if (stocks.length > 0) {
        set({ stockList: stocks, selectedStock: stocks[0] })
      }
    } catch {
      // fallback to mock data
    }
  },

  searchStocks: async (q: string): Promise<Stock[]> => {
    try {
      const { data } = await api.get('/stocks/search', { params: { q } })
      return (data.stocks ?? data ?? []).map((s: { code: string; name: string; close?: number; change_pct?: number }) => ({
        code: s.code,
        name: s.name,
        price: s.close,
        change_pct: s.change_pct,
      }))
    } catch {
      return []
    }
  },
    }),
    {
      name: 'stocksense-watchlist',
      partialize: (state) => ({ watchlist: state.watchlist }),
    }
  )
)
