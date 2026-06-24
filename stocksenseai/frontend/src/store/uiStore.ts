// frontend/src/store/uiStore.ts
import { create } from 'zustand'
import type { TabId } from '@/types'

interface UIState {
  darkMode: boolean
  activeTab: TabId
  sidebarOpen: boolean
  /** 주문 성공 시 Date.now() 타임스탬프. 변경될 때마다 RecentTradesPanel이 갱신된다. */
  lastOrderAt: number | undefined
  toggleDarkMode: () => void
  setActiveTab: (tab: TabId) => void
  toggleSidebar: () => void
  notifyOrderPlaced: () => void
}

export const useUIStore = create<UIState>((set) => ({
  darkMode: true,
  activeTab: 'chart',
  sidebarOpen: true,
  lastOrderAt: undefined,

  toggleDarkMode: () =>
    set((state) => {
      const next = !state.darkMode
      document.documentElement.classList.toggle('dark', next)
      return { darkMode: next }
    }),

  setActiveTab: (tab) => set({ activeTab: tab }),

  toggleSidebar: () =>
    set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  notifyOrderPlaced: () => set({ lastOrderAt: Date.now() }),
}))
