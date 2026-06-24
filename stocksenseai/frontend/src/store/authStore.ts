// frontend/src/store/authStore.ts
import { create } from 'zustand'
import type { User } from '@/types'
import api from '@/lib/api'
import { useAuthTokenRef } from '@/lib/api'

interface AuthState {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  initAuth: () => Promise<void>
  setUser: (user: User) => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isLoading: false,

  login: async (email, password) => {
    set({ isLoading: true })
    try {
      const { data } = await api.post('/auth/login', { email, password })
      useAuthTokenRef.setToken(data.access_token)
      const { data: user } = await api.get('/auth/me')
      set({ user, isLoading: false })
    } catch (e) {
      set({ isLoading: false })
      throw e
    }
  },

  logout: async () => {
    try {
      await api.post('/auth/logout')
    } finally {
      useAuthTokenRef.clearToken()
      set({ user: null })
    }
  },

  initAuth: async () => {
    try {
      const { data: refreshData } = await api.post('/auth/refresh', {}, { withCredentials: true })
      useAuthTokenRef.setToken(refreshData.access_token)
      const { data: user } = await api.get('/auth/me')
      set({ user })
    } catch {
      // no valid session
    }
  },

  setUser: (user) => set({ user }),
}))
