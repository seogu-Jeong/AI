// frontend/src/components/Layout/MainLayout.tsx
import { Header } from './Header'
import { MobileTabBar } from './MobileTabBar'
import { Sidebar } from '@/components/Sidebar/Sidebar'
import { MainPanel } from '@/components/MainPanel/MainPanel'
import { WatchlistPanel } from '@/components/WatchlistPanel/WatchlistPanel'
import { useUIStore } from '@/store/uiStore'
import { useState, useEffect } from 'react'
import { LoginModal } from '@/components/auth/LoginModal'
import { RegisterModal } from '@/components/auth/RegisterModal'
import { OrderBook } from '@/components/Trade/OrderBook'
import { MOCK_ORDER_BOOK } from '@/lib/mockData'
import { useStockStore } from '@/store/stockStore'
import { AccountPanel } from '@/components/Account/AccountPanel'

export function MainLayout() {
  const { sidebarOpen } = useUIStore()
  const [modal, setModal] = useState<'login' | 'register' | null>(null)
  const [accountOpen, setAccountOpen] = useState(false)
  const { selectedStock, realtimePrice, loadStocks } = useStockStore()

  useEffect(() => { loadStocks() }, [loadStocks])

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <Header onLoginClick={() => setModal('login')} onAccountClick={() => setAccountOpen(true)} />
      <MobileTabBar />

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {sidebarOpen && (
          <div className="hidden md:flex md:flex-col min-h-0">
            <Sidebar />
          </div>
        )}
        <MainPanel />
        {/* 우측 패널 — 호가창 */}
        <div className="hidden lg:flex lg:flex-col w-56 shrink-0 bg-card border-l border-border overflow-hidden">
          <OrderBook
            asks={MOCK_ORDER_BOOK.asks}
            bids={MOCK_ORDER_BOOK.bids}
            currentPrice={realtimePrice?.price ?? selectedStock?.price ?? 73400}
          />
        </div>
      </div>

      <div className="hidden md:block">
        <WatchlistPanel />
      </div>

      <LoginModal
        open={modal === 'login'}
        onClose={() => setModal(null)}
        onRegister={() => setModal('register')}
      />
      <RegisterModal
        open={modal === 'register'}
        onClose={() => setModal(null)}
        onLogin={() => setModal('login')}
      />

      {accountOpen && (
        <>
          <div
            className="fixed inset-0 z-40 bg-black/30"
            onClick={() => setAccountOpen(false)}
          />
          <AccountPanel onClose={() => setAccountOpen(false)} />
        </>
      )}
    </div>
  )
}
