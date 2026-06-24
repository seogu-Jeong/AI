// frontend/src/components/Layout/MobileTabBar.tsx
import { useUIStore } from '@/store/uiStore'
import type { TabId } from '@/types'
import { cn } from '@/lib/utils'

const TABS: { id: TabId; label: string }[] = [
  { id: 'chart', label: '차트' },
  { id: 'ai', label: 'AI' },
  { id: 'simulator', label: '시뮬' },
  { id: 'portfolio', label: '포트폴리오' },
  { id: 'screener', label: '스크리너' },
  { id: 'backtest', label: '백테스트' },
]

export function MobileTabBar() {
  const { activeTab, setActiveTab } = useUIStore()

  return (
    <div className="md:hidden border-b border-border bg-card overflow-x-auto">
      <div className="flex h-9">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'px-4 h-full text-sm whitespace-nowrap shrink-0 border-b-2 transition-colors',
              activeTab === tab.id
                ? 'border-primary text-foreground font-medium'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  )
}
