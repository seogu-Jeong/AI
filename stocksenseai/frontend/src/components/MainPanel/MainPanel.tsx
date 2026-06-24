import { useUIStore } from '@/store/uiStore'
import { ChartTab } from './ChartTab/ChartTab'
import { AITab } from './AITab/AITab'
import { PortfolioTab } from './PortfolioTab/PortfolioTab'
import { SimulatorTab } from './SimulatorTab/SimulatorTab'
import { BacktestTab } from './BacktestTab/BacktestTab'
import { RecommendTab } from './RecommendTab/RecommendTab'
import { ScreenerTab } from './ScreenerTab/ScreenerTab'
import { MarketTab } from './MarketTab/MarketTab'
import { AutoTradePanel } from '@/components/AutoTrade/AutoTradePanel'
import { cn } from '@/lib/utils'
import type { TabId } from '@/types'

const ALL_TABS: { id: TabId; label: string }[] = [
  { id: 'chart', label: '차트' },
  { id: 'ai', label: 'AI' },
  { id: 'recommend', label: '추천' },
  { id: 'market', label: '시장' },
  { id: 'simulator', label: '시뮬' },
  { id: 'portfolio', label: '포트폴리오' },
  { id: 'screener', label: '스크리너' },
  { id: 'backtest', label: '백테스트' },
  { id: 'autotrade', label: '자동매매' },
]

export function MainPanel() {
  const { activeTab, setActiveTab } = useUIStore()

  return (
    <div className="flex-1 min-w-0 min-h-0 overflow-hidden flex flex-col">
      <div className="hidden md:flex border-b border-border bg-card shrink-0 overflow-x-auto">
        {ALL_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'px-4 h-9 text-sm whitespace-nowrap shrink-0 border-b-2 transition-colors',
              activeTab === tab.id
                ? 'border-primary text-foreground font-medium'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === 'chart'      && <ChartTab />}
        {activeTab === 'ai'         && <AITab />}
        {activeTab === 'recommend'  && <RecommendTab />}
        {activeTab === 'simulator'  && <SimulatorTab />}
        {activeTab === 'portfolio'  && <PortfolioTab />}
        {activeTab === 'screener'   && <ScreenerTab />}
        {activeTab === 'backtest'   && <BacktestTab />}
        {activeTab === 'market'     && <MarketTab />}
        {activeTab === 'autotrade'  && <AutoTradePanel />}
      </div>
    </div>
  )
}
