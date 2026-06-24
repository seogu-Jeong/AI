// frontend/src/components/Layout/Header.tsx
import { Search, Sun, Moon, User, Settings, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useUIStore } from '@/store/uiStore'
import { useAuthStore } from '@/store/authStore'
import { useStockStore } from '@/store/stockStore'
import { MOCK_STOCKS } from '@/lib/mockData'
import { useState } from 'react'
import { RiskSettingsModal } from '@/components/Risk/RiskSettingsModal'
import { SystemStatusPanel } from '@/components/System/SystemStatusPanel'

export function Header({ onLoginClick, onAccountClick }: { onLoginClick: () => void; onAccountClick?: () => void }) {
  const { darkMode, toggleDarkMode } = useUIStore()
  const { user } = useAuthStore()
  const { setSelectedStock } = useStockStore()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(MOCK_STOCKS.slice(0, 0))
  const [riskOpen, setRiskOpen] = useState(false)
  const [statusOpen, setStatusOpen] = useState(false)

  const handleSearch = (q: string) => {
    setQuery(q)
    setResults(
      q.length > 0
        ? MOCK_STOCKS.filter(
            (s) => s.name.includes(q) || s.code.includes(q)
          ).slice(0, 5)
        : []
    )
  }

  return (
    <header className="flex items-center justify-between px-4 h-12 bg-card border-b border-border shrink-0">
      <span className="text-primary font-bold text-lg tracking-tight">StockSenseAI</span>

      <div className="relative w-64">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          className="pl-8 h-8 text-sm"
          placeholder="종목명 또는 코드"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onBlur={() => setTimeout(() => setResults([]), 200)}
        />
        {results.length > 0 && (
          <ul className="absolute top-9 left-0 w-full bg-popover border border-border rounded-md shadow-lg z-50">
            {results.map((s) => (
              <li
                key={s.code}
                className="px-3 py-2 text-sm cursor-pointer hover:bg-accent"
                onMouseDown={() => {
                  setSelectedStock(s)
                  setQuery('')
                  setResults([])
                }}
              >
                <span className="font-medium">{s.name}</span>
                <span className="ml-2 text-muted-foreground text-xs">{s.code}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={() => setStatusOpen(true)} aria-label="시스템 상태">
          <Activity className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={() => setRiskOpen(true)} aria-label="리스크 설정">
          <Settings className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={toggleDarkMode} aria-label="테마 전환">
          {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
        {user ? (
          <Button variant="ghost" size="sm" onClick={onAccountClick} className="text-sm text-muted-foreground gap-1.5">
            <User className="h-4 w-4" />
            <span className="max-w-32 truncate">{user.email}</span>
          </Button>
        ) : (
          <Button variant="outline" size="sm" onClick={onLoginClick}>
            <User className="h-4 w-4 mr-1" /> 로그인
          </Button>
        )}
      </div>
      <RiskSettingsModal open={riskOpen} onClose={() => setRiskOpen(false)} />
      {statusOpen && <SystemStatusPanel onClose={() => setStatusOpen(false)} />}
    </header>
  )
}
