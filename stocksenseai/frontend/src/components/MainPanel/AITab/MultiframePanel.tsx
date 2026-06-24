import type { MultiframeSignal } from '@/types'
import { cn } from '@/lib/utils'

interface MultiframePanelProps {
  signals: MultiframeSignal[]
}

const signalColor = {
  BUY:  'text-green-400',
  HOLD: 'text-gray-400',
  SELL: 'text-red-400',
}

export function MultiframePanel({ signals }: MultiframePanelProps) {
  return (
    <div className="p-4 rounded-lg bg-card border border-border">
      <h3 className="text-sm font-semibold text-foreground mb-3">멀티 타임프레임</h3>
      <div className="grid grid-cols-3 gap-2">
        {signals.map((s) => (
          <div key={s.timeframe} className="text-center p-3 rounded bg-background border border-border">
            <div className="text-xs text-muted-foreground mb-1">{s.timeframe}</div>
            <div className={cn('text-base font-bold', signalColor[s.signal])}>{s.signal}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{s.score}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
