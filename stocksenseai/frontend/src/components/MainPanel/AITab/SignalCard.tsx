import { cn } from '@/lib/utils'

interface SignalCardProps {
  signal: 'BUY' | 'HOLD' | 'SELL'
  signal_score: number
  confidence: number
}

const signalStyles = {
  BUY:  { text: 'text-green-400', bg: 'bg-green-400/10', border: 'border-green-400/30' },
  HOLD: { text: 'text-gray-400',  bg: 'bg-gray-400/10',  border: 'border-gray-400/30' },
  SELL: { text: 'text-red-400',   bg: 'bg-red-400/10',   border: 'border-red-400/30' },
}

export function SignalCard({ signal, signal_score, confidence }: SignalCardProps) {
  const styles = signalStyles[signal]

  return (
    <div className={cn('rounded-lg border p-4 text-center', styles.bg, styles.border)}>
      <div className={cn('text-3xl font-bold mb-1', styles.text)}>{signal}</div>
      <div className="text-2xl font-semibold text-foreground mb-1">{signal_score}<span className="text-sm text-muted-foreground">/100</span></div>
      <div className="text-xs text-muted-foreground">신뢰도 {Math.round(confidence * 100)}%</div>
    </div>
  )
}
