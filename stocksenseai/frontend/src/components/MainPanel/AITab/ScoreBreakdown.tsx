import { Progress } from '@/components/ui/progress'

interface ScoreBreakdownProps {
  tech_score: number
  lstm_score: number
  confidence: number
}

export function ScoreBreakdown({ tech_score, lstm_score, confidence }: ScoreBreakdownProps) {
  const items = [
    { label: '기술적 지표', value: tech_score },
    { label: 'LSTM 예측', value: lstm_score },
    { label: '신뢰도', value: Math.round(confidence * 100), suffix: '%' },
  ]

  return (
    <div className="space-y-3 p-4 rounded-lg bg-card border border-border">
      <h3 className="text-sm font-semibold text-foreground">점수 분해</h3>
      {items.map(({ label, value, suffix }) => (
        <div key={label}>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-muted-foreground">{label}</span>
            <span className="text-foreground font-medium">{value}{suffix ?? ''}</span>
          </div>
          <Progress value={value} className="h-1.5" />
        </div>
      ))}
    </div>
  )
}
