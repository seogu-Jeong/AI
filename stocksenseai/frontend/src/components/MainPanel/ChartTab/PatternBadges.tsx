import type { CandlePattern } from '@/types'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface PatternBadgesProps {
  patterns: CandlePattern[]
}

const typeStyles: Record<CandlePattern['type'], string> = {
  bullish: 'text-green-400 bg-green-400/10 border-green-400/30',
  bearish: 'text-red-400 bg-red-400/10 border-red-400/30',
  neutral: 'text-gray-400 bg-gray-400/10 border-gray-400/30',
}

export function PatternBadges({ patterns }: PatternBadgesProps) {
  if (patterns.length === 0) return null

  return (
    <TooltipProvider>
      <div className="flex items-center gap-1.5 px-4 py-1.5 border-b border-border">
        <span className="text-xs text-muted-foreground mr-1">패턴:</span>
        {patterns.slice(0, 5).map((p) => (
          <Tooltip key={p.name}>
            <TooltipTrigger asChild>
              <span
                className={cn(
                  'text-xs px-2 py-0.5 rounded-full border cursor-help font-medium',
                  typeStyles[p.type]
                )}
              >
                {p.name}
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-xs max-w-48">{p.description}</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  )
}
