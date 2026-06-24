// frontend/src/components/Sidebar/StockGroup.tsx
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'

interface StockGroupProps {
  name: string
  children: React.ReactNode
}

export function StockGroup({ name, children }: StockGroupProps) {
  const [open, setOpen] = useState(true)
  return (
    <div>
      <button
        className="flex items-center gap-1 w-full px-3 py-1.5 text-xs font-semibold text-muted-foreground hover:text-foreground uppercase tracking-wider"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {name}
      </button>
      {open && <div>{children}</div>}
    </div>
  )
}
