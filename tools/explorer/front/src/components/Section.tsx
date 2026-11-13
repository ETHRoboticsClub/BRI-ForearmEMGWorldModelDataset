import { ChevronDown } from 'lucide-react'
import type { ReactNode } from 'react'

interface Props {
  title: string
  collapsed?: boolean
  onToggle?: () => void
  actions?: ReactNode
  children: ReactNode
}

/** A titled block whose header folds it away. */
export function Section({ title, collapsed, onToggle, actions, children }: Props) {
  return (
    <section className="flex h-full flex-col bg-panel">
      <header className="flex h-8 shrink-0 items-center gap-2 border-b border-line pr-2 pl-1">
        <button onClick={onToggle} className="flex items-center gap-1 rounded px-1 py-0.5 text-xs font-medium hover:bg-raised">
          <ChevronDown size={13} className={`text-muted transition-transform ${collapsed ? '-rotate-90' : ''}`} />
          {title}
        </button>
        {!collapsed && <div className="ml-auto flex items-center gap-2">{actions}</div>}
      </header>
      {!collapsed && <div className="relative min-h-0 flex-1">{children}</div>}
    </section>
  )
}
