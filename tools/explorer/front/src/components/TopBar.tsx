import { Moon, PanelLeft, PanelRight, Sun } from 'lucide-react'
import { useTheme } from '../lib/theme'
import { useExplorer } from '../store'

interface Props {
  sessions: { collapsed: boolean; toggle: () => void }
  hand: { collapsed: boolean; toggle: () => void }
}

const icon = 'grid size-7 place-items-center rounded-md hover:bg-raised hover:text-text'

export function TopBar({ sessions, hand }: Props) {
  const key = useExplorer((s) => s.key)
  const { theme, toggle } = useTheme()
  return (
    <header className="flex items-center gap-3 border-b border-line bg-panel px-2">
      <button onClick={sessions.toggle} title="Sessions" className={`${icon} ${sessions.collapsed ? 'text-muted' : 'text-text'}`}>
        <PanelLeft size={15} />
      </button>
      <img src="/icon.svg" className="size-5" alt="" />
      <span className="text-sm font-semibold">femg explorer</span>
      {key && <span className="font-mono text-xs text-muted">{key}</span>}
      <div className="ml-auto flex items-center gap-1 text-muted">
        <button onClick={toggle} title="Theme" className={icon}>{theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}</button>
        <button onClick={hand.toggle} title="Hand" className={`${icon} ${hand.collapsed ? 'text-muted' : 'text-text'}`}>
          <PanelRight size={15} />
        </button>
      </div>
    </header>
  )
}
