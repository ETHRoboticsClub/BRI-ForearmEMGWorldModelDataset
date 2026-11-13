import { Check, ChevronDown } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

interface Props {
  value?: string
  options: { value: string; label: string }[]
  onChange: (value: string) => void
}

/** A compact dropdown for a section header. */
export function Menu({ value, options, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: PointerEvent) => !root.current?.contains(e.target as Node) && setOpen(false)
    addEventListener('pointerdown', close)
    return () => removeEventListener('pointerdown', close)
  }, [open])
  return (
    <div ref={root} className="relative">
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-1 rounded-md border border-line bg-bg px-2 py-0.5 text-xs text-muted hover:text-text">
        {options.find((o) => o.value === value)?.label ?? '—'} <ChevronDown size={12} />
      </button>
      {open && (
        <div className="absolute top-7 right-0 z-20 min-w-44 rounded-md border border-line bg-panel p-1 shadow-lg">
          {options.map((o) => (
            <button key={o.value} onClick={() => { onChange(o.value); setOpen(false) }}
              className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs hover:bg-raised">
              <span className="w-3">{o.value === value && <Check size={12} />}</span>{o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
