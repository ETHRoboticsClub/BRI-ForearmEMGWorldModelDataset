import { ChevronDown } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { jointStyle, shortName } from '../lib/joints'
import { useExplorer } from '../store'

/** Which joints the plot draws: a small menu instead of a row of chips. */
export function JointPicker() {
  const { details, joints, toggleJoint } = useExplorer(useShallow((s) => ({ details: s.details, joints: s.joints, toggleJoint: s.toggleJoint })))
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: PointerEvent) => !root.current?.contains(e.target as Node) && setOpen(false)
    addEventListener('pointerdown', close)
    return () => removeEventListener('pointerdown', close)
  }, [open])
  if (!details) return null
  return (
    <div ref={root} className="relative">
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-1 rounded-md border border-line bg-bg px-2 py-0.5 text-xs text-muted hover:text-text">
        {joints.length} joints <ChevronDown size={12} />
      </button>
      {open && (
        <div className="absolute top-7 right-0 z-20 grid w-56 grid-cols-2 gap-0.5 rounded-md border border-line bg-panel p-1.5 shadow-lg">
          {details.measured_joints.map((j) => {
            const name = details.joints[j]
            const on = joints.includes(j)
            return (
              <button key={j} onClick={() => toggleJoint(j)} title={name}
                className={`flex items-center gap-1.5 rounded px-1.5 py-1 text-left font-mono text-[11px] ${on ? 'bg-raised text-text' : 'text-muted hover:text-text'}`}>
                <span className="size-1.5 rounded-full" style={{ background: on ? `var(--${jointStyle(name).color})` : 'var(--line)' }} />
                {shortName(name)}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
