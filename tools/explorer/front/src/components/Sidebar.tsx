import { ChevronRight, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import type { Details, Summary } from '../lib/api'
import { duration } from '../lib/format'
import { useExplorer } from '../store'

function SessionRow({ session, active, onOpen }: { session: Summary; active: boolean; onOpen: () => void }) {
  const amputee = session.condition.startsWith('amputee')
  return (
    <button onClick={onOpen}
      className={`flex w-full items-center gap-2 rounded-md px-2 py-1 text-left ${active ? 'bg-accent-soft text-text' : 'hover:bg-raised'}`}>
      <span className="font-mono text-[11.5px]">{session.subject.split('/')[1]}</span>
      <span className="font-mono text-[11.5px] text-muted">{session.session}</span>
      {amputee && <span className="size-1.5 rounded-full bg-[var(--c-2)]" title="amputee" />}
      <span className="ml-auto text-[11px] text-muted">{duration(session.duration_s)}</span>
    </button>
  )
}

function SessionCard({ details }: { details: Details }) {
  const rows: [string, string][] = [
    ['EMG', `${details.n_channels} ch · ${details.electrode_type} · ${details.native_fs / 1000} kHz`],
    ['labels', `${details.label_source} · ${details.measured_joints.length}/22 joints`],
    ['task', details.stimuli.length ? `${details.stimuli.length} movements` : 'free motion'],
  ]
  return (
    <div className="space-y-1 border-t border-line p-3 text-[11.5px]">
      {rows.map(([label, value]) => (
        <div key={label} className="flex gap-2"><span className="w-12 shrink-0 text-muted">{label}</span>{value}</div>
      ))}
      <a href={details.url} target="_blank" rel="noreferrer" title={details.device}
        className="block truncate pt-1 text-muted hover:text-accent">{details.citation}</a>
    </div>
  )
}

/** The sessions on disk, by dataset, and what is known about the open one. */
export function Sidebar() {
  const { sessions, key, details, open } = useExplorer(useShallow((s) => ({ sessions: s.sessions, key: s.key, details: s.details, open: s.open })))
  const [query, setQuery] = useState('')
  const [closed, setClosed] = useState<Set<string>>(new Set())

  const groups = useMemo(() => {
    const out = new Map<string, Summary[]>()
    for (const s of sessions.filter((s) => s.key.includes(query.trim()))) out.set(s.dataset, [...(out.get(s.dataset) ?? []), s])
    return [...out.entries()]
  }, [sessions, query])

  const toggle = (dataset: string) =>
    setClosed((c) => { const next = new Set(c); if (next.has(dataset)) next.delete(dataset); else next.add(dataset); return next })

  return (
    <aside className="flex h-full flex-col bg-panel">
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-line px-3">
        <Search size={13} className="text-muted" />
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Filter sessions"
          className="min-w-0 flex-1 bg-transparent text-xs outline-none placeholder:text-muted" />
      </div>
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-2">
        {groups.map(([dataset, list]) => (
          <div key={dataset}>
            <button onClick={() => toggle(dataset)} className="flex w-full items-center gap-1 px-1 py-1 text-xs font-medium">
              <ChevronRight size={13} className={`text-muted transition-transform ${closed.has(dataset) ? '' : 'rotate-90'}`} />
              {dataset}
              <span className="ml-auto text-[11px] font-normal text-muted">{list.length}</span>
            </button>
            {!closed.has(dataset) && list.map((s) => <SessionRow key={s.key} session={s} active={s.key === key} onOpen={() => open(s.key)} />)}
          </div>
        ))}
        {sessions.length === 0 && <p className="p-2 text-muted">No sessions in the data folder. Build some with scripts/build_dataset.py.</p>}
      </div>
      {details && <SessionCard details={details} />}
    </aside>
  )
}
