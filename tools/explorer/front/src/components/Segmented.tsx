interface Props<T extends string | number> {
  value: T
  options: { value: T; label: string; title?: string }[]
  onChange: (value: T) => void
}

/** A row of exclusive buttons, like the studio's mode switch. */
export function Segmented<T extends string | number>({ value, options, onChange }: Props<T>) {
  return (
    <div className="flex rounded-md border border-line bg-bg p-0.5">
      {options.map((o) => (
        <button key={String(o.value)} title={o.title} onClick={() => onChange(o.value)}
          className={`rounded px-2 py-0.5 text-xs ${o.value === value ? 'bg-panel text-text shadow-sm' : 'text-muted hover:text-text'}`}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
