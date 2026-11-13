import { Loader2, Maximize2, Pause, Play, ZoomIn, ZoomOut } from 'lucide-react'
import { useCallback, useRef } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { clock } from '../lib/format'
import { playback, usePlayback, usePlayState } from '../lib/playback'
import { useExplorer } from '../store'
import { Segmented } from './Segmented'

const SPEEDS = [0.25, 0.5, 1, 2, 4]
const icon = 'grid size-7 shrink-0 place-items-center rounded-md text-muted hover:bg-raised hover:text-text'

function Clock({ total }: { total: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  usePlayback(useCallback((t: number) => { if (ref.current) ref.current.textContent = clock(t) }, []))
  return <span className="font-mono text-xs tabular-nums"><span ref={ref}>{clock(playback.time)}</span><span className="text-muted"> / {clock(total)}</span></span>
}

export function Toolbar() {
  const { playing, speed } = usePlayState()
  const { details, view, zoom, setView, loading, error } = useExplorer(useShallow((s) => ({
    details: s.details, view: s.view, zoom: s.zoom, setView: s.setView, loading: s.loading, error: s.error,
  })))
  const center = (view.start + view.end) / 2
  return (
    <div className="flex h-10 shrink-0 items-center gap-2 overflow-x-auto whitespace-nowrap border-b border-line bg-panel px-2">
      <button onClick={playback.toggle} title="Play / pause (space)"
        className="grid size-7 shrink-0 place-items-center rounded-md bg-accent text-white dark:text-bg">
        {playing ? <Pause size={13} /> : <Play size={13} />}
      </button>
      <Clock total={details?.duration_s ?? 0} />
      <Segmented value={speed} onChange={(v) => usePlayState.setState({ speed: v })}
        options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))} />
      {error && <span className="truncate text-xs text-danger">{error}</span>}
      <div className="ml-auto flex items-center">
        {loading && <Loader2 size={13} className="mr-1 animate-spin text-muted" />}
        <button className={icon} title="Zoom in (+)" onClick={() => zoom(0.5, center)}><ZoomIn size={14} /></button>
        <button className={icon} title="Zoom out (−)" onClick={() => zoom(2, center)}><ZoomOut size={14} /></button>
        <button className={icon} title="10 s around the cursor" onClick={() => setView(playback.time - 2, playback.time + 8)}><Maximize2 size={13} /></button>
      </div>
    </div>
  )
}
