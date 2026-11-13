import { useCallback } from 'react'
import { useCanvas } from '../lib/canvas'
import { PAD, timeScale } from '../lib/interaction'
import { playback, usePlayback } from '../lib/playback'
import { token } from '../lib/theme'
import { useExplorer } from '../store'

/** The playback cursor over a plot, on its own canvas: moving it does not redraw the signals. */
export function Cursor({ full }: { full?: { duration: number } }) {
  const view = useExplorer((s) => s.view)
  const [ref, redraw] = useCanvas((ctx, w, h) => {
    const range = full ? { start: 0, end: full.duration } : view
    const pad = full ? { left: 0, right: 0 } : PAD
    const x = full
      ? (playback.time / full.duration) * w
      : timeScale(w, range).x(playback.time)
    if (x < pad.left - 1 || x > w - pad.right + 1) return
    ctx.strokeStyle = token('accent')
    ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke()
  })
  usePlayback(useCallback(() => redraw(), [redraw]))
  return <canvas ref={ref} className="pointer-events-none absolute inset-0 size-full" />
}
