import { useEffect, useRef } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { MONO, ticks, useCanvas } from '../lib/canvas'
import { playback } from '../lib/playback'
import { token } from '../lib/theme'
import { useExplorer } from '../store'
import { Cursor } from './Cursor'
import { stimulusColor } from './EmgPlot'

/** The whole session: EMG activity (grey), hand closure (accent), instructed movements (ticks) and the view (box). */
export function Navigator() {
  const { overview, details, view } = useExplorer(useShallow((s) => ({ overview: s.overview, details: s.details, view: s.view })))
  const box = useRef<HTMLDivElement>(null)
  const duration = details?.duration_s ?? 1

  const [ref] = useCanvas((ctx, w, h) => {
    if (!overview) return
    const x = (t: number) => (t / duration) * w
    const plotBottom = h - 16

    for (const [start, end, id] of overview.stimulus) {
      ctx.fillStyle = stimulusColor(id)
      ctx.fillRect(x(start), plotBottom - 3, Math.max(1, x(end) - x(start)), 3)
    }

    ctx.fillStyle = token('muted')
    ctx.globalAlpha = 0.35
    ctx.beginPath()
    ctx.moveTo(0, plotBottom - 4)
    overview.envelope.forEach((v, i) => ctx.lineTo(x(i * overview.dt), plotBottom - 4 - Math.min(v, 1) * (plotBottom - 10)))
    ctx.lineTo(x(overview.envelope.length * overview.dt), plotBottom - 4)
    ctx.fill()
    ctx.globalAlpha = 1

    ctx.strokeStyle = token('accent')
    ctx.lineWidth = 1.2
    ctx.beginPath()
    overview.flexion.forEach((v, i) => {
      const px = x(i * overview.dt)
      const py = plotBottom - 4 - Math.min(Math.max(v, 0), 1) * (plotBottom - 10)
      if (i) ctx.lineTo(px, py)
      else ctx.moveTo(px, py)
    })
    ctx.stroke()

    ctx.font = MONO
    ctx.fillStyle = token('muted')
    for (const t of ticks(0, duration, w / 120)) ctx.fillText(t >= 60 ? `${+(t / 60).toFixed(1)} min` : `${t} s`, x(t) + 2, h - 4)

    const a = x(view.start)
    const b = Math.max(x(view.end), a + 3)
    ctx.fillStyle = token('accent-soft')
    ctx.globalAlpha = 0.55
    ctx.fillRect(a, 2, b - a, plotBottom - 2)
    ctx.globalAlpha = 1
    ctx.strokeStyle = token('accent')
    ctx.strokeRect(a + 0.5, 2.5, b - a - 1, plotBottom - 3)
  })

  useEffect(() => {
    const el = box.current
    if (!el) return
    let drag: { x: number; start: number; end: number } | null = null
    const timeAt = (clientX: number) => {
      const r = el.getBoundingClientRect()
      return ((clientX - r.left) / r.width) * (useExplorer.getState().details?.duration_s ?? 1)
    }
    const down = (e: PointerEvent) => {
      const { view, setView } = useExplorer.getState()
      const t = timeAt(e.clientX)
      if (t < view.start || t > view.end) {
        const span = view.end - view.start
        setView(t - span / 2, t + span / 2)
        playback.seek(t)
      }
      const v = useExplorer.getState().view
      drag = { x: e.clientX, start: v.start, end: v.end }
      el.setPointerCapture(e.pointerId)
    }
    const move = (e: PointerEvent) => {
      if (!drag) return
      const dt = timeAt(e.clientX) - timeAt(drag.x)
      useExplorer.getState().setView(drag.start + dt, drag.end + dt)
    }
    const up = () => { drag = null }
    el.addEventListener('pointerdown', down)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    return () => {
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
    }
  }, [])

  return (
    <div ref={box} className="relative h-full cursor-grab touch-none select-none active:cursor-grabbing">
      <canvas ref={ref} className="absolute inset-0 size-full" />
      <Cursor full={{ duration }} />
    </div>
  )
}
