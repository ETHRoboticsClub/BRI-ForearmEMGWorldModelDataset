import { useRef } from 'react'
import { useShallow } from 'zustand/react/shallow'
import type { Segment } from '../lib/api'
import { MONO, ticks, useCanvas } from '../lib/canvas'
import { PAD, timeScale, usePlotControls } from '../lib/interaction'
import { token } from '../lib/theme'
import { useExplorer } from '../store'
import { Cursor } from './Cursor'

export function stimulusColor(id: number) {
  return token(`c-${(id % 6) + 1}`)
}

/** Shaded bands where a movement was instructed, with its id. */
export function drawStimulus(ctx: CanvasRenderingContext2D, segments: Segment[], x: (t: number) => number,
  top: number, bottom: number) {
  ctx.font = MONO
  for (const [start, end, id] of segments) {
    const a = Math.max(x(start), PAD.left)
    const b = Math.min(x(end), ctx.canvas.clientWidth - PAD.right)
    if (b <= a) continue
    ctx.globalAlpha = 0.09
    ctx.fillStyle = stimulusColor(id)
    ctx.fillRect(a, top, b - a, bottom - top)
    ctx.globalAlpha = 1
  }
}

export function drawTimeAxis(ctx: CanvasRenderingContext2D, w: number, h: number, view: { start: number; end: number }) {
  const { x } = timeScale(w, view)
  ctx.font = MONO
  ctx.fillStyle = token('muted')
  ctx.strokeStyle = token('grid')
  ctx.lineWidth = 1
  for (const t of ticks(view.start, view.end, Math.max(2, w / 110))) {
    const px = Math.round(x(t)) + 0.5
    ctx.beginPath(); ctx.moveTo(px, PAD.top); ctx.lineTo(px, h - PAD.bottom); ctx.stroke()
    ctx.fillText(`${+t.toFixed(3)} s`, px + 3, h - 6)
  }
}

/** Robust half-height of the lanes: the 99th percentile of the visible |EMG|. */
function amplitude(min: number[][], max: number[][], i0: number, i1: number) {
  const values: number[] = []
  for (let c = 0; c < min.length; c++)
    for (let i = i0; i < i1; i += Math.max(1, Math.floor((i1 - i0) / 400)))
      values.push(Math.max(Math.abs(min[c][i]), Math.abs(max[c][i])))
  values.sort((a, b) => a - b)
  return values[Math.floor(values.length * 0.99)] || 1
}

/** One lane per channel: the min/max of each pixel (so no spike is lost when zoomed out) and the 100 ms envelope. */
export function EmgPlot() {
  const { data, view, emgScale } = useExplorer(useShallow((s) => ({ data: s.data, view: s.view, emgScale: s.emgScale })))
  const box = useRef<HTMLDivElement>(null)
  usePlotControls(box)

  const [ref] = useCanvas((ctx, w, h) => {
    if (!data) return
    const { x } = timeScale(w, view)
    const top = PAD.top
    const bottom = h - PAD.bottom
    drawStimulus(ctx, data.stimulus, x, top, bottom)
    drawTimeAxis(ctx, w, h, view)

    const { emg, envelope } = data
    const i0 = Math.max(0, Math.floor((view.start - emg.t0) / emg.dt))
    const i1 = Math.min(emg.min[0]?.length ?? 0, Math.ceil((view.end - emg.t0) / emg.dt) + 1)
    const j0 = Math.max(0, Math.floor((view.start - envelope.t0) / envelope.dt))
    const j1 = Math.min(envelope.values[0]?.length ?? 0, Math.ceil((view.end - envelope.t0) / envelope.dt) + 1)
    const half = emgScale === 'active' ? 3 : amplitude(emg.min, emg.max, i0, i1)
    const lane = (bottom - top) / emg.channels.length

    emg.channels.forEach((channel, c) => {
      const mid = top + lane * (c + 0.5)
      const y = (v: number) => mid - (v / half) * lane * 0.48
      ctx.save()
      ctx.beginPath(); ctx.rect(PAD.left, mid - lane / 2, w - PAD.left - PAD.right, lane); ctx.clip()

      ctx.strokeStyle = token('grid')
      ctx.beginPath(); ctx.moveTo(PAD.left, Math.round(mid) + 0.5); ctx.lineTo(w - PAD.right, Math.round(mid) + 0.5); ctx.stroke()

      ctx.strokeStyle = token('signal')
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let i = i0; i < i1; i++) {
        const px = x(emg.t0 + i * emg.dt)
        if (i === i0) ctx.moveTo(px, y(emg.min[c][i]))
        else ctx.lineTo(px, y(emg.min[c][i]))
        ctx.lineTo(px, y(emg.max[c][i]))
      }
      ctx.stroke()

      ctx.strokeStyle = token('accent')
      ctx.lineWidth = 1.4
      ctx.beginPath()
      for (let j = j0; j < j1; j++) {
        const px = x(envelope.t0 + j * envelope.dt)
        if (j === j0) ctx.moveTo(px, y(envelope.values[c][j]))
        else ctx.lineTo(px, y(envelope.values[c][j]))
      }
      ctx.stroke()
      ctx.restore()

      ctx.font = MONO
      ctx.fillStyle = token('muted')
      ctx.fillText(`${channel + 1}`, 14, mid + 4)
    })

    ctx.fillStyle = token('muted')
    ctx.textAlign = 'right'
    ctx.fillText(emgScale === 'active' ? '±3' : `±${half.toFixed(0)} µV`, w - PAD.right - 4, top + 10)
    ctx.textAlign = 'left'
  })

  return (
    <div ref={box} className="absolute inset-0 cursor-crosshair touch-none select-none">
      <canvas ref={ref} className="absolute inset-0 size-full" />
      <Cursor />
      {!data && <p className="absolute inset-0 grid place-items-center text-muted">loading…</p>}
    </div>
  )
}
