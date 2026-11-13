import { useRef } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { MONO, ticks, useCanvas } from '../lib/canvas'
import { PAD, timeScale, usePlotControls } from '../lib/interaction'
import { jointStyle } from '../lib/joints'
import { token } from '../lib/theme'
import { useExplorer } from '../store'
import { Cursor } from './Cursor'
import { drawStimulus, drawTimeAxis } from './EmgPlot'

/** The chosen joints over time: colour = finger, dash = segment (solid MCP, dashed PIP, dotted DIP). */
export function KinPlot() {
  const { data, view, details, joints, kinScale } = useExplorer(useShallow((s) => ({
    data: s.data, view: s.view, details: s.details, joints: s.joints, kinScale: s.kinScale,
  })))
  const box = useRef<HTMLDivElement>(null)
  usePlotControls(box)

  const [ref] = useCanvas((ctx, w, h) => {
    if (!data || !details) return
    const { x } = timeScale(w, view)
    const top = PAD.top
    const bottom = h - PAD.bottom
    drawStimulus(ctx, data.stimulus, x, top, bottom)
    drawTimeAxis(ctx, w, h, view)

    const { kin } = data
    const i0 = Math.max(0, Math.floor((view.start - kin.t0) / kin.dt))
    const i1 = Math.min(kin.values[0]?.length ?? 0, Math.ceil((view.end - kin.t0) / kin.dt) + 1)
    const rows = joints.map((j) => kin.joints.indexOf(j)).filter((r) => r >= 0)
    let low = kinScale === 'unit' ? -0.1 : Infinity
    let high = kinScale === 'unit' ? 1.1 : -Infinity
    if (kinScale === 'rad') {
      for (const r of rows) for (let i = i0; i < i1; i++) { low = Math.min(low, kin.values[r][i]); high = Math.max(high, kin.values[r][i]) }
      if (!isFinite(low)) { low = 0; high = 1 }
      const margin = (high - low) * 0.08 || 0.1
      low -= margin; high += margin
    }
    const y = (v: number) => bottom - ((v - low) / (high - low)) * (bottom - top)

    ctx.font = MONO
    ctx.fillStyle = token('muted')
    ctx.strokeStyle = token('grid')
    for (const v of ticks(low, high, Math.max(2, (bottom - top) / 40))) {
      const py = Math.round(y(v)) + 0.5
      ctx.beginPath(); ctx.moveTo(PAD.left, py); ctx.lineTo(w - PAD.right, py); ctx.stroke()
      ctx.fillText(kinScale === 'unit' ? v.toFixed(1) : `${+v.toFixed(2)}`, 8, py + 4)
    }

    ctx.save()
    ctx.beginPath(); ctx.rect(PAD.left, top, w - PAD.left - PAD.right, bottom - top); ctx.clip()
    for (const r of rows) {
      const style = jointStyle(details.joints[kin.joints[r]])
      ctx.strokeStyle = token(style.color)
      ctx.setLineDash(style.dash)
      ctx.lineWidth = 1.6
      ctx.beginPath()
      for (let i = i0; i < i1; i++) {
        const px = x(kin.t0 + i * kin.dt)
        if (i === i0) ctx.moveTo(px, y(kin.values[r][i]))
        else ctx.lineTo(px, y(kin.values[r][i]))
      }
      ctx.stroke()
    }
    ctx.restore()
    ctx.setLineDash([])
  })

  if (details && details.measured_joints.length === 0)
    return <p className="grid size-full place-items-center text-muted">This session has no kinematics (cue only).</p>

  return (
    <div ref={box} className="absolute inset-0 cursor-crosshair touch-none select-none">
      <canvas ref={ref} className="absolute inset-0 size-full" />
      <Cursor />
    </div>
  )
}
