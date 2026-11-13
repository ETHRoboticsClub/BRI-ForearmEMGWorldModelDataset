import { useCallback } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { valueAt } from '../lib/api'
import { MONO, useCanvas } from '../lib/canvas'
import { jointStyle, shortName } from '../lib/joints'
import { playback, usePlayback } from '../lib/playback'
import { token } from '../lib/theme'
import { useExplorer } from '../store'

/** Every canonical joint at the cursor: a bar from open (0) to closed (1) and the angle in degrees. */
const ROW_HEIGHT = 17

export function JointBars() {
  const { data, details, kinScale } = useExplorer(useShallow((s) => ({ data: s.data, details: s.details, kinScale: s.kinScale })))

  const [ref, redraw] = useCanvas((ctx, w, h) => {
    if (!details) return
    const rowHeight = h / details.joints.length
    const labelWidth = 74
    const valueWidth = 44
    ctx.font = MONO
    details.joints.forEach((name, j) => {
      const y = j * rowHeight
      const row = data?.kin.joints.indexOf(j) ?? -1
      const value = row >= 0 && data ? valueAt(data.kin, data.kin.values[row], playback.time) : undefined
      const low = details.kin_stats.low[j]
      const high = details.kin_stats.high[j]
      const unit = value === undefined ? undefined
        : kinScale === 'unit' ? value : low !== null && high !== null ? (value - low) / (high - low) : undefined
      const degrees = value === undefined || kinScale === 'unit' || low === null || high === null ? undefined
        : (value * 180) / Math.PI

      ctx.fillStyle = row >= 0 ? token('text') : token('muted')
      ctx.globalAlpha = row >= 0 ? 1 : 0.5
      ctx.fillText(shortName(name), 4, y + rowHeight * 0.7)
      ctx.fillStyle = token('raised')
      ctx.fillRect(labelWidth, y + rowHeight * 0.3, w - labelWidth - valueWidth, rowHeight * 0.4)
      if (unit !== undefined) {
        ctx.fillStyle = token(jointStyle(name).color)
        const clamped = Math.min(Math.max(unit, 0), 1)
        ctx.fillRect(labelWidth, y + rowHeight * 0.3, clamped * (w - labelWidth - valueWidth), rowHeight * 0.4)
        ctx.fillStyle = token('muted')
        ctx.textAlign = 'right'
        ctx.fillText(degrees !== undefined ? `${degrees.toFixed(0)}°` : unit.toFixed(2), w - 4, y + rowHeight * 0.7)
        ctx.textAlign = 'left'
      }
      ctx.globalAlpha = 1
    })
  })
  usePlayback(useCallback(() => redraw(), [redraw]))
  return <canvas ref={ref} className="w-full" style={{ height: (details?.joints.length ?? 22) * ROW_HEIGHT }} />
}
