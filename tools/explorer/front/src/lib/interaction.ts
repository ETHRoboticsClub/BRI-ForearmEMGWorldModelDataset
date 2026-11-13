import { useEffect, type RefObject } from 'react'
import { useExplorer } from '../store'
import { playback } from './playback'

export const PAD = { left: 52, right: 12, top: 8, bottom: 20 }

/** Time <-> x in a plot that shows the current view between PAD.left and width - PAD.right. */
export function timeScale(width: number, view: { start: number; end: number }) {
  const span = view.end - view.start
  const inner = width - PAD.left - PAD.right
  return {
    x: (t: number) => PAD.left + ((t - view.start) / span) * inner,
    t: (x: number) => view.start + ((x - PAD.left) / inner) * span,
  }
}

/** Wheel zooms around the pointer (shift or a horizontal wheel pans), drag pans, a click moves the cursor. */
export function usePlotControls(ref: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const timeAt = (clientX: number) => {
      const box = el.getBoundingClientRect()
      return timeScale(box.width, useExplorer.getState().view).t(clientX - box.left)
    }
    const wheel = (e: WheelEvent) => {
      e.preventDefault()
      const { view, pan, zoom } = useExplorer.getState()
      const horizontal = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)
      if (horizontal) pan(((e.deltaX || e.deltaY) / 600) * (view.end - view.start))
      else zoom(Math.exp(e.deltaY * 0.0015), timeAt(e.clientX))
    }
    let drag: { x: number; start: number; end: number; moved: boolean } | null = null
    const down = (e: PointerEvent) => {
      const { view } = useExplorer.getState()
      drag = { x: e.clientX, start: view.start, end: view.end, moved: false }
      el.setPointerCapture(e.pointerId)
    }
    const move = (e: PointerEvent) => {
      if (!drag) return
      const dx = e.clientX - drag.x
      if (Math.abs(dx) > 3) drag.moved = true
      if (!drag.moved) return
      const width = el.getBoundingClientRect().width - PAD.left - PAD.right
      const dt = (dx / width) * (drag.end - drag.start)
      useExplorer.getState().setView(drag.start - dt, drag.end - dt)
    }
    const up = (e: PointerEvent) => {
      if (drag && !drag.moved) playback.seek(timeAt(e.clientX))
      drag = null
    }
    el.addEventListener('wheel', wheel, { passive: false })
    el.addEventListener('pointerdown', down)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    return () => {
      el.removeEventListener('wheel', wheel)
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
    }
  }, [ref])
}
