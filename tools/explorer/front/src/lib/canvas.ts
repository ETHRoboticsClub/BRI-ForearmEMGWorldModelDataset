import { useCallback, useEffect, useRef } from 'react'
import { useTheme } from './theme'

export type Draw = (ctx: CanvasRenderingContext2D, width: number, height: number) => void

/** A canvas sized to its box at device resolution; returns the ref and a function to redraw it. */
export function useCanvas(draw: Draw) {
  const ref = useRef<HTMLCanvasElement>(null)
  const drawRef = useRef(draw)
  const theme = useTheme((s) => s.theme)
  useEffect(() => { drawRef.current = draw })

  const redraw = useCallback(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    const { width, height } = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
    drawRef.current(ctx, width, height)
  }, [])

  useEffect(() => {
    const observer = new ResizeObserver(redraw)
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [redraw])

  useEffect(() => redraw(), [redraw, theme])
  useEffect(() => redraw())
  return [ref, redraw] as const
}

/** Round tick values between a and b, about `count` of them. */
export function ticks(a: number, b: number, count: number) {
  const raw = (b - a) / Math.max(count, 1)
  const power = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

export const MONO = '10.5px "JetBrains Mono Variable", monospace'
