/** 75.3 -> "01:15.30" */
export function clock(seconds: number) {
  const s = Math.max(0, seconds)
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(2).padStart(5, '0')}`
}

export function duration(seconds: number) {
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`
  return `${(seconds / 3600).toFixed(1)} h`
}
