import { useEffect } from 'react'
import { create } from 'zustand'

type Listener = (t: number) => void

const listeners = new Set<Listener>()
let t = 0
let end = 0
let frame = 0
let last = 0

/** Whether it plays and how fast, for the buttons; the time itself lives in `playback`. */
export const usePlayState = create<{ playing: boolean; speed: number }>(() => ({ playing: false, speed: 1 }))

const emit = () => listeners.forEach((fn) => fn(t))

const tick = (now: number) => {
  t = Math.min(t + ((now - last) / 1000) * usePlayState.getState().speed, end)
  last = now
  emit()
  if (t >= end) return playback.pause()
  frame = requestAnimationFrame(tick)
}

/** The cursor time of the open session, outside React: canvases redraw on it without re-rendering. */
export const playback = {
  get time() { return t },
  reset(duration: number, at = 0) {
    playback.pause()
    end = duration
    t = Math.min(at, end)
    emit()
  },
  play() {
    if (usePlayState.getState().playing || end <= 0) return
    if (t >= end) t = 0
    usePlayState.setState({ playing: true })
    last = performance.now()
    frame = requestAnimationFrame(tick)
  },
  pause() {
    cancelAnimationFrame(frame)
    usePlayState.setState({ playing: false })
  },
  toggle() { return usePlayState.getState().playing ? playback.pause() : playback.play() },
  seek(time: number) {
    t = Math.max(0, Math.min(time, end))
    emit()
  },
  subscribe(fn: Listener) {
    listeners.add(fn)
    return () => { listeners.delete(fn) }
  },
}

/** Calls `fn` on every change of the cursor time. */
export function usePlayback(fn: Listener) {
  useEffect(() => playback.subscribe(fn), [fn])
}
