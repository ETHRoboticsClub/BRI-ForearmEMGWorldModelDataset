import { Network, Tag } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { create } from 'zustand'
import { playback } from '../lib/playback'
import { useTheme } from '../lib/theme'
import { useExplorer } from '../store'
import { HandMap } from './HandMap'
import { JointBars } from './JointBars'
import { useSize } from '../lib/size'
import { Menu } from './Menu'

const VISER_MIN_WIDTH = 600

export const useHandLink = create<{ connected: boolean }>(() => ({ connected: false }))

const SEND_EVERY_MS = 33
const RAD_PER_PIXEL = 0.008

let handSocket: WebSocket | null = null

function sendToHand(message: object) {
  if (handSocket?.readyState === WebSocket.OPEN) handSocket.send(JSON.stringify(message))
}

/** Keeps the hand of the viser server at the cursor of the open session. */
function useHandSync(enabled: boolean, labels: boolean) {
  const socket = useRef<WebSocket | null>(null)
  const key = useExplorer((s) => s.key)
  const kinScale = useExplorer((s) => s.kinScale)
  const hand = useExplorer((s) => s.hand)
  const theme = useTheme((s) => s.theme)

  useEffect(() => {
    if (!enabled) return
    let closed = false
    let retry = 0
    const connect = () => {
      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/hand`)
      ws.onopen = () => {
        useHandLink.setState({ connected: true })
        ws.send(JSON.stringify({ dark: useTheme.getState().theme === 'dark', hand: useExplorer.getState().hand }))
      }
      ws.onclose = () => {
        useHandLink.setState({ connected: false })
        if (!closed) retry = window.setTimeout(connect, 1500)
      }
      socket.current = ws
      handSocket = ws
    }
    connect()
    return () => { closed = true; clearTimeout(retry); socket.current?.close() }
  }, [enabled])

  useEffect(() => {
    let sentAt = 0
    let sentT = -1
    let pending = 0
    const send = () => {
      const ws = socket.current
      const key = useExplorer.getState().key
      if (!ws || ws.readyState !== WebSocket.OPEN || !key || playback.time === sentT) return
      ws.send(JSON.stringify({ key, t: playback.time, unit: useExplorer.getState().kinScale === 'unit' }))
      sentAt = performance.now()
      sentT = playback.time
    }
    const unsubscribe = playback.subscribe(() => {
      clearTimeout(pending)
      const wait = SEND_EVERY_MS - (performance.now() - sentAt)
      if (wait <= 0) send()
      else pending = window.setTimeout(send, wait)
    })
    send()
    return () => { unsubscribe(); clearTimeout(pending) }
  }, [key, kinScale, hand])

  useEffect(() => {
    const ws = socket.current
    if (ws?.readyState === WebSocket.OPEN && hand) {
      ws.send(JSON.stringify({ hand }))
      ws.send(JSON.stringify({ key: useExplorer.getState().key, t: playback.time, unit: useExplorer.getState().kinScale === 'unit' }))
    }
  }, [hand])

  useEffect(() => {
    const ws = socket.current
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ dark: theme === 'dark', labels }))
  }, [theme, labels])
}

interface HandOptions {
  on: boolean
  map: boolean
  toggle: () => void
  toggleMap: () => void
}

export const useHandLabels = create<HandOptions>((set, get) => ({
  on: true,
  map: false,
  toggle: () => set({ on: !get().on }),
  toggleMap: () => set({ map: !get().map }),
}))

/** Free rotation over the viser view: drag turns the camera about its own axes (no stuck poles), wheel zooms,
 * double-click resets. The server moves the camera, see HandScene.orbit. */
function Trackball() {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    let last: { x: number; y: number } | null = null
    let pending = { x: 0, y: 0 }
    let frame = 0
    const flush = () => {
      frame = 0
      if (pending.x || pending.y) sendToHand({ orbit: [pending.x * RAD_PER_PIXEL, pending.y * RAD_PER_PIXEL] })
      pending = { x: 0, y: 0 }
    }
    const down = (e: PointerEvent) => { last = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId) }
    const move = (e: PointerEvent) => {
      if (!last) return
      pending.x += e.clientX - last.x
      pending.y += e.clientY - last.y
      last = { x: e.clientX, y: e.clientY }
      frame ||= requestAnimationFrame(flush)
    }
    const up = () => { last = null }
    const wheel = (e: WheelEvent) => { e.preventDefault(); sendToHand({ zoom: Math.exp(e.deltaY * 0.0012) }) }
    const reset = () => sendToHand({ reset_view: true })
    el.addEventListener('pointerdown', down)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    el.addEventListener('wheel', wheel, { passive: false })
    el.addEventListener('dblclick', reset)
    return () => {
      cancelAnimationFrame(frame)
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
      el.removeEventListener('wheel', wheel)
      el.removeEventListener('dblclick', reset)
    }
  }, [])
  return <div ref={ref} title="Drag: rotate · wheel: zoom · double-click: reset view"
    className="absolute inset-0 cursor-grab touch-none active:cursor-grabbing" />
}

/** Sync dot and the labels switch, for the section header. */
export function HandActions() {
  const connected = useHandLink((s) => s.connected)
  const { on, toggle, map, toggleMap } = useHandLabels()
  const hands = useExplorer((s) => s.hands)
  const hand = useExplorer((s) => s.hand)
  const setHand = useExplorer((s) => s.setHand)
  return (
    <>
      <span className={`size-1.5 rounded-full ${connected ? 'bg-accent' : 'bg-muted'}`} title={connected ? 'in sync' : 'connecting'} />
      <Menu value={hand} onChange={setHand} options={hands.map((h) => ({ value: h.name, label: h.title }))} />
      <button onClick={toggleMap} title="Joint ↔ label map" aria-pressed={map}
        className={`grid size-6 place-items-center rounded-md ${map ? 'bg-raised text-text' : 'text-muted hover:text-text'}`}>
        <Network size={12} />
      </button>
      <button onClick={toggle} title="Joint labels" aria-pressed={on}
        className={`grid size-6 place-items-center rounded-md ${on ? 'bg-raised text-text' : 'text-muted hover:text-text'}`}>
        <Tag size={12} />
      </button>
    </>
  )
}

export function HandView() {
  const port = useExplorer((s) => s.handPort)
  const labels = useHandLabels((s) => s.on)
  const map = useHandLabels((s) => s.map)
  const hand = useExplorer((s) => s.hand)
  useHandSync(port !== null, labels)
  const [box, size] = useSize<HTMLDivElement>()
  // viser switches to its mobile layout (control panel always open) below 576 px: render wider and scale down
  const scale = size.width > 0 && size.width < VISER_MIN_WIDTH ? size.width / VISER_MIN_WIDTH : 1
  return (
    <div ref={box} className="absolute inset-0 overflow-hidden">
      {port === null
        ? <p className="grid size-full place-items-center text-muted">Hand off (--no-hand)</p>
        : <iframe key={hand} title="hand" src={`${location.protocol}//${location.hostname}:${port}`}
            className="absolute top-0 left-0 origin-top-left border-0 bg-bg"
            style={{ width: size.width / scale, height: size.height / scale, transform: `scale(${scale})` }} />}
      {port !== null && !map && <Trackball />}
      {map && <HandMap />}
    </div>
  )
}

export function JointBarsView() {
  return <div className="absolute inset-0 overflow-y-auto px-2 py-1.5"><JointBars /></div>
}
