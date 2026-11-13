import { create } from 'zustand'
import { api, type Details, type HandRule, type Overview, type Summary, type WindowData, type WindowQuery } from './lib/api'
import { DEFAULT_JOINTS } from './lib/joints'
import { playback } from './lib/playback'

type EmgScale = WindowQuery['emg_scale']
type KinScale = WindowQuery['kin_scale']
type Channels = WindowQuery['channels']

interface View {
  start: number
  end: number
}

interface State {
  sessions: Summary[]
  handPort: number | null
  hands: { name: string; title: string }[]
  hand?: string
  handMap: HandRule[]
  key?: string
  details?: Details
  overview?: Overview
  view: View
  emgScale: EmgScale
  kinScale: KinScale
  channels: Channels
  joints: number[]
  data?: WindowData
  loading: boolean
  error?: string
  plotWidth: number

  init: () => Promise<void>
  open: (key: string) => Promise<void>
  setView: (start: number, end: number) => void
  zoom: (factor: number, around: number) => void
  pan: (seconds: number) => void
  set: (patch: Partial<Pick<State, 'emgScale' | 'kinScale' | 'channels' | 'joints' | 'plotWidth'>>) => void
  toggleJoint: (joint: number) => void
  setHand: (name: string) => Promise<void>
}

const MIN_SPAN = 0.2
const FIRST_SPAN = 10

export const useExplorer = create<State>((set, get) => ({
  sessions: [],
  handPort: null,
  hands: [],
  handMap: [],
  view: { start: 0, end: FIRST_SPAN },
  emgScale: 'uv',
  kinScale: 'unit',
  channels: 'ring8',
  joints: [],
  loading: false,
  plotWidth: 1200,

  init: async () => {
    const [config, sessions, hands] = await Promise.all([api.config(), api.sessions(), api.hands()])
    set({ sessions, handPort: config.hand_port, hands: hands.hands })
    const stored = (() => { try { return localStorage.getItem('explorer.hand') } catch { return null } })()
    const hand = hands.hands.find((h) => h.name === stored)?.name ?? hands.current ?? hands.hands[0]?.name
    if (hand) await get().setHand(hand)
    const fromUrl = new URLSearchParams(location.search).get('session')
    const first = sessions.find((s) => s.key === fromUrl) ?? sessions[0]
    if (first) await get().open(first.key)
  },

  open: async (key) => {
    set({ key, details: undefined, overview: undefined, data: undefined, error: undefined })
    history.replaceState(null, '', `?session=${encodeURIComponent(key)}`)
    try {
      const [details, overview] = await Promise.all([api.details(key), api.overview(key)])
      if (get().key !== key) return
      const measured = new Set(details.measured_joints)
      const joints = DEFAULT_JOINTS.map((name) => details.joints.indexOf(name)).filter((j) => measured.has(j))
      set({ details, overview, joints, view: { start: 0, end: Math.min(FIRST_SPAN, details.duration_s) } })
      playback.reset(details.duration_s)
    } catch (error) {
      set({ error: String(error) })
    }
  },

  setView: (start, end) => {
    const duration = get().details?.duration_s ?? end
    const span = Math.min(Math.max(end - start, MIN_SPAN), duration)
    const s = Math.min(Math.max(start, 0), duration - span)
    set({ view: { start: s, end: s + span } })
  },

  zoom: (factor, around) => {
    const { start, end } = get().view
    get().setView(around - (around - start) * factor, around + (end - around) * factor)
  },

  pan: (seconds) => {
    const { start, end } = get().view
    get().setView(start + seconds, end + seconds)
  },

  set: (patch) => set(patch),

  setHand: async (name) => {
    set({ hand: name, handMap: await api.handMap(name) })
    try { localStorage.setItem('explorer.hand', name) } catch { /* private window */ }
  },

  toggleJoint: (joint) => {
    const joints = get().joints
    set({ joints: joints.includes(joint) ? joints.filter((j) => j !== joint) : [...joints, joint] })
  },
}))

// Window data: fetch the view with half a view of margin on each side, and reuse it while the view stays inside
// at a similar zoom, so panning and playing rarely wait for the server.

let fetched: { query: WindowQuery; key: string } | undefined
let controller: AbortController | undefined
let timer = 0

function covers(state: State) {
  if (!fetched || fetched.key !== state.key) return false
  const { query } = fetched
  const span = state.view.end - state.view.start
  const fetchedSpan = (query.end - query.start) / 2
  const sameOptions = query.emg_scale === state.emgScale && query.kin_scale === state.kinScale
    && query.channels === state.channels
  const inside = state.view.start >= query.start - 1e-6 && state.view.end <= query.end + 1e-6
  return sameOptions && inside && span / fetchedSpan > 0.7 && span / fetchedSpan < 1.4
}

function refresh() {
  const state = useExplorer.getState()
  if (!state.key || !state.details || covers(state)) return
  clearTimeout(timer)
  timer = window.setTimeout(async () => {
    const s = useExplorer.getState()
    if (!s.key) return
    const span = s.view.end - s.view.start
    const query: WindowQuery = {
      start: Math.max(0, s.view.start - span / 2),
      end: s.view.end + span / 2,
      points: Math.min(8000, Math.round(s.plotWidth * 2)),
      emg_scale: s.emgScale,
      kin_scale: s.kinScale,
      channels: s.channels,
    }
    controller?.abort()
    controller = new AbortController()
    useExplorer.setState({ loading: true })
    try {
      const data = await api.window(s.key, query, controller.signal)
      fetched = { query, key: s.key }
      useExplorer.setState({ data, loading: false, error: undefined })
    } catch (error) {
      if ((error as Error).name !== 'AbortError') useExplorer.setState({ loading: false, error: String(error) })
    }
  }, 40)
}

useExplorer.subscribe((state, previous) => {
  if (state.key !== previous.key) fetched = undefined
  if (state.view !== previous.view || state.details !== previous.details || state.emgScale !== previous.emgScale
    || state.kinScale !== previous.kinScale || state.channels !== previous.channels) refresh()
})

// While playing, the view follows the cursor: it jumps ahead when the cursor reaches 90% of it.
playback.subscribe((t) => {
  const { view, setView } = useExplorer.getState()
  const span = view.end - view.start
  if (t > view.start + span * 0.9 || t < view.start) setView(t - span * 0.1, t + span * 0.9)
})
