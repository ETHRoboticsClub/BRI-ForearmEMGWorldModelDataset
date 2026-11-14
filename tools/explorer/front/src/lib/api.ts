export interface Summary {
  key: string
  dataset: string
  subject: string
  session: string
  condition: string
  label_source: string
  kin_units: string | null
  n_channels: number
  duration_s: number
}

export interface Details extends Summary {
  device: string
  electrode_type: string
  citation: string
  url: string
  native_fs: number
  effective_fs: number
  fs: number
  label_fs: number
  montages: Record<string, number[]>
  display_montage: string
  stimulus_names: Record<string, string>
  joints: string[]
  measured_joints: number[]
  kin_lag_s: number
  verified: Record<string, string>
  stimuli: number[]
  emg_stats: { floor: number[]; active: number[] }
  kin_stats: { low: (number | null)[]; high: (number | null)[] }
}

export type Segment = [start: number, end: number, id: number]

export interface Overview {
  dt: number
  envelope: number[]
  flexion: number[]
  stimulus: Segment[]
}

interface Series {
  t0: number
  dt: number
}

export interface WindowData {
  start: number
  end: number
  emg: Series & { channels: number[]; min: number[][]; max: number[][] }
  envelope: Series & { values: number[][] }
  kin: Series & { joints: number[]; values: number[][] }
  stimulus: Segment[]
}

export interface HandRule {
  joint: string
  label: string | null
  at_0: number
  at_1: number
  neutral_mean: boolean
  source: string | null
  ratio: number
  fallback: number
}

export interface WindowQuery {
  start: number
  end: number
  points: number
  emg_scale: 'uv' | 'active'
  kin_scale: 'rad' | 'unit'
  channels: 'ring8' | 'all'
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api${path}`, { signal })
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`)
  return response.json()
}

export const api = {
  config: () => get<{ hand_port: number | null; data: string }>('/config'),
  sessions: () => get<Summary[]>('/sessions'),
  hands: () => get<{ current: string | null; hands: { name: string; title: string }[] }>('/hands'),
  handMap: (name: string) => get<HandRule[]>(`/hands/${name}`),
  details: (key: string) => get<Details>(`/sessions/${key}`),
  overview: (key: string) => get<Overview>(`/sessions/${key}/overview?points=1600`),
  window: (key: string, query: WindowQuery, signal: AbortSignal) => {
    const params = new URLSearchParams(Object.entries(query).map(([k, v]) => [k, String(v)]))
    return get<WindowData>(`/sessions/${key}/window?${params}`, signal)
  },
}

/** The value of a series at time t (nearest sample), or undefined outside it. */
export function valueAt(series: Series, values: number[], t: number) {
  const i = Math.round((t - series.t0) / series.dt)
  return i >= 0 && i < values.length ? values[i] : undefined
}
