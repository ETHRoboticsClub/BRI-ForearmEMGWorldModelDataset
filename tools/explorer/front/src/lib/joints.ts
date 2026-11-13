const FINGER_COLOR: Record<string, string> = {
  thumb: 'c-5', index: 'c-1', middle: 'c-2', ring: 'c-3', pinky: 'c-4', wrist: 'c-6', palm: 'muted',
}

const SEGMENT_DASH: Record<string, number[]> = { mcp: [], cmc: [], pip: [5, 3], ip: [5, 3], dip: [1.5, 2.5] }

/** Colour token by finger, dash by segment: index_pip_flex -> ("c-1", [5, 3]). */
export function jointStyle(name: string) {
  const [first, second] = name.split('_')
  const color = name.endsWith('_abd') && first !== 'thumb' ? 'muted' : FINGER_COLOR[first] ?? 'muted'
  return { color, dash: SEGMENT_DASH[second] ?? [] }
}

const SHORT: Record<string, string> = { thumb: 'T', index: 'I', middle: 'M', ring: 'R', pinky: 'P' }

/** index_pip_flex -> "I pip", index_middle_abd -> "IM abd", wrist_flex -> "wrist flex". */
export function shortName(name: string) {
  const parts = name.split('_')
  if (!(parts[0] in SHORT)) return parts.join(' ')
  const kept = name.replace(/_flex$/, '').split('_')
  return [kept.filter((p) => p in SHORT).map((p) => SHORT[p]).join(''), ...kept.filter((p) => !(p in SHORT))].join(' ')
}

export const DEFAULT_JOINTS = ['thumb_mcp_flex', 'index_mcp_flex', 'middle_mcp_flex', 'ring_mcp_flex', 'pinky_mcp_flex', 'wrist_flex']
