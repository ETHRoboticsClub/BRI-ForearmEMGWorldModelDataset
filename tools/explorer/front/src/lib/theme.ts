import { create } from 'zustand'

type Theme = 'light' | 'dark'

const stored = (): Theme | null => {
  try { return localStorage.getItem('explorer.theme') as Theme | null } catch { return null }
}

let cache: Record<string, string> = {}

const apply = (theme: Theme) => {
  document.documentElement.dataset.theme = theme
  cache = {}
  try { localStorage.setItem('explorer.theme', theme) } catch { /* private window */ }
}

export const useTheme = create<{ theme: Theme; toggle: () => void }>((set, get) => ({
  theme: stored() ?? (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'),
  toggle: () => set({ theme: get().theme === 'dark' ? 'light' : 'dark' }),
}))

apply(useTheme.getState().theme)
useTheme.subscribe((s) => apply(s.theme))

/** A colour token of the current theme, for canvases. */
export const token = (name: string) =>
  (cache[name] ??= getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim())
