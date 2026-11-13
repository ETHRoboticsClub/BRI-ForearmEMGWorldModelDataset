import { useEffect, useRef, useState } from 'react'

/** The size of an element, kept up to date. */
export function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: Math.round(entry.contentRect.width), height: Math.round(entry.contentRect.height) }))
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, size] as const
}
