import { useState } from 'react'
import { usePanelRef } from 'react-resizable-panels'

/** A panel that folds to its header (or to nothing) and back. */
export function useCollapse() {
  const ref = usePanelRef()
  const [collapsed, setCollapsed] = useState(false)
  return {
    panelRef: ref,
    collapsed,
    toggle: () => (ref.current?.isCollapsed() ? ref.current.expand() : ref.current?.collapse()),
    onResize: () => setCollapsed(!!ref.current?.isCollapsed()),
  }
}

export type Collapse = ReturnType<typeof useCollapse>

export const HEADER = 32
