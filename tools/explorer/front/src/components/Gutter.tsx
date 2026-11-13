import { Separator } from 'react-resizable-panels'

/** A one-pixel divider that lights up while it can be dragged. */
export function Gutter({ across }: { across?: boolean }) {
  return (
    <Separator className={`bg-line outline-none transition-[background-color,box-shadow] duration-150
      data-[separator=active]:bg-accent data-[separator=hover]:bg-accent
      data-[separator=active]:shadow-[0_0_0_1px_var(--accent)] data-[separator=hover]:shadow-[0_0_0_1px_var(--accent)]
      relative z-10 after:absolute after:content-[''] ${across ? 'h-px after:inset-x-0 after:-inset-y-1' : 'w-px after:inset-y-0 after:-inset-x-1'}`} />
  )
}
