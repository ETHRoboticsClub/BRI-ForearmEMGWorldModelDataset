import { useEffect } from 'react'
import { Group, Panel, useDefaultLayout } from 'react-resizable-panels'
import { useShallow } from 'zustand/react/shallow'
import { EmgPlot } from './components/EmgPlot'
import { Gutter } from './components/Gutter'
import { HandActions, HandView, JointBarsView } from './components/HandPanel'
import { JointPicker } from './components/JointPicker'
import { KinPlot } from './components/KinPlot'
import { Navigator } from './components/Navigator'
import { Section } from './components/Section'
import { Segmented } from './components/Segmented'
import { Sidebar } from './components/Sidebar'
import { Toolbar } from './components/Toolbar'
import { TopBar } from './components/TopBar'
import { HEADER, useCollapse } from './lib/collapse'
import { playback } from './lib/playback'
import { useSize } from './lib/size'
import { useExplorer } from './store'

function useKeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return
      const { view, zoom } = useExplorer.getState()
      const step = e.shiftKey ? 10 : 1
      if (e.key === ' ') { e.preventDefault(); playback.toggle() }
      else if (e.key === 'ArrowLeft') playback.seek(playback.time - step)
      else if (e.key === 'ArrowRight') playback.seek(playback.time + step)
      else if (e.key === '+' || e.key === '=') zoom(0.5, (view.start + view.end) / 2)
      else if (e.key === '-') zoom(2, (view.start + view.end) / 2)
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [])
}

function EmgOptions() {
  const { emgScale, channels, montage, set } = useExplorer(useShallow((s) => ({
    emgScale: s.emgScale, channels: s.channels, montage: s.details?.display_montage ?? 'ring8', set: s.set,
  })))
  return (
    <>
      <Segmented value={emgScale} onChange={(v) => set({ emgScale: v })} options={[
        { value: 'uv', label: 'µV' },
        { value: 'active', label: '÷ active', title: 'divided by the p95 of the 100 ms RMS (default normalization)' },
      ]} />
      <Segmented value={channels} onChange={(v) => set({ channels: v })} options={[
        { value: 'ring8', label: montage, title: 'the 8-electrode ring, or the dataset\'s small montage when it has none' },
        { value: 'all', label: 'all' },
      ]} />
    </>
  )
}

function KinOptions() {
  const { kinScale, set } = useExplorer(useShallow((s) => ({ kinScale: s.kinScale, set: s.set })))
  return (
    <>
      <JointPicker />
      <Segmented value={kinScale} onChange={(v) => set({ kinScale: v })} options={[
        { value: 'rad', label: 'rad' },
        { value: 'unit', label: '0–1', title: 'open (p2) = 0, closed (p98) = 1' },
      ]} />
    </>
  )
}

/** Overview, transport, and the EMG over the joints on one time axis. */
function Signals() {
  const rows = useDefaultLayout({ id: 'explorer-signals' })
  const emg = useCollapse()
  const kin = useCollapse()
  const [ref, size] = useSize<HTMLDivElement>()
  useEffect(() => { if (size.width) useExplorer.getState().set({ plotWidth: size.width }) }, [size.width])
  return (
    <div ref={ref} className="flex h-full flex-col">
      <div className="h-16 shrink-0 border-b border-line bg-panel"><Navigator /></div>
      <Toolbar />
      <div className="min-h-0 flex-1">
        <Group orientation="vertical" defaultLayout={rows.defaultLayout} onLayoutChanged={rows.onLayoutChanged}>
          <Panel id="emg" defaultSize="62%" minSize={120} collapsible collapsedSize={HEADER} panelRef={emg.panelRef} onResize={emg.onResize}>
            <Section title="EMG" collapsed={emg.collapsed} onToggle={emg.toggle} actions={<EmgOptions />}><EmgPlot /></Section>
          </Panel>
          <Gutter across />
          <Panel id="kin" minSize={100} collapsible collapsedSize={HEADER} panelRef={kin.panelRef} onResize={kin.onResize}>
            <Section title="Joints" collapsed={kin.collapsed} onToggle={kin.toggle} actions={<KinOptions />}><KinPlot /></Section>
          </Panel>
        </Group>
      </div>
    </div>
  )
}

function HandColumn() {
  const layout = useDefaultLayout({ id: 'explorer-right' })
  const model = useCollapse()
  const bars = useCollapse()
  return (
    <Group orientation="vertical" defaultLayout={layout.defaultLayout} onLayoutChanged={layout.onLayoutChanged}>
      <Panel id="model" minSize={160} collapsible collapsedSize={HEADER} panelRef={model.panelRef} onResize={model.onResize}>
        <Section title="Hand" collapsed={model.collapsed} onToggle={model.toggle} actions={<HandActions />}><HandView /></Section>
      </Panel>
      <Gutter across />
      <Panel id="bars" defaultSize={340} minSize={120} maxSize="70%" collapsible collapsedSize={HEADER}
        panelRef={bars.panelRef} onResize={bars.onResize} groupResizeBehavior="preserve-pixel-size">
        <Section title="At cursor" collapsed={bars.collapsed} onToggle={bars.toggle}><JointBarsView /></Section>
      </Panel>
    </Group>
  )
}

export default function App() {
  useKeys()
  useEffect(() => { useExplorer.getState().init() }, [])
  const cols = useDefaultLayout({ id: 'explorer-main' })
  const sessions = useCollapse()
  const hand = useCollapse()
  return (
    <div className="grid h-full grid-rows-[44px_minmax(0,1fr)]">
      <TopBar sessions={sessions} hand={hand} />
      <Group orientation="horizontal" defaultLayout={cols.defaultLayout} onLayoutChanged={cols.onLayoutChanged}>
        <Panel id="sessions" defaultSize={230} minSize={180} maxSize={420} collapsible collapsedSize={0}
          panelRef={sessions.panelRef} onResize={sessions.onResize} groupResizeBehavior="preserve-pixel-size">
          <Sidebar />
        </Panel>
        <Gutter />
        <Panel id="signals" minSize={320}><Signals /></Panel>
        <Gutter />
        <Panel id="hand" defaultSize="30%" minSize={240} maxSize="75%" collapsible collapsedSize={0}
          panelRef={hand.panelRef} onResize={hand.onResize} groupResizeBehavior="preserve-pixel-size">
          <HandColumn />
        </Panel>
      </Group>
    </div>
  )
}
