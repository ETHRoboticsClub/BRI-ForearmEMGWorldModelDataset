import { useShallow } from 'zustand/react/shallow'
import type { HandRule } from '../lib/api'
import { jointStyle, shortName } from '../lib/joints'
import { useExplorer } from '../store'

const deg = (rad: number) => `${Math.round((rad * 180) / Math.PI)}°`

/** What drives a hand joint in this session. */
function status(rule: HandRule, measured: Set<string>) {
  if (rule.label && measured.has(rule.label)) return { text: 'data', tone: 'text-accent' }
  if (rule.source) return { text: `≈ ${rule.ratio} × ${rule.source}`, tone: 'text-muted' }
  if (rule.label) return { text: `held ${deg(rule.at_0)}`, tone: 'text-muted' }
  return { text: `fixed ${deg(rule.fallback)}`, tone: 'text-muted' }
}

/** Every joint of the hand and the label that drives it (src/femg/hands/<hand>.csv). */
export function HandMap() {
  const { rules, details } = useExplorer(useShallow((s) => ({ rules: s.handMap, details: s.details })))
  const measured = new Set(details?.measured_joints.map((j) => details.joints[j]) ?? [])
  const used = new Set(rules.map((r) => r.label))
  const unused = [...measured].filter((label) => !used.has(label))
  return (
    <div className="absolute inset-0 overflow-y-auto bg-panel px-3 py-2">
      <table className="w-full text-[11.5px]">
        <thead className="text-left text-muted">
          <tr><th className="pb-1 font-normal">joint</th><th className="pb-1 font-normal">label</th><th className="pb-1 font-normal">0 → 1</th><th className="pb-1 font-normal">now</th></tr>
        </thead>
        <tbody>
          {rules.map((r) => {
            const s = status(r, measured)
            return (
              <tr key={r.joint} className="border-t border-line/60">
                <td className="py-0.5 pr-2 font-mono text-[11px]">{r.joint}</td>
                <td className="py-0.5 pr-2">
                  {r.label ? (
                    <span className="flex items-center gap-1.5" title={r.label}>
                      <span className="size-1.5 rounded-full" style={{ background: `var(--${jointStyle(r.label).color})` }} />
                      {shortName(r.label)}
                    </span>
                  ) : <span className="text-muted">—</span>}
                </td>
                <td className="py-0.5 pr-2 font-mono text-[11px] text-muted">
                  {r.label ? `${deg(r.at_0)}→${deg(r.at_1)}${r.neutral_mean ? ' ±' : ''}` : ''}
                </td>
                <td className={`py-0.5 ${s.tone}`}>{s.text}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {unused.length > 0 && (
        <p className="mt-2 text-[11px] text-muted">Not on this hand: {unused.map(shortName).join(', ')}</p>
      )}
    </div>
  )
}
