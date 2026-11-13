#import "@preview/cetz:0.5.3"
#import "@preview/suiji:0.5.1": gen-rng-f, normal-f

#let ink = rgb("#28282a")
#let muted = rgb("#6b6a66")
#let blue = rgb("#2a78d6")
#let orange = rgb("#eb6834")
#let green = rgb("#1baf7a")

#let kinds = (
  step: (fill: white, stroke: blue),
  data: (fill: rgb("#eaf2fc"), stroke: blue),
  stats: (fill: rgb("#fdeee7"), stroke: orange),
  result: (fill: rgb("#e6f5ee"), stroke: green),
)

#let setup(body) = {
  set page(width: auto, height: auto, margin: 6pt)
  set text(size: 8pt, fill: ink)
  body
}

// Signals

#let samples(n) = range(n).map(i => i / (n - 1))
#let noise(n, seed) = normal-f(gen-rng-f(seed), size: n).at(1)
#let bump(t, center, width: 0.07) = calc.exp(-calc.pow((t - center) / width, 2))
#let activation(t) = bump(t, 0.2) + 0.6 * bump(t, 0.5) + 0.9 * bump(t, 0.8)
#let emg(n, gain: 1, floor: 0.06, seed: 1) = samples(n).zip(noise(n, seed)).map(((t, z)) => (
  gain * (activation(t) + floor) * z
))
#let flexion(t, shift: 0) = 0.5 - 0.5 * calc.cos(2 * calc.pi * 1.6 * (t - shift))

#let percentile(xs, q) = xs.sorted().at(int(q / 100 * (xs.len() - 1)))
#let moving-rms(xs, window) = xs.enumerate().map(((i, _)) => {
  let part = xs.slice(calc.max(0, i - window + 1), i + 1)
  calc.sqrt(part.map(x => x * x).sum() / part.len())
})
#let rescale(xs, low, high) = xs.map(x => (x - low) / (high - low))

#let trace(ys, color: blue) = (ys: ys, color: color)
#let level(y, color: muted, label: none) = (y: y, color: color, label: label)

// Drawing

#let sparkline(area, traces, levels: (), log: false) = {
  import cetz.draw: *
  let scale(v) = if log { calc.log(calc.max(v, 1e-6)) } else { v }
  let values = traces.map(t => t.ys).flatten().map(scale) + levels.map(l => scale(l.y))
  let (low, high) = (calc.min(..values), calc.max(..values))
  let ((x0, y0), (x1, y1)) = area
  let px(i, n) = x0 + (x1 - x0) * i / (n - 1)
  let py(v) = y0 + (y1 - y0) * (scale(v) - low) / (high - low)

  for t in traces {
    let n = t.ys.len()
    line(..t.ys.enumerate().map(((i, v)) => (px(i, n), py(v))), stroke: 0.5pt + t.color)
  }
  for l in levels {
    line((x0, py(l.y)), (x1, py(l.y)), stroke: (paint: l.color, thickness: 0.5pt, dash: "dashed"))
    if l.label != none { content((x1 + 0.05, py(l.y)), anchor: "west", text(size: 5.5pt, fill: l.color, l.label)) }
  }
}

#let node(center, kind, title, detail: none, size: (3.4, 2.4), traces: (), levels: (), log: false) = {
  import cetz.draw: *
  let (x, y) = center
  let (w, h) = size
  let style = kinds.at(kind)
  rect((x - w / 2, y - h / 2), (x + w / 2, y + h / 2), radius: 0.12, fill: style.fill, stroke: 0.8pt + style.stroke)
  content((x, y + h / 2 - 0.3), text(weight: "bold", title))
  if detail != none { content((x, y + h / 2 - 0.68), text(size: 6.5pt, fill: muted, detail)) }
  if traces.len() > 0 {
    let area = ((x - w / 2 + 0.25, y - h / 2 + 0.2), (x + w / 2 - 0.45, y + h / 2 - 1.0))
    sparkline(area, traces, levels: levels, log: log)
  }
}

#let panel(from, to, label) = {
  import cetz.draw: *
  rect(from, to, radius: 0.2, fill: rgb("#f6f6f4"), stroke: (paint: rgb("#a8a7a2"), thickness: 0.6pt, dash: "dashed"))
  content((from.at(0) + 0.2, to.at(1) - 0.25), anchor: "west", text(weight: "bold", fill: muted, label))
}

#let arrow(..points, label: none) = {
  import cetz.draw: *
  let pts = points.pos()
  line(..pts, stroke: 0.8pt + ink, mark: (end: "stealth", fill: ink, scale: 0.7))
  if label != none {
    let (a, b) = (pts.at(-2), pts.at(-1))
    content(((a.at(0) + b.at(0)) / 2, (a.at(1) + b.at(1)) / 2), anchor: "south", padding: 0.08, text(
      size: 6.5pt,
      fill: muted,
      label,
    ))
  }
}

#let title(position, body) = cetz.draw.content(position, anchor: "west", text(size: 11pt, weight: "bold", body))
