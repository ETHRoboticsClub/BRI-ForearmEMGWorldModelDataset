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

// Blue > 0, orange < 0, white 0; `top` is the value of full color
#let heat-color(v, top: 1) = {
  let w = calc.min(calc.abs(v) / top, 1)
  color.mix((if v >= 0 { blue } else { orange }, w * 100%), (white, (1 - w) * 100%))
}

// One cell per value of a (rows, columns) array, colored with heat-color
#let heatmap(area, rows, top: 1) = {
  import cetz.draw: *
  let ((x0, y0), (x1, y1)) = area
  let (dx, dy) = ((x1 - x0) / rows.at(0).len(), (y1 - y0) / rows.len())
  for (i, row) in rows.enumerate() {
    for (j, v) in row.enumerate() {
      rect((x0 + j * dx, y1 - (i + 1) * dy), (x0 + (j + 1) * dx, y1 - i * dy), fill: heat-color(v, top: top), stroke: none)
    }
  }
  rect((x0, y0), (x1, y1), stroke: 0.4pt + muted)
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


// Paper-style figures: panel letters and math labels
#let math-label(position, body, anchor: "north") = cetz.draw.content(position, anchor: anchor, padding: 0.1, body)
#let letter(position, body) = cetz.draw.content(position, anchor: "north-west", text(size: 10pt, weight: "bold", body))

// Journal figures: a white frame, and small cards (a short title over a signal sketch)
#let frame(width, height) = cetz.draw.rect((0, 0), (width, height), radius: 0.15, stroke: 0.5pt + rgb("#d8d7d2"), fill: white)

#let card(center, title, size: (2.3, 1.6), kind: "step", traces: (), levels: (), log: false) = {
  import cetz.draw: *
  let (x, y) = center
  let (w, h) = size
  let style = kinds.at(kind)
  rect((x - w / 2, y - h / 2), (x + w / 2, y + h / 2), radius: 0.08, fill: style.fill, stroke: 0.6pt + style.stroke)
  content((x, if traces.len() > 0 { y + h / 2 - 0.22 } else { y }), text(size: 7pt, title)) // no sketch: centred
  if traces.len() > 0 {
    sparkline(((x - w / 2 + 0.18, y - h / 2 + 0.15), (x + w / 2 - 0.3, y + h / 2 - 0.45)), traces, levels: levels, log: log)
  }
}

#let row-label(position, letter-body, body, color: muted) = {
  import cetz.draw: *
  content(position, anchor: "west", text(size: 10pt, weight: "bold", letter-body) + h(6pt) + text(size: 7pt, fill: color, body))
}
