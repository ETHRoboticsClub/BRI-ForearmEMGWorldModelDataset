#import "style.typ": *
#show: setup

// Synthetic calibration (femg.calib_baseline): 3 actions, each a pattern on the 8-electrode virtual ring (log RMS vs
// rest, minus the ring mean). Training sessions record them at unknown placements (rotation k, mirror m) plus noise;
// aligning them gives the templates. A new user's band sits at k = 3 (135°), mirrored: matching finds the placement.
#let n = 8
#let actions = ((name: [fist], peak: 6.0), (name: [pointing], peak: 6.6), (name: [wrist flexion], peak: 0.0))
#let pattern(peak) = {
  let raw = range(n).map(i => calc.exp(1.6 * calc.cos(2 * calc.pi * (i - peak) / n)))
  let mean = raw.sum() / n
  raw.map(v => (v - mean) / 2.5)
}
#let templates = actions.map(a => pattern(a.peak))
#let transform(x, k, m) = range(n).map(i => {
  let j = calc.rem-euclid(i - k, n)
  x.at(if m { calc.rem-euclid(-j, n) } else { j })
})
#let placements = (false, true).map(m => range(n).map(k => (k, m))).flatten().chunks(2)
#let noisy(x, seed) = x.zip(noise(n, seed)).map(((v, z)) => v + 0.12 * z)

#let sessions = ((2, false), (5, true), (0, false), (7, true), (3, false), (6, false))
#let recorded = sessions.enumerate().map(((s, (k, m))) => templates
  .enumerate()
  .map(((a, t)) => noisy(transform(t, k, m), 10 * s + a))
  .flatten())
#let undo(x, k, m) = {
  // the placement that brings transform(·, k, m) back: found by search, as the alignment does
  let target = range(n).map(i => i)
  let moved = transform(target, k, m)
  let back = placements.find(p => transform(moved, p.at(0), p.at(1)) == target)
  range(templates.len()).map(a => transform(x.slice(a * n, (a + 1) * n), back.at(0), back.at(1))).flatten()
}
#let aligned = sessions.zip(recorded).map((((k, m), x)) => undo(x, k, m))

#let user-placement = (3, true)
#let user = templates.enumerate().map(((a, t)) => noisy(transform(t, ..user-placement), 100 + a))
#let corr(u, v) = {
  let (mu, mv) = (u.sum() / u.len(), v.sum() / v.len())
  let cov = u.zip(v).map(((x, y)) => (x - mu) * (y - mv)).sum()
  cov / calc.sqrt(u.map(x => calc.pow(x - mu, 2)).sum() * v.map(y => calc.pow(y - mv, 2)).sum())
}
#let scores = placements.map(p => corr(user.map(x => transform(x, ..p)).flatten(), templates.flatten()))
#let best = scores.position(s => s == calc.max(..scores))
#let matched = user.map(x => transform(x, ..placements.at(best)))
// V(θ*, m*) as a permutation on the 8 channels: row i takes channel j
#let v-matrix = range(n).map(i => {
  let unit = range(n).map(j => if j == i { 1.0 } else { 0.0 })
  transform(unit, ..placements.at(best))
})

// A forearm cross-section, seen from the hand: palmar down, radial right; one dot per virtual electrode
#let ring(center, values, r: 0.48) = {
  import cetz.draw: *
  let (x, y) = center
  circle((x, y), radius: r + 0.17, fill: rgb("#f3e6dc"), stroke: 0.4pt + muted)
  for (i, v) in values.enumerate() {
    let a = i * 45deg
    circle((x + r * calc.sin(a), y - r * calc.cos(a)), radius: 0.11, fill: heat-color(v, top: 0.7), stroke: 0.4pt + muted)
  }
}

#cetz.canvas({
  import cetz.draw: *
  let (fw, fh) = (17.8, 7.3)
  rect((0, 0), (fw, fh), radius: 0.15, stroke: 0.5pt + rgb("#d8d7d2"), fill: white)
  let cell = (0.115, 0.24)

  // a: training data -> templates
  let ay = 5.15
  letter((0.3, fh - 0.25), [a])
  content((0.75, fh - 0.3), anchor: "north-west", text(size: 7pt, fill: muted)[training sessions])
  let (hw, hh) = (3 * n * cell.at(0), sessions.len() * cell.at(1))
  let stack(x0, rows) = {
    heatmap(((x0, ay - hh / 2), (x0 + hw, ay + hh / 2)), rows, top: 0.7)
    for a in range(1, templates.len()) {
      let xa = x0 + a * n * cell.at(0)
      line((xa, ay - hh / 2), (xa, ay + hh / 2), stroke: 1pt + white)
    }
  }
  let (s1, s2) = (2.35, 6.95) // centred in the frame
  stack(s1, recorded)
  stack(s2, aligned)
  for (x0, label) in ((s1, $s_i$), (s2, $T_(k_i, m_i) s_i$)) {
    math-label((x0 + hw / 2, ay - hh / 2), label)
  }
  arrow((s1 + hw + 0.15, ay), (s2 - 0.15, ay))
  content(((s1 + hw + s2) / 2, ay + 0.12), anchor: "south", text(size: 7pt)[$max_(k, m) rho$])
  arrow((s2 + hw + 0.15, ay), (s2 + hw + 1.0, ay))
  content((s2 + hw + 0.575, ay + 0.12), anchor: "south", text(size: 7pt)[mean])
  let t0 = s2 + hw + 1.95
  for (a, act) in actions.enumerate() {
    let cx = t0 + a * 1.75
    ring((cx, ay), templates.at(a))
    content((cx, ay - 0.85), anchor: "north", text(size: 7pt, fill: muted, act.name))
  }
  math-label((t0 + 1.75, ay - 1.15), $T_a$, anchor: "north")

  // b: a new user -> placement
  let by = 1.75
  letter((0.3, 3.55), [b])
  content((0.75, 3.5), anchor: "north-west", text(size: 7pt, fill: muted)[new user])
  for (a, x) in user.enumerate() {
    ring((1.35 + a * 1.45, by), x)
  }
  math-label((1.35 + 1.45, by - 0.85), $u_a$, anchor: "north")
  // the 16 placements
  let (bx0, bw, bh) = (5.75, 0.22, 1.5)
  arrow((5.0, by), (bx0 - 0.15, by))
  let base = by - bh / 2
  let (low, high) = (calc.min(..scores), calc.max(..scores))
  for (i, s) in scores.enumerate() {
    let x = bx0 + i * (bw + 0.06) + if i >= n { 0.25 } else { 0 }
    let height = 0.1 + (bh - 0.1) * (s - low) / (high - low)
    rect((x, base), (x + bw, base + height), fill: if i == best { orange } else { rgb("#c9c8c3") }, stroke: none)
  }
  let bx1 = bx0 + 2 * n * (bw + 0.06) + 0.25
  line((bx0 - 0.05, base), (bx1, base), stroke: 0.5pt + muted)
  content((bx0 + n * (bw + 0.06) / 2, base - 0.08), anchor: "north", text(size: 7pt)[$m = 0$])
  content((bx0 + n * (bw + 0.06) * 1.5 + 0.25, base - 0.08), anchor: "north", text(size: 7pt)[$m = 1$])
  content(((bx0 + bx1) / 2, base - 0.5), anchor: "north", text(size: 7pt)[$rho(T_(k, m) u, T)$, $k = 0..7$])
  // the templates feed the match
  line((t0 + 1.75, ay - 1.6), (t0 + 1.75, 3.15), ((bx0 + bx1) / 2, 3.15), ((bx0 + bx1) / 2, base + bh + 0.12), stroke: (
    paint: muted,
    thickness: 0.5pt,
    dash: "dashed",
  ), mark: (end: "stealth", fill: muted, scale: 0.5))
  // V(θ*, m*) and the user's aligned patterns
  let vx = bx1 + 0.75
  arrow((bx1 + 0.12, by), (vx - 0.12, by))
  let vs = 1.3
  heatmap(((vx, by - vs / 2), (vx + vs, by + vs / 2)), v-matrix, top: 1)
  math-label((vx + vs / 2, by - vs / 2), $V(theta^*, m^*)$)
  arrow((vx + vs + 0.12, by), (vx + vs + 0.65, by))
  for (a, x) in matched.enumerate() {
    ring((vx + vs + 1.4 + a * 1.45, by), x)
  }
  math-label((vx + vs + 1.4 + 1.45, by - 0.85), $T_(k^*, m^*) u_a approx T_a$, anchor: "north")
})
