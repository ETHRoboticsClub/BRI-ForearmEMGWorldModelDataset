#import "style.typ": *
#show: setup

// Synthetic forearm: 8 muscles around the arm, muscle k fires at time slot k (a staircase in the channel heatmaps).
// Electrode i of the ring picks up muscle k with a Gaussian of their distance around the arm (volume conduction).
// The new user wears the band rotated by 1.2 electrodes (54°) over thicker skin (wider pickup).
// Calibration: 1. find the rotation with the virtual array (VirtualArray(rotation=θ)), 2. fit W on what is left.
#let n-ch = 8
#let n-t = 48
#let shift = 1.2
#let sources = range(n-ch).map(k => samples(n-t).map(t => bump(t, (k + 0.5) / n-ch, width: 0.06)))

#let pickup(offset, spread) = range(n-ch).map(i => range(n-ch).map(k => {
  let d = calc.rem-euclid(i + offset - k, n-ch)
  calc.exp(-0.5 * calc.pow(calc.min(d, n-ch - d) / spread, 2))
}))

#let transpose(a) = range(a.at(0).len()).map(j => a.map(row => row.at(j)))
#let matmul(a, b) = a.map(row => transpose(b).map(col => row.zip(col).map(((x, y)) => x * y).sum()))
#let inverse(m) = {
  let n = m.len()
  let a = m.enumerate().map(((i, row)) => row + range(n).map(j => if i == j { 1.0 } else { 0.0 }))
  for c in range(n) {
    let p = range(c, n).fold(c, (best, r) => if calc.abs(a.at(r).at(c)) > calc.abs(a.at(best).at(c)) { r } else {
      best
    })
    (a.at(c), a.at(p)) = (a.at(p), a.at(c))
    let pivot = a.at(c).map(v => v / a.at(c).at(c))
    a = a.enumerate().map(((r, row)) => if r == c { pivot } else {
      row.zip(pivot).map(((x, y)) => x - row.at(c) * y)
    })
  }
  a.map(row => row.slice(n))
}

#let reference = pickup(0, 0.5)
#let displaced = pickup(shift, 0.6)
#let ridge(target, mixing) = matmul(
  matmul(target, transpose(mixing)),
  inverse(matmul(mixing, transpose(mixing)).enumerate().map(((i, row)) => row
    .enumerate()
    .map(((j, v)) => if i == j { v + 0.05 } else { v }))),
)

// The virtual array at rotation θ (in electrodes, 1 = 45°): virtual k from real i at i + θ, sd 1.5 cm = 0.5 electrode
#let turn(theta) = range(n-ch).map(k => {
  let row = range(n-ch).map(i => {
    let d = calc.rem-euclid(i + theta - k, n-ch)
    calc.exp(-0.5 * calc.pow(calc.min(d, n-ch - d) / 0.5, 2))
  })
  let total = row.sum()
  row.map(w => w / total)
})

#let x-ref = matmul(reference, sources)
#let x-new = matmul(displaced, sources)
#let corr(a, b) = {
  let (u, v) = (a.flatten(), b.flatten())
  let (mu, mv) = (u.sum() / u.len(), v.sum() / v.len())
  let cov = u.zip(v).map(((x, y)) => (x - mu) * (y - mv)).sum()
  cov / calc.sqrt(u.map(x => calc.pow(x - mu, 2)).sum() * v.map(y => calc.pow(y - mv, 2)).sum())
}
// 1. the rotation search: how well the rotated virtual channels match the templates (the reference user)
#let thetas = range(64).map(i => i / 8)
#let scores = thetas.map(th => corr(matmul(turn(th), x-new), x-ref))
#let best = thetas.at(scores.position(s => s == calc.max(..scores)))
#let x-rot = matmul(turn(best), x-new)
// 2. W on what is left (identity at the start of the fit)
#let adapter = ridge(reference, matmul(turn(best), displaced))
#let x-fixed = matmul(adapter, x-rot)

// The frozen encoder, reduced to one readout: flexors (channels 1-3) close the hand, extensors (5-7) open it
#let readout = (1, 1, 0.6, 0, -0.6, -1, -1, 0)
#let pose(x) = transpose(x).map(col => col.zip(readout).map(((v, w)) => v * w).sum())

#let forearm(center, offset, skin, r: 0.85) = {
  import cetz.draw: *
  let (x, y) = center
  circle((x, y), radius: r, fill: rgb("#f3e6dc"), stroke: 0.5pt + muted)
  circle((x, y), radius: r - skin, fill: rgb("#fbf6f2"), stroke: none)
  for k in range(n-ch) {
    let a = 90deg - k * 45deg
    circle((x + 0.46 * calc.cos(a), y + 0.46 * calc.sin(a)), radius: 0.09, fill: rgb("#c9dcf3"), stroke: 0.4pt + blue)
  }
  for i in range(n-ch) {
    let a = 90deg - (i + offset) * 45deg
    let (ex, ey) = (x + r * calc.cos(a), y + r * calc.sin(a))
    rect((ex - 0.09, ey - 0.09), (ex + 0.09, ey + 0.09), fill: if i == 0 { orange } else { ink }, stroke: none)
  }
}

#cetz.canvas({
  import cetz.draw: *
  let (fw, fh) = (17.8, 7.8)
  rect((0, 0), (fw, fh), radius: 0.15, stroke: 0.5pt + rgb("#d8d7d2"), fill: white)
  let dashed = (paint: muted, thickness: 0.5pt, dash: "dashed")

  // a: the reference band and the new user's, turned (and over thicker skin)
  let top = 5.6
  letter((0.3, fh - 0.25), [a])
  forearm((1.75, top), 0, 0.08)
  forearm((4.05, top), shift, 0.22)
  content((1.75, top - 1.05), anchor: "north", text(size: 7pt, fill: muted)[reference])
  content((4.05, top - 1.05), anchor: "north", text(size: 7pt, fill: muted)[new user])

  // b: the rotation search over the virtual array
  let (s0, s1, sy0, sy1) = (6.6, 16.9, 4.75, 6.95)
  letter((5.9, fh - 0.25), [b])
  let sx(th) = s0 + (s1 - s0) * th / n-ch
  let (low, high) = (calc.min(..scores), calc.max(..scores))
  let sy(v) = sy0 + (sy1 - sy0) * (v - low) / (high - low)
  line((s0, sy0 - 0.05), (s1, sy0 - 0.05), stroke: 0.5pt + muted)
  line((s0, sy0 - 0.05), (s0, sy1), stroke: 0.5pt + muted)
  for deg in (0, 90, 180, 270, 360) {
    line((sx(deg / 45), sy0 - 0.05), (sx(deg / 45), sy0 - 0.13), stroke: 0.5pt + muted)
    content((sx(deg / 45), sy0 - 0.18), anchor: "north", text(size: 6.5pt, fill: muted)[#deg°])
  }
  line(..thetas.zip(scores).map(((th, v)) => (sx(th), sy(v))), (sx(n-ch), sy(scores.at(0))), stroke: 1pt + blue)
  line((sx(best), sy0 - 0.05), (sx(best), sy(high)), stroke: (paint: orange, thickness: 0.8pt, dash: "dashed"))
  circle((sx(best), sy(high)), radius: 0.07, fill: orange, stroke: none)
  content((sx(best) + 0.12, sy(high) + 0.04), anchor: "south-west", text(fill: orange)[$theta^*$])
  content((s0 - 0.15, (sy0 + sy1) / 2), anchor: "east", $rho$)
  content(((s0 + s1) / 2, sy0 - 0.5), anchor: "north", $theta$)

  // c: the chain on the signals, x -> V(θ*) -> x_θ* -> W -> x̂ -> f
  let (h, wide, op) = (1.8, 1.95, 1.3)
  let cy = 1.95
  let (y0, y1) = (cy - h / 2, cy + h / 2)
  letter((0.3, 3.55), [c])
  let x0 = 0.75
  let v0 = x0 + wide + 0.45
  let r0 = v0 + op + 0.45
  let w0 = r0 + wide + 0.45
  let z0 = w0 + op + 0.45
  let (e0, e1) = (z0 + wide + 0.45, z0 + wide + 1.5)
  heatmap(((x0, y0), (x0 + wide, y1)), x-new)
  heatmap(((v0, cy - op / 2), (v0 + op, cy + op / 2)), turn(best), top: 1)
  heatmap(((r0, y0), (r0 + wide, y1)), x-rot)
  heatmap(((w0, cy - op / 2), (w0 + op, cy + op / 2)), adapter, top: 1.1)
  heatmap(((z0, y0), (z0 + wide, y1)), x-fixed)
  for (a, b) in ((x0 + wide, v0), (v0 + op, r0), (r0 + wide, w0), (w0 + op, z0), (z0 + wide, e0)) {
    arrow((a + 0.06, cy), (b - 0.06, cy))
  }
  rect((e0, cy - 0.5), (e1, cy + 0.5), radius: 0.1, fill: white, stroke: 0.8pt + blue)
  content(((e0 + e1) / 2, cy), $f_theta$)
  math-label((x0 + wide / 2, y0), $x$)
  math-label((v0 + op / 2, cy - op / 2), $V(theta^*)$)
  math-label((r0 + wide / 2, y0), $x_(theta^*)$)
  math-label((w0 + op / 2, cy - op / 2), $W$)
  math-label((z0 + wide / 2, y0), $hat(x) = W x_(theta^*) + b$)
  // links to a and b: the new user's band gives x, θ* gives V
  line((4.05, top - 1.45), (4.05, 3.4), (x0 + wide / 2, 3.4), (x0 + wide / 2, y1 + 0.1), stroke: dashed, mark: (
    end: "stealth",
    fill: muted,
    scale: 0.5,
  ))
  line((sx(best), sy0 - 0.65), (sx(best), 3.75), (v0 + op / 2, 3.75), (v0 + op / 2, cy + op / 2 + 0.1), stroke: (
    paint: orange,
    thickness: 0.5pt,
    dash: "dashed",
  ), mark: (end: "stealth", fill: orange, scale: 0.5))

  // d: the frozen model's output
  let (p0, p1) = (e1 + 0.55, e1 + 3.1)
  letter((p0 - 0.3, 3.55), [d])
  arrow((e1 + 0.06, cy), (p0 - 0.06, cy))
  line((p0, y1), (p0, y0), (p1, y0), stroke: 0.5pt + muted)
  sparkline(((p0 + 0.05, y0 + 0.1), (p1, y1 - 0.1)), (
    trace(pose(x-new), color: orange),
    trace(pose(x-rot), color: blue),
    trace(pose(x-fixed), color: green),
    trace(pose(x-ref), color: ink),
  ))
  content((p1 + 0.15, y1 + 0.05), anchor: "north-west", text(size: 7pt)[
    #text(fill: ink)[$y$] \
    #text(fill: orange)[$f_theta (x)$] \
    #text(fill: blue)[$f_theta (x_(theta^*))$] \
    #text(fill: green)[$f_theta (hat(x))$]
  ])
})
