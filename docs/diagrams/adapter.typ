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

#let forearm(center, offset, skin, r: 0.62) = {
  import cetz.draw: *
  let (x, y) = center
  circle((x, y), radius: r, fill: rgb("#f3e6dc"), stroke: 0.5pt + muted)
  circle((x, y), radius: r - skin, fill: rgb("#fbf6f2"), stroke: none)
  for k in range(n-ch) {
    let a = 90deg - k * 45deg
    circle((x + 0.34 * calc.cos(a), y + 0.34 * calc.sin(a)), radius: 0.07, fill: rgb("#c9dcf3"), stroke: 0.4pt + blue)
  }
  for i in range(n-ch) {
    let a = 90deg - (i + offset) * 45deg
    let (ex, ey) = (x + r * calc.cos(a), y + r * calc.sin(a))
    rect((ex - 0.07, ey - 0.07), (ex + 0.07, ey + 0.07), fill: if i == 0 { orange } else { ink }, stroke: none)
  }
}


#cetz.canvas({
  import cetz.draw: *
  let h = 2.0 // heatmap height (8 channels)
  let wide = 2.4 // heatmap width (time)
  let (y0, y1) = (-h / 2, h / 2)

  title((-0.9, 3.0), [Per-user calibration: rotation from the virtual array, then a small adapter W])

  // a: the displacement
  letter((-0.9, 2.3), [a])
  forearm((0, 0.6), 0, 0.07)
  forearm((1.65, 0.6), shift, 0.18)
  math-label((0, -0.15), text(size: 7pt)[reference])
  math-label((1.65, -0.15), text(size: 7pt)[new user])
  content((-0.7, -0.75), anchor: "north-west", text(size: 6pt, fill: muted)[
    band turned by 54°, thicker skin\ (wider pickup)
  ])

  // b: the rotation search
  letter((3.2, 2.3), [b])
  let (s0, s1) = (3.6, 8.4)
  let (sy0, sy1) = (-0.6, 1.8)
  content((s0, 2.2), anchor: "north-west", text(size: 7pt, weight: "bold")[Step 3: find the rotation])
  let sx(th) = s0 + (s1 - s0) * th / n-ch
  let (low, high) = (calc.min(..scores), calc.max(..scores))
  let sy(v) = sy0 + (sy1 - 0.5 - sy0) * (v - low) / (high - low)
  line((s0, sy0), (s1, sy0), stroke: 0.4pt + muted)
  line(..thetas.zip(scores).map(((th, v)) => (sx(th), sy(v))), stroke: 0.8pt + blue)
  line((sx(best), sy0), (sx(best), sy(high)), stroke: (paint: orange, thickness: 0.6pt, dash: "dashed"))
  content((sx(best) + 0.12, sy(high) + 0.12), anchor: "south-west", text(size: 6pt, fill: orange)[$theta^* = #calc.round(best * 45)°$])
  for deg in (0, 90, 180, 270) {
    content((sx(deg / 45), sy0 - 0.05), anchor: "north", text(size: 5pt, fill: muted)[#deg°])
  }
  content(((s0 + s1) / 2, sy0 - 0.35), anchor: "north", text(size: 6pt)[
    $"corr"("VirtualArray"(theta) x, "templates")$, any $theta$
  ])

  // c: the chain on the signals: x -> [V(θ*)] -> x_θ* -> [W] -> x̂
  let cy = -3.5
  letter((-0.9, cy + 1.75), [c])
  content((-0.5, cy + 1.65), anchor: "north-west", text(size: 7pt, weight: "bold")[Step 4: undo it, then fit W on what is left])
  let op = 1.5 // operator (8 × 8) size
  let x0 = -0.5
  let v0 = x0 + wide + 0.5
  let r0 = v0 + op + 0.5
  let w0 = r0 + wide + 0.5
  let z0 = w0 + op + 0.5
  heatmap(((x0, cy + y0), (x0 + wide, cy + y1)), x-new)
  heatmap(((v0, cy - op / 2), (v0 + op, cy + op / 2)), turn(best), top: 1)
  heatmap(((r0, cy + y0), (r0 + wide, cy + y1)), x-rot)
  heatmap(((w0, cy - op / 2), (w0 + op, cy + op / 2)), adapter, top: 1.1)
  heatmap(((z0, cy + y0), (z0 + wide, cy + y1)), x-fixed)
  arrow((x0 + wide + 0.05, cy), (v0 - 0.05, cy))
  arrow((v0 + op + 0.05, cy), (r0 - 0.05, cy))
  arrow((r0 + wide + 0.05, cy), (w0 - 0.05, cy))
  arrow((w0 + op + 0.05, cy), (z0 - 0.05, cy))
  math-label((x0 + wide / 2, cy + y0), $x$ + text(size: 6pt, fill: muted)[ as worn])
  math-label((v0 + op / 2, cy - op / 2), $V(theta^*)$ + text(size: 6pt, fill: muted)[\ virtual array])
  math-label((r0 + wide / 2, cy + y0), $x_(theta^*)$ + text(size: 6pt, fill: muted)[ rotation undone])
  math-label((w0 + op / 2, cy - op / 2), $W$ + text(size: 6pt, fill: muted)[\ ≈ identity])
  math-label((z0 + wide / 2, cy + y0), $hat(x) = W x_(theta^*) + b$)
  content((x0 - 0.12, cy + y1 - 0.12), anchor: "east", text(size: 5.5pt, fill: muted)[1])
  content((x0 - 0.12, cy + y0 + 0.12), anchor: "east", text(size: 5.5pt, fill: muted)[8])
  line((x0, cy + y1 + 0.15), (x0 + 0.8, cy + y1 + 0.15), stroke: 0.4pt + muted, mark: (
    end: "stealth",
    fill: muted,
    scale: 0.4,
  ))
  content((x0 + 0.85, cy + y1 + 0.15), anchor: "west", text(size: 5.5pt, fill: muted)[$t$])

  // d: frozen encoder and its prediction
  let z1 = z0 + wide
  let (e0, e1) = (z1 + 0.6, z1 + 1.9)
  arrow((z1 + 0.1, cy), (e0, cy))
  rect((e0, cy - 0.55), (e1, cy + 0.55), radius: 0.1, fill: white, stroke: 0.8pt + blue)
  content(((e0 + e1) / 2, cy + 0.1), $f_theta$)
  content(((e0 + e1) / 2, cy - 0.25), text(size: 5.5pt, fill: muted)[frozen])
  letter((e1 + 0.45, cy + 1.75), [d])
  let (p0, p1) = (e1 + 0.75, e1 + 3.8)
  arrow((e1, cy), (p0 - 0.05, cy))
  line((p0, cy + y1), (p0, cy + y0), (p1, cy + y0), stroke: 0.4pt + muted)
  sparkline(((p0 + 0.05, cy + y0 + 0.1), (p1, cy + y1 - 0.1)), (
    trace(pose(x-new), color: orange),
    trace(pose(x-rot), color: blue),
    trace(pose(x-fixed), color: green),
    trace(pose(x-ref), color: ink),
  ))
  math-label(((p0 + p1) / 2, cy + y0), $hat(y)$)
  content((p1 + 0.1, cy + y1 - 0.1), anchor: "north-west", text(size: 6pt)[
    #text(fill: ink)[true] \ #text(fill: green)[rotation + $W$] \ #text(fill: blue)[rotation only] \ #text(fill: orange)[as worn]
  ])

  // the protocol
  let (q0, q1) = (9.3, 2.2)
  content((q0, q1), anchor: "north-west", text(size: 7pt, weight: "bold")[Calibration of a new user (about 1 min)])
  content((q0, q1 - 0.4), anchor: "north-west", text(size: 6.5pt)[
    + Guided actions: rest, fist, open hand, wrist flexion / extension.
    + Gains: `Normalizer` anchors of this recording (active level per channel).
    + Rotation: $theta^* = arg max_theta "corr"("VirtualArray"(theta) x, "templates")$, the band's\
      angle in any step (the templates: the same actions in the training data).
    + Adapter: $min_(W, b) sum_t ell(f_theta (W x_(theta^*)(t) + b), y(t))$, $W$ from the identity,\
      $theta$ fixed: only what the rotation leaves (skin, pickup width).
  ])
})
