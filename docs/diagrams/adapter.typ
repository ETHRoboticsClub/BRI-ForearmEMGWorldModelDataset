#import "style.typ": *
#show: setup

// Synthetic forearm: 8 muscles around the arm, muscle k fires at time slot k (a staircase in the channel heatmaps).
// Electrode i of the ring picks up muscle k with a Gaussian of their distance around the arm (volume conduction).
// The new user wears the band rotated by 1.2 electrodes (54°) over thicker skin (wider pickup).
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
#let identity = range(n-ch).map(i => range(n-ch).map(j => if i == j { 1.0 } else { 0.0 }))
// What calibration converges to: the ridge fit of W · displaced ≈ reference
#let adapter = matmul(
  matmul(reference, transpose(displaced)),
  inverse(matmul(displaced, transpose(displaced)).enumerate().map(((i, row)) => row
    .enumerate()
    .map(((j, v)) => if i == j { v + 0.05 } else { v }))),
)

#let x-ref = matmul(reference, sources)
#let x-new = matmul(displaced, sources)
#let x-fixed = matmul(adapter, x-new)

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
  let (y0, y1) = (-h / 2, h / 2)
  let t-star = 17 // the time step zoomed in below
  let col(x, j) = x.map(row => row.at(j))

  // a: the displacement
  letter((-0.9, 1.75), [a])
  forearm((0, 0.2), 0, 0.07)
  forearm((1.65, 0.2), shift, 0.18)
  math-label((0, -0.55), text(size: 7pt)[reference])
  math-label((1.65, -0.55), text(size: 7pt)[new user])

  // b: the adapter, x̂ = W x
  letter((3.0, 1.75), [b])
  let (w0, w1) = (3.4, 3.4 + h)
  let (x0, x1) = (w1 + 0.55, w1 + 0.55 + 2.6)
  let (z0, z1) = (x1 + 0.55, x1 + 0.55 + 2.6)
  heatmap(((w0, y0), (w1, y1)), adapter, top: 1.1)
  heatmap(((x0, y0), (x1, y1)), x-new)
  heatmap(((z0, y0), (z1, y1)), x-fixed)
  content(((w1 + x0) / 2, 0), text(size: 11pt, $dot$))
  content(((x1 + z0) / 2, 0), text(size: 11pt, $=$))
  math-label(((w0 + w1) / 2, y0), $W in RR^(8 times 8)$)
  math-label(((x0 + x1) / 2, y0), $x in RR^(8 times T)$)
  math-label(((z0 + z1) / 2, y0), $hat(x) = W x + b$)
  content((x0 - 0.12, y1 - 0.12), anchor: "east", text(size: 5.5pt, fill: muted)[1])
  content((x0 - 0.12, y0 + 0.12), anchor: "east", text(size: 5.5pt, fill: muted)[8])
  line((x0, y1 + 0.15), (x0 + 0.8, y1 + 0.15), stroke: 0.4pt + muted, mark: (end: "stealth", fill: muted, scale: 0.4))
  content((x0 + 0.85, y1 + 0.15), anchor: "west", text(size: 5.5pt, fill: muted)[$t$])

  // the zoomed time step, outlined in x and x̂
  let dt = 2.6 / n-t
  let cx(left) = left + (t-star + 0.5) * dt
  for left in (x0, z0) {
    rect((left + t-star * dt, y0), (left + (t-star + 1) * dt, y1), stroke: 0.7pt + ink)
  }

  // c: the same W at every time step, as a graph: x(t) -> x̂(t) (a 1×1 convolution over time)
  let (g-top, gap) = (-1.95, 0.27)
  let ny(i) = g-top - 0.15 - i * gap
  let (in-x, out-x) = (cx(x0), cx(z0))
  line((cx(x0), y0 - 0.45), (in-x, g-top), stroke: (paint: muted, thickness: 0.4pt, dash: "dotted"))
  line((cx(z0), y0 - 0.45), (out-x, g-top), stroke: (paint: muted, thickness: 0.4pt, dash: "dotted"))
  for i in range(n-ch) {
    for j in range(n-ch) {
      let w = adapter.at(i).at(j)
      if calc.abs(w) > 0.06 {
        line((in-x + 0.12, ny(j)), (out-x - 0.12, ny(i)), stroke: (
          paint: if w > 0 { blue } else { orange },
          thickness: 0.2pt + 1.6pt * calc.min(calc.abs(w), 1.1),
        ))
      }
    }
  }
  let x-col = col(x-new, t-star)
  let z-col = col(x-fixed, t-star)
  for i in range(n-ch) {
    circle((in-x, ny(i)), radius: 0.1, fill: heat-color(x-col.at(i)), stroke: 0.4pt + ink)
    circle((out-x, ny(i)), radius: 0.1, fill: heat-color(z-col.at(i)), stroke: 0.4pt + ink)
  }
  math-label((in-x - 0.2, ny(3.5)), $x(t)$, anchor: "east")
  math-label((out-x + 0.2, ny(3.5)), $hat(x)(t)$, anchor: "west")
  math-label(((in-x + out-x) / 2, ny(7) - 0.1), $hat(x)_i (t) = sum_j W_(i j) x_j (t) + b_i quad forall t$ + std.h(4pt) + text(size: 6.5pt, fill: muted)[(1×1 conv)])

  // d: frozen encoder and its prediction
  let (e0, e1) = (z1 + 0.7, z1 + 2.0)
  arrow((z1 + 0.1, 0), (e0, 0))
  rect((e0, -0.55), (e1, 0.55), radius: 0.1, fill: white, stroke: 0.8pt + blue)
  content(((e0 + e1) / 2, 0.1), $f_theta$)
  content(((e0 + e1) / 2, -0.25), text(size: 5.5pt, fill: muted)[frozen])
  letter((e1 + 0.45, 1.75), [c])
  let (p0, p1) = (e1 + 0.75, e1 + 3.6)
  arrow((e1, 0), (p0 - 0.05, 0))
  line((p0, y1), (p0, y0), (p1, y0), stroke: 0.4pt + muted)
  sparkline(((p0 + 0.05, y0 + 0.1), (p1, y1 - 0.1)), (
    trace(pose(x-new), color: orange),
    trace(pose(x-fixed), color: green),
    trace(pose(x-ref), color: ink),
  ))
  math-label(((p0 + p1) / 2, y0), $hat(y)$)
  content((p1 + 0.1, y1 - 0.1), anchor: "north-west", text(size: 6pt)[
    #text(fill: ink)[true] \ #text(fill: green)[with $W$] \ #text(fill: orange)[without]
  ])

  math-label((out-x + 1.5, ny(3.5)), anchor: "west", text(size: 7pt)[
    Calibration: $min_(W, b) sum_t ell(f_theta (W x(t) + b), y(t))$, $theta$ fixed
  ])
})
