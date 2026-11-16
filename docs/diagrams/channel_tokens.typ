#import "style.typ": *
#show: setup

// Same geometry as src/femg/electrodes.py: angle around the forearm (0 palmar, 90 radial), z from elbow (0) to wrist (1)
#let ring(n, z) = range(n).map(i => (i * 360 / n, z))
#let grid(rows, columns, center, z0) = {
  let (da, dz) = (0.4 / 24 * 360, 0.4 / 26)
  range(rows)
    .map(r => range(columns).map(c => (center + (c - (columns - 1) / 2) * da, z0 + (r - (rows - 1) / 2) * dz)))
    .flatten()
    .chunks(2)
}
#let layouts = (
  (ring(8, 0.23), blue, [ring of 8 (EMG-FK)]),
  (ring(8, 0.05) + ring(8, 0.2), orange, [2 rings of 8 (DB8, GREAT)]),
  (grid(14, 9, -15, 0.35), green, [grid of 126 (SEEDS)]),
)

// A synthetic attention: query k prefers angle 90 k; weight ~ softmax of angular closeness
#let token-angles = range(8).map(i => i * 45)
#let preferred = (0, 90, 180, 270)
#let weights = preferred.map(p => {
  let raw = token-angles.map(a => calc.exp(2 * calc.cos((a - p) * 1deg)))
  raw.map(w => w / raw.sum())
})

#let token(position, fill) = cetz.draw.circle(position, radius: 0.14, fill: fill, stroke: 0.5pt + ink)
#let latent(position) = cetz.draw.rect(
  (position.at(0) - 0.14, position.at(1) - 0.14),
  (position.at(0) + 0.14, position.at(1) + 0.14),
  fill: rgb("#e6f5ee"),
  stroke: 0.6pt + green,
)
#let box-node(at, size, body, color: blue) = {
  import cetz.draw: *
  let (x, y) = at
  let (w, h) = size
  rect((x - w / 2, y - h / 2), (x + w / 2, y + h / 2), radius: 0.1, fill: white, stroke: 0.8pt + color)
  content(at, align(center, body))
}

#cetz.canvas({
  import cetz.draw: *

  // a: every layout in one frame, the forearm unrolled
  letter((-0.6, 2.1), [a])
  let (ax0, ax1, ay0, ay1) = (0, 4.2, -1.6, 1.6) // angle -180..180, z 0..0.5 (elbow at the top)
  let px(a) = ax0 + (ax1 - ax0) * (calc.rem-euclid(a + 180, 360)) / 360
  let py(z) = ay1 - (ay1 - ay0) * z / 0.5
  rect((ax0, ay0), (ax1, ay1), stroke: 0.4pt + muted)
  for (points, color, _) in layouts {
    for (a, z) in points { circle((px(a), py(z)), radius: 0.035, fill: color, stroke: none) }
  }
  for (a, label) in ((-179.9, [dorsal]), (-90, [ulnar]), (0, [palmar]), (90, [radial]), (179.9, [dorsal])) {
    line((px(a), ay0), (px(a), ay0 - 0.08), stroke: 0.4pt + muted)
    content((px(a), ay0 - 0.1), anchor: "north", text(size: 6pt, fill: muted, label))
  }
  content(((ax0 + ax1) / 2, ay0 - 0.45), anchor: "north", text(size: 7pt)[angle $theta$ around the forearm])
  content((ax0 - 0.1, ay1), anchor: "east", text(size: 6pt, fill: muted)[elbow])
  content((ax0 - 0.1, py(0.5)), anchor: "east", text(size: 6pt, fill: muted)[$z = 0.5$])
  for (i, (_, color, label)) in layouts.enumerate() {
    let y = ay0 - 1.05 - i * 0.3
    circle((ax0 + 0.1, y), radius: 0.05, fill: color, stroke: none)
    content((ax0 + 0.25, y), anchor: "west", text(size: 6.5pt, label))
  }

  // b: one channel -> one token per time step
  letter((5.3, 2.1), [b])
  let (bx, by) = (5.8, 0.9)
  let wave = emg(60, seed: 4)
  let wx(i) = bx + 1.4 * i / 59
  line(..wave.enumerate().map(((i, v)) => (wx(i), by + 0.18 * v)), stroke: 0.5pt + blue)
  content((bx + 0.7, by - 0.45), $x_c (t)$)
  arrow((bx + 1.55, by), (bx + 1.95, by))
  box-node((bx + 2.65, by), (1.4, 0.75), [$g$ \ #text(size: 6pt, fill: muted)[conv, shared]])
  content((bx + 0.7, by - 1.5), $(theta_c, z_c, kappa_c)$)
  arrow((bx + 1.55, by - 1.5), (bx + 1.95, by - 1.5))
  box-node((bx + 2.65, by - 1.5), (1.4, 0.75), [$phi$ \ #text(size: 6pt, fill: muted)[Fourier + MLP]])
  circle((bx + 4.0, by - 0.75), radius: 0.2, fill: white, stroke: 0.6pt + ink)
  content((bx + 4.0, by - 0.75), $+$)
  arrow((bx + 3.35, by), (bx + 3.85, by - 0.62))
  arrow((bx + 3.35, by - 1.5), (bx + 3.85, by - 0.88))
  arrow((bx + 4.2, by - 0.75), (bx + 4.8, by - 0.75))
  token((bx + 5.0, by - 0.75), rgb("#c9dcf3"))
  content((bx + 5.25, by - 0.75), anchor: "west", $u_c (t)$)
  content((bx + 2.9, by - 2.35), text(size: 7.5pt)[
    $u_c (t) = g(x_c)(t) + phi(theta_c, z_c, kappa_c)$
  ])
  content((bx + 2.9, by - 2.75), anchor: "north", text(size: 6.5pt, fill: muted, align(center)[
    $phi = "MLP"(sin k theta, cos k theta, sin 2^i pi z, cos 2^i pi z) + E_kappa$ \
    unknown position: learned $e_"unk"$ · $kappa$: bipolar / monopolar
  ]))

  // c: C channel tokens -> K latents, whatever C
  letter((-0.6, -3.6), [c])
  let (tx, qx, zx) = (1.0, 5.6, 7.6)
  let ty(i) = -4.15 - i * 0.36
  let qy(k) = -4.6 - k * 0.55
  for (k, row) in weights.enumerate() {
    for (i, w) in row.enumerate() {
      line((tx + 1.55, ty(i)), (qx - 0.16, qy(k)), stroke: (paint: green, thickness: 3pt * w))
    }
  }
  for i in range(8) {
    token((tx, ty(i)), heat-color(0.25 + 0.6 * calc.abs(calc.cos(i * 22.5deg))))
    content((tx - 0.25, ty(i)), anchor: "east", text(size: 5.5pt, fill: muted)[#(i * 45)°])
  }
  rect((tx + 0.35, ty(0) + 0.2), (tx + 1.35, ty(7) - 0.2), radius: 0.1, fill: white, stroke: 0.8pt + blue)
  content((tx + 0.85, ty(3.5)), text(size: 6.5pt, align(center)[self-attn\ over\ channels]))
  for i in range(8) {
    line((tx + 0.15, ty(i)), (tx + 0.35, ty(i)), stroke: 0.4pt + ink)
    line((tx + 1.35, ty(i)), (tx + 1.55, ty(i)), stroke: 0.4pt + ink)
  }
  for k in range(4) { latent((qx, qy(k))) }
  content((tx, ty(7) - 0.45), anchor: "north", text(size: 7pt)[$C$ tokens ($C = 7 … 134$)])
  content(((tx + qx) / 2 + 0.9, qy(3) - 0.45), anchor: "north", text(size: 7pt)[cross-attention: $K$ learned queries])
  content((qx, qy(0) + 0.35), anchor: "south", text(size: 7pt)[$q_1 … q_K$])
  arrow((qx + 0.3, qy(1.5)), (zx - 0.1, qy(1.5)))
  heatmap(((zx, qy(1.5) - 0.7), (zx + 1.0, qy(1.5) + 0.7)), range(4).map(k => range(6).map(j => calc.sin(k * 1.3 + j))))
  content((zx + 0.5, qy(1.5) - 0.75), anchor: "north", $Z(t) in RR^(K times D)$)
  arrow((zx + 1.1, qy(1.5)), (zx + 1.7, qy(1.5)))
  content((zx + 1.75, qy(1.5)), anchor: "west", text(size: 7pt)[temporal model,\ pose head])

  // d: calibration, encoder frozen
  letter((11.2, -3.6), [d])
  let (cx, cy, r) = (12.6, -5.3, 0.65)
  circle((cx, cy), radius: r, fill: rgb("#f3e6dc"), stroke: 0.5pt + muted)
  for i in range(8) {
    let a = 90deg - i * 45deg
    let b = a - 25deg
    rect(
      (cx + r * calc.cos(a) - 0.06, cy + r * calc.sin(a) - 0.06),
      (cx + r * calc.cos(a) + 0.06, cy + r * calc.sin(a) + 0.06),
      fill: rgb("#c8c7c2"),
      stroke: none,
    )
    rect(
      (cx + r * calc.cos(b) - 0.06, cy + r * calc.sin(b) - 0.06),
      (cx + r * calc.cos(b) + 0.06, cy + r * calc.sin(b) + 0.06),
      fill: orange,
      stroke: none,
    )
  }
  arc((cx, cy), start: 90deg, delta: -25deg, radius: r + 0.25, anchor: "origin", stroke: 0.6pt + orange, mark: (
    end: "stealth",
    fill: orange,
    scale: 0.5,
  ))
  content((cx + 0.45, cy + r + 0.25), anchor: "south-west", text(fill: orange, size: 7pt)[$Delta theta$])
  content((cx, cy - r - 0.2), anchor: "north", text(size: 6pt, fill: muted)[expected (grey), worn (orange)])
  content((cx, cy - r - 0.55), anchor: "north", align(center, text(size: 7.5pt)[
    $min_(Delta theta, Delta z, bold(g)) sum_t ell(h(f_theta (bold(g) dot.o x, p + Delta p)), y)$ \
    #text(size: 6.5pt, fill: muted)[$f_theta$, $h$ frozen · $2 + C$ numbers]
  ]))
})
