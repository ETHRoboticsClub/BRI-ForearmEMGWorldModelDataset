#import "style.typ": *
#show: setup

// The kernel of src/femg/virtual_array.py with the settings of src/femg/config/virtual_array.yaml, on NinaPro DB8
#let (circumference, length) = (24.0, 26.0) // cm, femg.electrodes
#let (n-angles, height, around-sd, along-sd) = (8, 0.13, 1.5, 1.5)
#let wrap(a) = calc.rem-euclid(a + 180, 360) - 180
#let kernel(at, real) = {
  let around = wrap(real.at(0) - at.at(0)) / 360 * circumference
  let along = (real.at(1) - at.at(1)) * length
  calc.exp(-0.5 * (calc.pow(around / around-sd, 2) + calc.pow(along / along-sd, 2)))
}
#let virtual = range(n-angles).map(k => (k * 45, height))
#let ring(z) = range(8).map(i => (i * 45, z))
#let real = ring(0.115) + ring(0.212) // NinaPro DB8: two rows, 3 and 5.5 cm below the elbow
#let weights = virtual.map(v => {
  let row = real.map(r => kernel(v, r))
  let total = row.sum()
  row.map(w => w / total)
})
#let row-color(i) = if i < 8 { blue } else { green }
#let accent = orange
#let picked = 0 // the virtual electrode shown in a (palmar)

#cetz.canvas({
  import cetz.draw: *

  // Frame
  let (fw, fh) = (17.6, 6.4)
  rect((0, 0), (fw, fh), radius: 0.15, stroke: 0.5pt + rgb("#d8d7d2"), fill: white)

  // a: the skin unrolled, the real electrodes, the virtual ring, one virtual electrode as a weighted mean
  let (sx, sy, sw, sh) = (1.35, 1.0, 7.6, 4.3)
  let z-max = 0.3
  let px(a) = if a == 180 { sx + sw } else { sx + (wrap(a) + 180) / 360 * sw } // dorsal on both edges
  let py(z) = sy + sh - z / z-max * sh
  letter((0.3, fh - 0.25), [a])
  rect((sx, sy), (sx + sw, sy + sh), fill: rgb("#f3e6dc"), stroke: none)
  line((sx, py(height)), (sx + sw, py(height)), stroke: (paint: muted, thickness: 0.5pt, dash: "dashed"))

  let v = virtual.at(picked)
  let (vx, vy) = (px(v.at(0)), py(v.at(1)))
  circle((vx, vy), radius: (around-sd / circumference * sw, along-sd / length / z-max * sh), fill: accent.transparentize(
    82%,
  ), stroke: (paint: accent, thickness: 0.5pt, dash: "dotted"))
  for (r, w) in real.zip(weights.at(picked)) {
    if w > 0.01 {
      line((vx, vy), (px(r.at(0)), py(r.at(1))), stroke: (paint: accent, thickness: 0.3pt + 3.5pt * w))
    }
  }
  for (k, p) in virtual.enumerate() {
    let (x, y) = (px(p.at(0)), py(p.at(1)))
    let fill = if k == picked { accent } else { rgb("#a8a7a2") }
    rect((x - 0.13, y - 0.13), (x + 0.13, y + 0.13), fill: fill, stroke: 0.5pt + white)
    if wrap(p.at(0)) == -180 {
      rect((sx - 0.13, y - 0.13), (sx + 0.13, y + 0.13), fill: fill, stroke: 0.5pt + white)
    }
  }
  for (i, r) in real.enumerate() {
    let (x, y) = (px(r.at(0)), py(r.at(1)))
    circle((x, y), radius: 0.09, fill: row-color(i), stroke: 0.6pt + white)
    if wrap(r.at(0)) == -180 { circle((sx, y), radius: 0.09, fill: row-color(i), stroke: 0.6pt + white) }
  }
  // axes: around the forearm, along it
  for (a, label) in ((-180, [dorsal]), (-90, [ulnar]), (0, [palmar]), (90, [radial]), (180, [dorsal])) {
    line((px(a), sy), (px(a), sy - 0.08), stroke: 0.5pt + muted)
    content((px(a), sy - 0.15), anchor: "north", text(size: 6.5pt, fill: muted, label))
  }
  line((sx - 0.25, sy + sh), (sx - 0.25, sy + 0.4), stroke: 0.5pt + muted, mark: (end: "stealth", fill: muted, scale: 0.5))
  content((sx - 0.35, sy + sh), anchor: "east", text(size: 6.5pt, fill: muted)[elbow])
  content((sx - 0.35, sy + 0.4), anchor: "east", text(size: 6.5pt, fill: muted)[wrist])

  // b: the same as one matrix, v = W x
  let cell = 0.3
  let (wx, wy) = (11.7, 1.9) // bottom-left of W
  let (mw, mh) = (real.len() * cell, n-angles * cell)
  letter((9.75, fh - 0.25), [b])
  heatmap(((wx, wy), (wx + mw, wy + mh)), weights, top: 0.6)
  rect((wx, wy + mh - (picked + 1) * cell), (wx + mw, wy + mh - picked * cell), stroke: 1.2pt + accent)
  line((wx + 8 * cell, wy), (wx + 8 * cell, wy + mh), stroke: 0.5pt + white)
  // x: one dot per real electrode, above its column
  for i in range(real.len()) {
    circle((wx + (i + 0.5) * cell, wy + mh + 0.32), radius: 0.09, fill: row-color(i), stroke: 0.6pt + white)
  }
  math-label((wx + mw / 2, wy + mh + 0.52), $x(t) in RR^16$, anchor: "south")
  // v: one square per virtual electrode, beside its row
  let vx0 = wx - 1.25
  for k in range(n-angles) {
    let y = wy + mh - (k + 0.5) * cell
    let fill = if k == picked { accent } else { rgb("#a8a7a2") }
    rect((vx0 - 0.11, y - 0.11), (vx0 + 0.11, y + 0.11), fill: fill, stroke: 0.5pt + white)
  }
  content((vx0 + 0.6, wy + mh / 2), text(size: 12pt, $=$))
  math-label((vx0, wy - 0.05), $v(t) in RR^8$)
  math-label((wx + mw / 2, wy - 0.05), $W in RR^(8 times 16)$)
})
