#import "style.typ": *
#show: setup

// The kernel of src/femg/virtual_array.py with the settings of src/femg/config/virtual_array.yaml
#let (circumference, length) = (24.0, 26.0) // cm, femg.electrodes
#let (n-angles, height, around-sd, along-sd) = (8, 0.13, 1.5, 1.5)
#let virtual = range(n-angles).map(k => (k * 45, height))
#let wrap(a) = calc.rem-euclid(a + 180, 360) - 180
#let kernel(at, real) = {
  let around = wrap(real.at(0) - at.at(0)) / 360 * circumference
  let along = (real.at(1) - at.at(1)) * length
  calc.exp(-0.5 * (calc.pow(around / around-sd, 2) + calc.pow(along / along-sd, 2)))
}
// rows: virtual electrodes, columns: real channels; each row sums to 1
#let weights(real) = virtual.map(v => {
  let row = real.map(r => kernel(v, r))
  let total = row.sum()
  row.map(w => w / total)
})
// monopolar: minus the common average, (I - 1/C) on the right
#let common-average(rows) = rows.map(row => {
  let mean = row.sum() / row.len()
  row.map(w => w - mean)
})

#let ring(z, first: 0) = range(8).map(i => (first + i * 45, z))
#let layouts = (
  (name: [EMG-FK], detail: [8 monopolar, 20 cm from the wrist], real: ring(0.231), spots: ()),
  (name: [NinaPro DB8], detail: [16 bipolar, rows 3, 5.5 cm from the elbow], real: ring(0.115) + ring(0.212), spots: ()),
  (name: [GREAT], detail: [16 bipolar, rows staggered (spacing assumed as DB8)], real: ring(0.115) + ring(
      0.212,
      first: -22.5,
    ), spots: ()),
  (name: [NinaPro DB2, DB7], detail: [12 bipolar: ring + FDS, EDC], real: ring(0.05) + ((345, 0.3), (180, 0.3)), spots: (
    8,
    9,
  )),
)
// DB2: built from its ring only (layouts: [ring]); the FDS, EDC, biceps and triceps columns are 0
#let db2-matrix = weights(ring(0.05)).map(row => row + (0, 0, 0, 0))

// Skin unrolled: angle across (palmar in the middle), elbow (z 0) at the top
#let skin(origin, size, layout, highlight: none) = {
  import cetz.draw: *
  let ((x0, y0), (w, h)) = (origin, size)
  let z-max = 0.4
  let px(a) = x0 + (wrap(a) + 180) / 360 * w
  let py(z) = y0 + h - z / z-max * h
  rect((x0, y0), (x0 + w, y0 + h), fill: rgb("#f3e6dc"), stroke: none)
  for (a, label) in ((-180, [dorsal]), (-90, [ulnar]), (0, [palmar]), (90, [radial])) {
    content((px(a) + if a == -180 { 0.02 } else { 0 }, y0 - 0.05), anchor: if a == -180 { "north-west" } else {
      "north"
    }, text(size: 5pt, fill: muted, label))
  }
  content((x0 + w + 0.05, py(0)), anchor: "west", text(size: 5pt, fill: muted)[elbow])
  content((x0 + w + 0.05, py(height)), anchor: "west", text(size: 5pt, fill: muted)[z #height])
  line((x0, py(height)), (x0 + w, py(height)), stroke: (paint: muted, thickness: 0.4pt, dash: "dashed"))
  if highlight != none {
    let (v, real) = highlight
    let (vx, vy) = (px(v.at(0)), py(v.at(1)))
    // the kernel: 1 sd around and along the arm
    circle((vx, vy), radius: (around-sd / circumference * w, along-sd / length / z-max * h), fill: rgb("#eaf2fc"), stroke: (
      paint: blue,
      thickness: 0.4pt,
      dash: "dotted",
    ))
    let row = weights(real).at(calc.quo(v.at(0), 45))
    for (r, wt) in real.zip(row) {
      if wt > 0.02 and r.at(1) >= 0 {
        line((vx, vy), (px(r.at(0)), py(r.at(1))), stroke: (paint: blue, thickness: 0.3pt + 3pt * wt))
        let above = r.at(1) < v.at(1) - 0.05
        content((px(r.at(0)), py(r.at(1)) + if above { 0.12 } else { -0.22 }), anchor: if above { "south" } else {
          "north"
        }, text(size: 5.5pt, fill: blue, weight: "bold")[#calc.round(wt, digits: 2)])
      }
    }
  }
  for v in virtual {
    let (vx, vy) = (px(v.at(0)), py(v.at(1)))
    rect((vx - 0.1, vy - 0.1), (vx + 0.1, vy + 0.1), fill: rgb("#a8a7a2").transparentize(40%), stroke: none)
    if wrap(v.at(0)) == -180 {
      rect((px(180) - 0.1, vy - 0.1), (px(180) + 0.1, vy + 0.1), fill: rgb("#a8a7a2").transparentize(40%), stroke: none)
    }
  }
  for (i, r) in layout.real.enumerate() {
    if r.at(1) >= 0 {
      let fill = if i in layout.spots { white } else { blue }
      circle((px(r.at(0)), py(r.at(1))), radius: 0.065, fill: fill, stroke: 0.5pt + if i in layout.spots { orange } else { white })
      if wrap(r.at(0)) == -180 and i not in layout.spots {
        circle((px(180), py(r.at(1))), radius: 0.065, fill: fill, stroke: 0.4pt + white)
      }
    }
  }
}

#cetz.canvas({
  import cetz.draw: *
  let (w, h) = (3.6, 1.35)

  title((-0.3, 3.0), [Virtual array: any electrode layout → the same 8 channels (`VirtualArray`)])

  // a: the layouts and the virtual ring
  letter((-0.3, 2.35), [a])
  for (i, layout) in layouts.enumerate() {
    let y = 0.55 - i * 2.15
    content((0, y + h + 0.3), anchor: "west", text(size: 7pt, weight: "bold", layout.name) + text(
      size: 6pt,
      fill: muted,
    )[ · #layout.detail])
    skin((0, y), (w, h), layout)
  }
  content((0, -6.4), anchor: "north-west", text(size: 6pt, fill: muted)[
    #box(std.circle(radius: 2.5pt, fill: blue, stroke: none)) ring electrode #std.h(6pt)
    #box(std.circle(radius: 2.5pt, fill: white, stroke: 0.6pt + orange)) spot (not used) #std.h(6pt)
    #box(std.rect(width: 6pt, height: 6pt, fill: rgb("#a8a7a2").transparentize(40%), stroke: none)) virtual electrode
  ])

  // b: one virtual electrode
  letter((5.2, 2.35), [b])
  let (bx, bw, bh) = (5.6, 5.4, 2.7)
  content((bx, 2.2), anchor: "north-west", text(size: 7pt, weight: "bold")[One virtual electrode: a weighted mean])
  content((bx, 1.85), anchor: "north-west", text(size: 6pt, fill: muted)[
    NinaPro DB8, palmar: Gaussian of the skin distance (dotted: 1 sd,\ #around-sd cm around and along), weights sum to 1
  ])
  skin((bx, -1.75), (bw, bh), layouts.at(1), highlight: (virtual.at(0), layouts.at(1).real))
  content((bx, -2.4), anchor: "north-west", text(size: 6pt, fill: muted)[
    The row 0.4 cm away dominates; the row 2.1 cm away adds some.\
    Only ring electrodes are used: NinaPro's FDS / EDC spots (approximate\
    positions) would replace ring channels.
  ])

  // c: one matrix per layout
  letter((12.4, 2.35), [c])
  let cx = 12.8
  content((cx, 2.2), anchor: "north-west", text(size: 7pt, weight: "bold")[One matrix per layout])
  content((cx, 1.85), anchor: "north-west", text(size: 6pt, fill: muted)[
    $v(t) = G dot W dot R dot "emg"(t) slash "active"$, every sample: 8 × C
  ])
  let cell = 0.2
  let matrices = (
    ([NinaPro DB2 (8 × 12)], db2-matrix, [ring 1-8 · FDS · EDC · biceps · triceps (0)]),
    ([EMG-FK (8 × 8), after the common average], common-average(weights(layouts.at(0).real)), [band 1-8]),
  )
  let y = 1.0
  for (name, m, columns) in matrices {
    let (mw, mh) = (m.at(0).len() * cell, m.len() * cell)
    content((cx, y + 0.05), anchor: "south-west", text(size: 6.5pt, name))
    heatmap(((cx, y - mh), (cx + mw, y)), m, top: 0.6)
    for (k, v) in virtual.enumerate() {
      content((cx - 0.05, y - (k + 0.5) * cell), anchor: "east", text(size: 4.5pt, fill: muted)[#v.at(0)°])
    }
    content((cx, y - mh - 0.05), anchor: "north-west", text(size: 5pt, fill: muted, columns))
    y -= mh + 0.85
  }
  content((cx, y + 0.35), anchor: "north-west", text(size: 6pt, fill: muted)[
    #box(std.rect(width: 6pt, height: 6pt, fill: blue, stroke: none)) weight > 0 #std.h(6pt)
    #box(std.rect(width: 6pt, height: 6pt, fill: orange, stroke: none)) < 0 (the common average)\
    $R$: minus the common average (monopolar only) \
    $W$: the weights of b, a row per virtual electrode\
    $G$: each virtual channel ÷ its own active level\
    `rotation=θ` moves the real electrodes: 45° rolls $v$ by one
  ])
})
