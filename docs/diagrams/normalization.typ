#import "style.typ": *
#show: setup

#let n = 400
#let window = 10
#let raw-dry = emg(n, gain: 5, seed: 1)
#let raw-bipolar = emg(n, gain: 1, seed: 2)
#let env-dry = moving-rms(raw-dry, window)
#let env-bipolar = moving-rms(raw-bipolar, window)
#let active-dry = percentile(env-dry, 95)
#let active-bipolar = percentile(env-bipolar, 95)

#let angle-calibrated = samples(n).map(t => 0.1 + 1.3 * flexion(t))
#let angle-raw-glove = samples(n).map(t => 3.0 + 1.0 * flexion(t, shift: 0.04))
#let anchors(xs) = (percentile(xs, 2), percentile(xs, 98))

#cetz.canvas({
  import cetz.draw: *
  let (fw, fh) = (16.6, 6.8)
  frame(fw, fh)
  let size = (2.9, 1.8)
  let half = size.at(0) / 2
  let (x1, x2, x3, x4) = (2.0, 5.95, 9.85, 13.8)
  let (emg-y, kin-y) = (4.65, 1.6)
  let op = (2.5, 0.75) // the operation applied with the anchors

  // a: EMG, per channel
  row-label((0.3, fh - 0.3), [a], [EMG, per channel], color: muted)
  card((x1, emg-y), [two devices], size: size, kind: "data", traces: (
    trace(raw-dry),
    trace(raw-bipolar, color: orange),
  ))
  card((x2, emg-y), [RMS · anchors], size: size, kind: "stats", log: true, traces: (
    trace(env-dry),
    trace(env-bipolar, color: orange),
  ), levels: (
    level(active-dry, color: blue, label: [$a_95$]),
    level(percentile(env-dry, 10), color: blue, label: [$a_10$]),
    level(active-bipolar, color: orange),
    level(percentile(env-bipolar, 10), color: orange),
  ))
  card((x3, emg-y), [$x slash a_95$], size: op)
  card((x4, emg-y), [same scale], size: size, kind: "result", traces: (
    trace(rescale(env-dry, 0, active-dry)),
    trace(rescale(env-bipolar, 0, active-bipolar), color: orange),
  ), levels: (level(1, label: [1]),))
  for (a, b, wa, wb) in ((x1, x2, half, half), (x2, x3, half, op.at(0) / 2), (x3, x4, op.at(0) / 2, half)) {
    arrow((a + wa, emg-y), (b - wb, emg-y))
  }

  // b: joint angles, per joint
  row-label((0.3, kin-y + 1.4), [b], [joint angles, per joint], color: muted)
  card((x1, kin-y), [two gloves], size: size, kind: "data", traces: (
    trace(angle-calibrated),
    trace(angle-raw-glove, color: orange),
  ))
  card((x2, kin-y), [anchors], size: size, kind: "stats", traces: (
    trace(angle-calibrated),
    trace(angle-raw-glove, color: orange),
  ), levels: (
    level(anchors(angle-raw-glove).at(1), color: orange, label: [$theta_98$]),
    level(anchors(angle-raw-glove).at(0), color: orange, label: [$theta_2$]),
    level(anchors(angle-calibrated).at(1), color: blue),
    level(anchors(angle-calibrated).at(0), color: blue),
  ))
  card((x3, kin-y), [$(theta - theta_2) slash (theta_98 - theta_2)$], size: op)
  card((x4, kin-y), [0 open · 1 closed], size: size, kind: "result", traces: (
    trace(rescale(angle-calibrated, ..anchors(angle-calibrated))),
    trace(rescale(angle-raw-glove, ..anchors(angle-raw-glove)), color: orange),
  ))
  for (a, b, wa, wb) in ((x1, x2, half, half), (x2, x3, half, op.at(0) / 2), (x3, x4, op.at(0) / 2, half)) {
    arrow((a + wa, kin-y), (b - wb, kin-y))
  }

  content((fw / 2, 0.3), anchor: "south", text(size: 6.5pt, fill: muted)[
    #box(std.line(length: 10pt, stroke: 1.2pt + blue)) dry monopolar band · calibrated angles #h(14pt)
    #box(std.line(length: 10pt, stroke: 1.2pt + orange)) bipolar electrodes · raw glove
  ])
})
