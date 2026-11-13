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
  let (x1, x2, x3, x4) = (0, 4.8, 9.6, 14.4)
  let (emg-y, kin-y) = (0, -3.8)

  title((-1.9, 2.45), [Normalization: two anchors per session, no labels needed])

  panel((-1.9, -1.55), (16.3, 1.9), [EMG, per channel])
  node((x1, emg-y), "data", [emg.npy], detail: [two devices, 5× apart (V)], traces: (
    trace(raw-dry),
    trace(raw-bipolar, color: orange),
  ))
  node((x2, emg-y), "stats", [Envelope + anchors], detail: [100 ms RMS · p10, p95 (log)], log: true, traces: (
    trace(env-dry),
    trace(env-bipolar, color: orange),
  ), levels: (
    level(active-dry, color: blue, label: [p95]),
    level(percentile(env-dry, 10), color: blue, label: [p10]),
    level(active-bipolar, color: orange, label: [p95]),
    level(percentile(env-bipolar, 10), color: orange, label: [p10]),
  ))
  node((x3, emg-y), "step", [normalize_emg], detail: [envelope ÷ active], size: (3.4, 1.2))
  node((x4, emg-y), "result", [Model input], detail: [same scale for every device], traces: (
    trace(rescale(env-dry, 0, active-dry)),
    trace(rescale(env-bipolar, 0, active-bipolar), color: orange),
  ), levels: (level(1, label: [1]),))

  arrow((x1 + 1.7, emg-y), (x2 - 1.7, emg-y), label: [build time])
  arrow((x2 + 1.7, emg-y), (x3 - 1.7, emg-y), label: [meta.json])
  arrow((x3 + 1.7, emg-y), (x4 - 1.7, emg-y), label: [load time])

  panel((-1.9, kin-y - 1.55), (16.3, kin-y + 1.9), [Kinematics, per joint])
  node((x1, kin-y), "data", [kin.npy], detail: [calibrated (rad) vs raw glove], traces: (
    trace(angle-calibrated),
    trace(angle-raw-glove, color: orange),
  ))
  node((x2, kin-y), "stats", [Anchors], detail: [p2 = open · p98 = closed], traces: (
    trace(angle-calibrated),
    trace(angle-raw-glove, color: orange),
  ), levels: (
    level(anchors(angle-raw-glove).at(1), color: orange, label: [p98]),
    level(anchors(angle-raw-glove).at(0), color: orange, label: [p2]),
    level(anchors(angle-calibrated).at(1), color: blue, label: [p98]),
    level(anchors(angle-calibrated).at(0), color: blue, label: [p2]),
  ))
  node((x3, kin-y), "step", [normalize_kin], detail: [(θ − low) ÷ (high − low)], size: (3.4, 1.2))
  node((x4, kin-y), "result", [Target], detail: [0 open … 1 closed], traces: (
    trace(rescale(angle-calibrated, ..anchors(angle-calibrated))),
    trace(rescale(angle-raw-glove, ..anchors(angle-raw-glove)), color: orange),
  ))

  arrow((x1 + 1.7, kin-y), (x2 - 1.7, kin-y), label: [build time])
  arrow((x2 + 1.7, kin-y), (x3 - 1.7, kin-y), label: [meta.json])
  arrow((x3 + 1.7, kin-y), (x4 - 1.7, kin-y), label: [load time])

  content((x1 - 1.7, kin-y - 2.0), anchor: "west", text(size: 6.5pt, fill: muted)[
    #box(std.line(length: 10pt, stroke: 1.2pt + blue)) dry monopolar armband (EMG-FK) #h(10pt)
    #box(std.line(length: 10pt, stroke: 1.2pt + orange)) bipolar electrodes, raw glove (NinaPro) #h(10pt)
    #box(std.line(length: 10pt, stroke: (dash: "dashed"))) anchor percentile
  ])
})
