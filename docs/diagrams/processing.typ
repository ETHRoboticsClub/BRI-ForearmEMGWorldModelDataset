#import "style.typ": *
#show: setup

#let n = 800
#let clean = emg(n, seed: 3)
#let artifacts = samples(n).map(t => 0.8 * calc.sin(2 * calc.pi * 0.9 * t) + 0.25 * calc.sin(2 * calc.pi * 90 * t))
#let raw = clean.zip(artifacts).map(((x, a)) => x + a)
#let every(xs, k) = xs.enumerate().filter(((i, _)) => calc.rem(i, k) == 0).map(((_, x)) => x)

#let hand(t) = activation(t - 0.04)
#let glove-held = samples(n).map(t => hand(calc.floor(t * 30) / 30))
#let glove-100hz = samples(100).map(hand)
#let envelope = moving-rms(every(clean, 4), 8)

#cetz.canvas({
  import cetz.draw: *
  let size = (3.2, 2.4)
  let (x1, x2, x3, x4, x5) = (0, 4.2, 8.4, 12.6, 16.8)
  let (emg-y, mid-y, kin-y) = (0, -1.9, -3.8)
  let half = size.at(0) / 2

  title((-1.8, 2.45), [Processing: one session, from raw files to the common format])

  panel((-1.8, -1.55), (10.2, 1.9), [EMG])
  node((x1, emg-y), "data", [load_session], detail: [2 kHz · volts · drift, mains], size: size, traces: (
    trace(raw),
  ))
  node((x2, emg-y), "step", [resample_emg], detail: [→ 500 Hz, anti-aliased], size: size, traces: (
    trace(every(raw, 4)),
  ))
  node((x3, emg-y), "step", [filter_emg], detail: [causal 10 Hz high-pass + notch], size: size, traces: (
    trace(every(clean, 4)),
  ))
  arrow((x1 + half, emg-y), (x2 - half, emg-y))
  arrow((x2 + half, emg-y), (x3 - half, emg-y))

  panel((-1.8, kin-y - 1.55), (10.2, kin-y + 1.9), [Kinematics + stimulus])
  node((x1, kin-y), "data", [load_session], detail: [glove · native rate, held samples], size: size, traces: (
    trace(glove-held, color: orange),
  ))
  node((x2, kin-y), "step", [resample_labels], detail: [→ 100 Hz · NaN if unmeasured], size: size, traces: (
    trace(glove-100hz, color: orange),
  ))
  arrow((x1 + half, kin-y), (x2 - half, kin-y))

  node((x4, mid-y), "step", [align_labels], detail: [shift lag · label k ↔ EMG 5k], size: size, traces: (
    trace(envelope),
    trace(every(glove-100hz, 1), color: orange),
  ))
  node((x5, mid-y), "result", [save_session], detail: [add_stats · atomic write], size: size, traces: (
    trace(every(clean, 4)),
    trace(samples(200).map(hand).map(y => 3 * y - 1.5), color: orange),
  ))
  arrow((x3 + half, emg-y), (x4, emg-y), (x4, mid-y + 1.2))
  arrow((x2 + half, kin-y), (x4, kin-y), (x4, mid-y - 1.2))
  arrow((x4 + half, mid-y), (x5 - half, mid-y))

  content((x5, mid-y - 1.55), anchor: "north", text(size: 6.5pt, fill: muted)[
    emg.npy (T, C) · kin.npy (T/5, 22)\ stimulus.npy · meta.json
  ])
  content((x1 - half, kin-y - 2.0), anchor: "west", text(size: 6.5pt, fill: muted)[
    #box(std.line(length: 10pt, stroke: 1.2pt + blue)) EMG #h(10pt)
    #box(std.line(length: 10pt, stroke: 1.2pt + orange)) joint angle
  ])
})
