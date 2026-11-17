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
#let active = percentile(envelope, 95)
#let channels = range(4).map(k => emg(200, gain: (1, 0.5, 0.8, 0.35).at(k), seed: 10 + k).map(v => v / 1.4 - 1.6 * k))

#cetz.canvas({
  import cetz.draw: *
  let (fw, fh) = (18.4, 6.6)
  frame(fw, fh)
  let size = (2.2, 1.55)
  let half = size.at(0) / 2
  let (x1, x2, x3) = (1.6, 4.1, 6.6) // per signal
  let (x4, x5, x6, x7) = (9.3, 11.8, 14.3, 16.8) // together
  let (emg-y, kin-y) = (4.85, 1.55)
  let mid-y = (emg-y + kin-y) / 2

  // a: the EMG
  row-label((0.3, fh - 0.3), [a], [EMG], color: blue)
  card((x1, emg-y), [raw, 2 kHz], size: size, traces: (trace(raw),))
  card((x2, emg-y), [500 Hz], size: size, traces: (trace(every(raw, 4)),))
  card((x3, emg-y), [high-pass, notch], size: size, traces: (trace(every(clean, 4)),))
  arrow((x1 + half, emg-y), (x2 - half, emg-y))
  arrow((x2 + half, emg-y), (x3 - half, emg-y))

  // b: the joint angles (and the instructed movement)
  row-label((0.3, kin-y + 1.15), [b], [joint angles], color: orange)
  card((x1, kin-y), [raw, glove], size: size, traces: (trace(glove-held, color: orange),))
  card((x2, kin-y), [100 Hz], size: size, traces: (trace(glove-100hz, color: orange),))
  arrow((x1 + half, kin-y), (x2 - half, kin-y))

  // c: together: aligned, then the two stored stages
  row-label((x4 - half, fh - 0.3), [c], [])
  card((x4, mid-y), [aligned], size: size, traces: (trace(envelope), trace(glove-100hz, color: orange)))
  card((x5, mid-y), [interim], size: size, kind: "result", traces: (
    trace(every(clean, 4).map(v => v / 2)),
    trace(samples(200).map(hand).map(y => 3 * y - 1.5), color: orange),
  ))
  card((x6, mid-y), [virtual array], size: size, traces: channels.map(trace))
  card((x7, mid-y), [processed], size: size, kind: "result", traces: (
    trace(envelope.map(v => v / active)),
    trace(glove-100hz, color: orange),
  ), levels: (level(1),))
  let join = x4 - half - 0.4 // both signals enter the aligned card from the left
  arrow((x3 + half, emg-y), (join, emg-y), (join, mid-y + 0.3), (x4 - half, mid-y + 0.3))
  arrow((x2 + half, kin-y), (join, kin-y), (join, mid-y - 0.3), (x4 - half, mid-y - 0.3))
  for (a, b) in ((x4, x5), (x5, x6), (x6, x7)) { arrow((a + half, mid-y), (b - half, mid-y)) }

  let below = mid-y - size.at(1) / 2 - 0.12
  for (x, body) in (
    (x4, [lag corrected]),
    (x5, [emg (V) · kin (rad)]),
    (x6, [÷ active level]),
    (x7, [virtual · labels 0-1]),
  ) {
    content((x, below), anchor: "north", text(size: 6.5pt, fill: muted, body))
  }
})
