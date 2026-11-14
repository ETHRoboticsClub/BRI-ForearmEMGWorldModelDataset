# Hand mapping files

One JSON file per hand model: which canonical label (`src/femg/joints.py`) drives each joint of the model.
The explorer (`python -m tools.explorer`) lists every file here in its hand menu, and so does `scripts/view_hand.py --hand`.

```json
{
  "title": "Shadow Hand E3M5",
  "model": "shadow_hand",
  "entry": "right_hand",
  "landmarks": {"palmar_joint": "rh_FFJ3", "radial_body": "rh_ffknuckle", "ulnar_body": "rh_lfknuckle"},
  "joints": {
    "rh_FFJ4": {"label": "index_middle_abd", "at_0": 0.0, "at_1": -0.349, "moves": "ulnar", "range": [-0.35, 0.35]},
    "rh_FFJ1": {"label": "index_dip_flex", "at_0": 0.0, "at_1": 1.571, "if_missing": {"copy": "rh_FFJ2", "times": 0.67}, ...},
    "rh_WRJ1": {"label": "wrist_flex", "at_0": -0.35, "at_1": 0.489, "neutral": "mean", ...},
    "rh_MFJ4": {"label": null, "at_0": 0.0, ...}
  }
}
```

- `model`, `entry`: the mujoco-menagerie robot and entry point (downloaded once by mujoco-menagerie). Or instead
  `builder`: `"module:function"` returning an `MjSpec` or `MjModel`. E.g. `"femg.hand_models:myo_hand"` builds the
  MyoSuite hand from the `myo-sim` package (`pip install myo-sim`) and removes the arm, so only the hand is shown.
  Write such a function in `src/femg/hand_models.py` to adapt a model (remove parts, joints) without copying its files.
- `landmarks`: an index MCP flexion joint and two bodies across the palm (thumb side, little-finger side). They define
  the hand's directions (palmar, radial, distal) for the checks and the camera.
- `joints`: every joint of the model, once:
  - `label`: the canonical label that drives it, or `null`. A label drives at most one joint.
  - `at_0`, `at_1`: the joint angle (rad) when the label is 0 (open hand, p2 of the session) and 1 (closed, p98).
    If the joint closes with negative angles, `at_1` is negative: their order also sets the sign in rad mode.
    Without a label, `at_0` is the fixed angle.
  - `neutral` (optional): `"mean"` for joints that move both ways from a rest pose (wrist): the session's mean pose
    is shown at angle 0.
  - `if_missing` (optional): when a dataset does not measure the label. Absent: stay at `at_0`. A number: that angle.
    `{"copy": "rh_FFJ2", "times": 0.67}`: 0.67 times another joint of this hand, listed before it (DIP from PIP).
  - `moves`, `range`: reference only (where the joint's tip goes for positive angles, and its limits).

## The thumb

Measured on the instructed NinaPro movements (DB2: thumb up, opposition, fist; DB8: thumb abduction, grips), the
glove's two thumb-base sensors do not mean what their names suggest:

- `thumb_cmc_flex` (CyberGlove "thumb rotation") rises in fists, grips and opposition: the thumb comes in front of the
  palm. It drives the joint that lifts the thumb out of the palm plane (Shadow THJ4, LEAP th_cmc, Aero
  thumb_cmc_abd, MyoHand cmc_abduction).
- `thumb_cmc_abd` (CyberGlove "thumb abduction") is highest at rest and ~0 in all of those: the thumb crosses the palm
  when it is low. It drives the joint that swings the lifted thumb across the palm, reversed (`at_0` = across,
  `at_1` = 0), and sits halfway when a dataset does not measure it (Shadow THJ5, LEAP th_axl, Aero thumb_cmc_flex,
  MyoHand cmc_flexion).

Swinging across the palm only shows once the thumb is lifted: with the thumb flat, Shadow's THJ5 just twists it.

## The datasets

A hand file serves every dataset: the dataset readers (`src/femg/datasets`) deliver the canonical convention of
`src/femg/joints.py` (flexion, finger spread and radial deviation > 0; the thumb pair as above), so in 0-1 units
1 is always flexed / spread / radial. Sensors that run the other way are flipped in the reader, from the instructed
movements of each dataset:

| Dataset | Flipped in the reader | Evidence |
|---|---|---|
| NinaPro DB2, DB7 (raw CyberGlove) | 3 spreads, `wrist_dev` | E1: abduction of all fingers < adduction (6/6 reps, 3 sessions), radial < ulnar (5/6) |
| NinaPro DB8 (calibrated glove) | none | no spread or deviation movement to check: verify visually |
| SEEDS (CyberGlove III, 0-1) | none | thumbFlex, index_flex, middle_flex; spreads and wrist not verifiable |
| KIN-MUS UJI (anatomical angles) | thumb base: `thumb_cmc_flex` = CMC1_A, `thumb_cmc_abd` = −CMC1_F | paper Table 3 + grasping vs open hand (22/22 subjects) |
| GREAT (raw CyberGlove) | none | open > rest on the 3 spreads, power raises thumb roll and lowers thumb abduction (4/4 sessions) |
| NinaPro DB3, DB4 | no kinematics (cue only): the hand stays open | |

`python scripts/check_signs.py` repeats these tests on `data/processed` and fails when a sign is wrong. Run it after
changing a reader or adding a dataset (add its instructed movements to `TESTS`).

SEEDS records the left forearm: its angles are shown on the right-hand models (same joint angles, mirrored hand).

## Add a hand

1. Write the template (a mujoco-menagerie name or a builder `module:function`). It lists every joint with its limits
   and where it moves:

   ```bash
   python -m tools.hands template leap_hand --title "LEAP Hand" \
       --palmar-joint if_mcp --radial-body if_bs --ulnar-body rf_bs > src/femg/hands/leap_hand.json
   ```

2. Fill `label` (and adjust `at_0`/`at_1`) using `moves`: flexions close `palmar`, the index spread goes `radial`, the
   ring/pinky spreads go `ulnar`.

3. Check it:

   ```bash
   python -m tools.hands check leap_hand
   ```

   It tests that every label closes the hand in the anatomical direction (fingertip to the palm, spreads away from
   the middle finger, thumb flexion and opposition towards the little finger) and lists the labels the hand lacks.
