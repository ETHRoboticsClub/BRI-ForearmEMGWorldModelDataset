import mujoco

WRIST = "lunate_r"
FOREARM_ROTATION = "pro_sup_r"


def myo_hand() -> mujoco.MjSpec:
    """MyoSuite's right hand (myo-sim, Apache-2.0) from the wrist on: no arm bones, no forearm rotation.

    The arm bodies stay as invisible links, because the muscles' paths still run through them.
    """
    from myo_sim.build.compose import build_right_hand_from_arm_spec

    spec = build_right_hand_from_arm_spec()
    model = spec.compile()
    body = model.body(WRIST).id
    arm = []
    while body:
        body = model.body_parentid[body]
        arm.append(model.body(body).name)
    for name in arm:
        if name and name != "world":
            for geom in list(spec.body(name).geoms):
                if geom.group != 3:  # group 3: wrapping surfaces of the muscles, invisible
                    spec.delete(geom)
    spec.delete(spec.joint(FOREARM_ROTATION))
    return spec
