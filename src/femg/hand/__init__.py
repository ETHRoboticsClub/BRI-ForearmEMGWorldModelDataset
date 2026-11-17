# femg.hand.joints needs only numpy; mapping, geometry, scene and builders need MuJoCo (pip install -e ".[viz]")
from femg.hand.joints import CANONICAL_JOINTS, JOINT_INDEX, N_JOINTS

__all__ = ["CANONICAL_JOINTS", "JOINT_INDEX", "N_JOINTS"]
