"""Shared simulator targets for this station's supported GELLO resting pose."""

# YAM joint 4's MJCF upper limit is +1.5708 rad. The leader rests at its
# corresponding upper stop, so calibration tests that joint in the negative direction.
GELLO_REST_HOME = (0.0, 0.0, 0.0, 1.5708, 0.0, 0.0)
GELLO_REST_NEGATIVE_JOINTS = (4,)
