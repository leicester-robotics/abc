# Migration provenance

Migrated from the local Gontrol working tree at
`55b868ca2483ccf46d552d1a5522026305f73beb` (including its working-tree changes).
The source was left untouched.

Included: passive Dynamixel discovery/ID setup, encoder monitor/socket transport,
calibration and recovery, measured joint envelopes, pose capture, keyboard and
GELLO simulation controls, and their hardware-free regression tests.

The preview cell, scene composition and YAM model came from Gontrol's simulation,
originally imported from `leicester-robotics/yam` at `d23022d`. The cell derives
from `amazon-far/abc` (Apache-2.0); YAM meshes/model derive from
`google-deepmind/mujoco_menagerie/i2rt_yam` (MIT) with ABC tuning.
The upstream licenses are preserved beside the assets. ABC's existing
`MuJoCoYAMEnv` replaces Gontrol's standalone physics/state/action implementation.

Not migrated: unfinished gravity compensation/alignment models, gravity assistance
experiments, RL/IK/training code, old upstream robot checkout, runtime logs, and
calibration progress journals. The three finished station calibration/envelope
files are copied locally into the ignored `calibrations/` directory.
