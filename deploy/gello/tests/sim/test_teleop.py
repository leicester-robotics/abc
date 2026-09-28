import numpy as np

from deploy.gello.sim.sim import BimanualSim
from deploy.gello.sim.teleop import JointTeleop


def test_selected_arm_joint_and_rate_limit():
    sim = BimanualSim()
    control = JointTeleop(sim)
    initial = control.command.copy()
    for key in "R2K":
        control.key(ord(key))
    assert np.isclose(control.target[8], initial[8] + 0.05)
    action = control.advance(0.02)
    assert np.isclose(action[8] - initial[8], 0.01)
    np.testing.assert_array_equal(action[:7], initial[:7])


def test_limits_grippers_and_stop():
    sim = BimanualSim()
    control = JointTeleop(sim)
    for _ in range(200):
        control.key(ord("K"))
        control.key(ord("U"))
    assert control.target[0] == sim.model.joint("left_joint1").range[1]
    assert control.target[6] == 0
    control.key(ord("X"))
    np.testing.assert_allclose(control.target, np.clip(sim.qpos_abc(), control.low, control.high))
    np.testing.assert_allclose(control.command, control.target)


def test_jog_moves_simulated_joint_with_physics():
    sim = BimanualSim()
    control = JointTeleop(sim)
    before = sim.arm_qpos("left")[0]
    for _ in range(4):
        control.key(ord("K"))
    for _ in range(60):
        sim.step(control.advance(1 / 30))
    assert sim.arm_qpos("left")[0] > before + 0.1
    assert np.isfinite(sim.data.qpos).all()


def test_shared_gello_home_matches_joint4_stop_and_allows_negative_jog():
    from deploy.gello.control.profiles import GELLO_REST_HOME

    sim = BimanualSim()
    home = np.r_[GELLO_REST_HOME, 1.0, GELLO_REST_HOME, 1.0]
    control = JointTeleop(sim, home=home)
    control.key(ord("H"))
    for side, offset, key in [("left", 0, "L"), ("right", 7, "R")]:
        assert control.target[offset + 3] == sim.model.joint(f"{side}_joint4").range[1]
        for pressed in (key, "4", "J"):
            control.key(ord(pressed))
        assert np.isclose(control.target[offset + 3], GELLO_REST_HOME[3] - 0.05)
    control.key(ord("H"))
    np.testing.assert_allclose(control.target, home)
