import argparse
import json

import numpy as np
import pytest

from deploy.gello.sim.gello import GelloController, mock_calibrations, run
from deploy.gello.sim.sim import BimanualSim


def samples(timestamp=1):
    return [
        (np.array([0.6, 0, 0, 0, 0, 0, 1]), timestamp),
        (np.array([-0.6, 0, 0, 0, 0, 0, -1]), timestamp),
    ]


def test_enable_dual_mapping_rate_and_stop():
    sim = BimanualSim()
    controller = GelloController(sim, mock_calibrations())
    initial = controller.motion.command.copy()
    command, _ = controller.advance(samples(), 1)
    np.testing.assert_array_equal(command, initial)
    command, target = controller.advance(samples(), 1, enable=True)
    np.testing.assert_allclose(target[[0, 7]], [0.6, -0.6])
    assert target[6] == 1 and target[13] == 0
    assert np.max(np.abs(command - initial)) <= 0.5 / 30 + 1e-10
    controller.hold("test")
    assert not controller.armed


@pytest.mark.parametrize("bad", ["stale", "future", "nan", "skew", "travel", "missing"])
def test_fault_latches_until_explicit_rearm(bad):
    controller = GelloController(BimanualSim(), mock_calibrations())
    controller.advance(samples(), 1, enable=True)
    current = samples()
    if bad == "stale":
        current = samples(0.5)
    elif bad == "future":
        current = samples(2)
    elif bad == "nan":
        current[0][0][0] = np.nan
    elif bad == "skew":
        current[0] = (current[0][0], 0.85)
    elif bad == "travel":
        current[0][0][0] = 1.2
    elif bad == "missing":
        current = current[:1]
    _, target = controller.advance(current, 1)
    assert target is None and not controller.armed
    controller.advance(samples(), 1)
    assert not controller.armed
    controller.advance(samples(), 1, enable=True)
    assert controller.armed


def test_distinct_port_and_side_required():
    c = mock_calibrations()
    c[1].port = c[0].port
    with pytest.raises(ValueError):
        GelloController(BimanualSim(), c)
    with pytest.raises(ValueError):
        GelloController(BimanualSim(), list(reversed(mock_calibrations())))


def test_physics_recording_and_no_overwrite(tmp_path):
    path = tmp_path / "episode.npz"
    args = argparse.Namespace(mock=True, headless=True, fast=True, record=str(path), seconds=2)
    run(args)
    with np.load(path, allow_pickle=False) as episode:
        assert episode["state"].shape == (60, 14)
        assert episode["armed"].all()
        assert np.isfinite(episode["next_state"]).all()
        assert np.ptp(episode["next_state"][:, 0]) > 0.2
        assert np.ptp(episode["next_state"][:, 7]) > 0.2
        np.testing.assert_allclose(episode["next_state"][:-1], episode["state"][1:])
        assert np.max(np.abs(np.diff(episode["action"], axis=0))) <= 0.5 / 30 + 1e-10
        assert json.loads(str(episode["metadata"]))["backend"] == "mujoco"
    with pytest.raises(FileExistsError):
        run(args)


def test_target_clips_to_follower_limits():
    c = mock_calibrations()
    c[0].home_target = [100] * 6
    sim = BimanualSim()
    controller = GelloController(sim, c)
    for _ in range(20):
        action, target = controller.advance(samples(), 1, enable=True)
        assert np.all(action <= controller.motion.high)
        assert np.all(action >= controller.motion.low)
    assert target[0] > controller.motion.high[0]
    np.testing.assert_allclose(controller.motion.target[:6], controller.motion.high[:6])


def test_safety_envelope_clamps_without_rescaling():
    from deploy.gello.control.joint_limits import JointEnvelope, fingerprint

    c = mock_calibrations()
    sim = BimanualSim()
    reference = GelloController(sim, c)
    indices = np.r_[0:6, 7:13]
    low, high = reference.motion.low[indices].copy(), reference.motion.high[indices].copy()
    low[[0, 6]], high[[0, 6]] = -0.2, 0.2
    envelope = JointEnvelope([fingerprint(x) for x in c], low.tolist(), high.tolist())
    controller = GelloController(sim, c, envelope=envelope)
    inside = samples()
    inside[0][0][0] = 0.1
    _, target = controller.advance(inside, 1, enable=True)
    assert target[0] == pytest.approx(0.1)
    assert controller.motion.target[0] == pytest.approx(0.1)
    for _ in range(30):
        action, target = controller.advance(samples(), 1, enable=True)
    assert target[0] == pytest.approx(0.6)
    assert action[0] == pytest.approx(0.2)
    assert action[7] == pytest.approx(-0.2)
    c[0].signs[0] *= -1
    with pytest.raises(ValueError, match="different calibration"):
        envelope.apply(c, reference.motion.low, reference.motion.high)
