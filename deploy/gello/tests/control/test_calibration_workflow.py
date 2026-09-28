"""Neutral returns keep previous-joint release motion out of direction checks."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from deploy.gello.control import gello_cli
from deploy.gello.control.calibration import Calibration


def test_return_to_neutral_retries_then_returns_measured_baseline():
    home = np.ones(7)
    moved = home.copy()
    moved[1] += 0.3
    settled = home + 0.01
    capture = Mock()
    capture.until_enter.side_effect = [moved, settled]
    pose = gello_cli.return_to_neutral(capture, home, "Reset")
    np.testing.assert_array_equal(pose, settled)
    assert capture.until_enter.call_count == 2
    capture.checkpoint.assert_called_once_with("neutral", "accepted")


@pytest.mark.parametrize("negative_joints", [[], [4]])
def test_each_direction_and_sweep_returns_to_neutral_before_next_joint(
    tmp_path, monkeypatch, negative_joints
):
    prompts = []
    poses = [np.zeros(7)]
    for i in range(6):
        direction = np.zeros(7)
        direction[i] = 0.2 * (-1 if i == 1 else 1)
        poses.append(direction)
        if i < 5:
            poses.append(np.zeros(7))
    samples = iter(poses)

    def capture(self, prompt):
        prompts.append(prompt)
        pose = next(samples)
        self.low = np.minimum(self.low, pose)
        self.high = np.maximum(self.high, pose)
        self.latest = (pose.copy(), pose.copy())
        return pose.copy()

    monkeypatch.setattr(gello_cli.Capture, "until_enter", capture)

    def limits(self, indices, prompt):
        prompts.append(prompt)
        self.low = np.minimum(self.low, np.array([0] * 6 + [-0.3]))
        self.high = np.maximum(self.high, np.array([0.4] * 6 + [0]))
        self.latest = (np.zeros(7), np.zeros(7))

    monkeypatch.setattr(gello_cli.Capture, "capture_limits", limits)
    output = tmp_path / "left.json"
    args = SimpleNamespace(
        side="left",
        port="test",
        baudrate=2000000,
        ids=list(range(1, 8)),
        home=[0] * 6,
        output=output,
        negative_joints=negative_joints,
    )
    bus = SimpleNamespace(port_name="test", baudrate=2000000)
    gello_cli.capture_calibration(args, bus, [1200] * 6 + [1190])
    result = Calibration.load(output)
    assert result.signs == [1, -1, 1, -1 if 4 in negative_joints else 1, 1, 1]
    if 4 in negative_joints:
        assert "then J" in prompts[7]
        assert "NEGATIVE" in prompts[7]
        raw = np.zeros(7)
        raw[3] = 0.2
        assert result.map(raw)[3] == pytest.approx(-0.2)
    assert len(prompts) == 13
    for index in range(2, 12, 2):
        assert "RETURN TO NEUTRAL" in prompts[index]
    assert "TRIGGER ENDPOINTS" in prompts[12]
    assert result.version == 2
    assert result.raw_min is None and result.raw_max is None
    assert result.gripper_closed == -0.3
    assert result.gripper_open == 0
