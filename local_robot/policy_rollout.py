"""Upstream policy rollout with two local fixes.

1. The first reset waits for every camera to boot. Camera nodes are spawned
   alongside the rollout and need ~1.3 s (spawn, pyrealsense2 import, pipeline
   start) before their first frame. With a policy server that is already
   running, the rollout reaches ``env.reset()`` sooner than that, and the
   500 ms camera read in ``YAMEnv.get_obs`` times out.

2. With direct episode keys (``--no-record``), the first-action confirmation
   is read from the key listener's key stream instead of ``input()``. The key
   listener shares the terminal, puts it in cbreak mode (no echo, so ENTER
   shows nothing) and polls it with ``select`` + ``read(1)``, racing
   ``input()`` for the ENTER keystroke; a keystroke it wins is published as
   ``"\\n"`` and ignored, leaving ``input()`` blocked.

3. Optional gentle reset (``LOCAL_RESET_SPEED`` rad/s, set by
   local_robot.run_policy; unset or 0 keeps upstream). Upstream sends init_q
   as one follower ``interp`` command: a linear 2 s ramp from wherever the arm
   is, however far. This splits the move into several such ramps through
   intermediate waypoints so no joint exceeds the speed limit.
"""

import math
import os
import time

import numpy as np

import deploy.robot.communication as comms
from deploy.robot.gym import policy_rollout
from deploy.robot.gym.policy_safety import prompt_first_action_safety_check
from deploy.robot.gym.yam_env import YAMEnv

CAMERA_BOOTUP_TIMEOUT_MS = 30_000
CONFIRM_KEYS = frozenset({"\n", "\r"})
RESET_SPEED_ENV = "LOCAL_RESET_SPEED"
FOLLOWER_INTERP_S = 2.0  # YAMFollowerNode.process_command

_upstream_reset = YAMEnv.reset
_upstream_get_state_obs = YAMEnv._get_state_obs
_episode_controller = None


def _get_state_obs_tracked(self):
    state = _upstream_get_state_obs(self)
    self._last_state = np.asarray(state, dtype=np.float64)
    return state


def _wait_for_followers(self) -> None:
    """Upstream's first-reset follower wait, keeping the boot-time joint state."""
    states = [
        comms.subscribe(
            socket, timeout_ms=60_000, topic_label=f"follower_state_bootup:{name}"
        )[0]
        for name, socket in self.robot_state_sockets.items()
    ]
    self._followers_ready = True
    self._last_state = np.concatenate(states).astype(np.float64)


def _approach_init_q(self, speed: float) -> None:
    """Step toward init_q through waypoints; upstream reset does the last ramp."""
    if not hasattr(self, "_followers_ready"):
        _wait_for_followers(self)
    start = getattr(self, "_last_state", None)
    target = self.get_init_q().astype(np.float64)
    if start is None or start.shape != target.shape:
        return
    segments = math.ceil(np.abs(target - start).max() / (speed * FOLLOWER_INTERP_S))
    if segments <= 1:
        return
    print(
        f"[local_robot] Moving to init_q in {segments} ramps of "
        f"{FOLLOWER_INTERP_S:.0f} s (<= {speed} rad/s)...",
        flush=True,
    )
    for index in range(1, segments):
        self.move_to(start + (target - start) * index / segments)


def _reset_after_camera_bootup(self, **kwargs):
    if not getattr(self, "_cameras_ready", False):
        if os.environ.get("DEPLOY_VERBOSE"):
            print("[local_robot] Waiting for camera nodes to publish...")
        for name, socket in self.camera_sockets.items():
            comms.subscribe(
                socket,
                timeout_ms=CAMERA_BOOTUP_TIMEOUT_MS,
                topic_label=f"camera_bootup:{name}",
            )
        self._cameras_ready = True
        if os.environ.get("DEPLOY_VERBOSE"):
            print("[local_robot] All cameras publishing.")
    speed = float(os.environ.get(RESET_SPEED_ENV) or 0)
    if speed > 0 and self.execute_actions:
        _approach_init_q(self, speed)
    return _upstream_reset(self, **kwargs)


class _KeyboardEpisodeController(policy_rollout.KeyboardEpisodeController):
    def __init__(self, context):
        global _episode_controller
        super().__init__(context)
        _episode_controller = self

    def close(self):
        global _episode_controller
        _episode_controller = None
        super().close()


def _pending_keys(controller):
    while controller._sub.poll(timeout=0):
        _, extras = comms.subscribe(controller._sub)
        if extras.get("pressed", True):
            yield str(extras.get("key", ""))


def _confirm_via_key_listener(controller) -> None:
    """Block until ENTER arrives from the key listener; c/j abort the session."""
    # Keys typed before the table was shown must not count as confirmation.
    for key in _pending_keys(controller):
        if controller.command_for_key(key) == "shutdown":
            raise KeyboardInterrupt
    print(
        "Press ENTER to execute, c/j to shut down, or Ctrl-C to abort > ",
        end="",
        flush=True,
    )
    while True:
        for key in _pending_keys(controller):
            if key in CONFIRM_KEYS:
                print(
                    "\nConfirmed. Executing; press a/b to stop and go home, "
                    "c/j to shut down.",
                    flush=True,
                )
                return
            if controller.command_for_key(key) == "shutdown":
                print("\nAborted before execution; shutting down.", flush=True)
                raise KeyboardInterrupt
        time.sleep(0.01)


def _first_action_safety_check(current_state, action, **kwargs):
    controller = _episode_controller
    if controller is None:
        return prompt_first_action_safety_check(current_state, action, **kwargs)
    prompt_first_action_safety_check(
        current_state, action, **{**kwargs, "enter_prompt": None}
    )
    _confirm_via_key_listener(controller)


YAMEnv.reset = _reset_after_camera_bootup
YAMEnv._get_state_obs = _get_state_obs_tracked
policy_rollout.KeyboardEpisodeController = _KeyboardEpisodeController
policy_rollout.prompt_first_action_safety_check = _first_action_safety_check


def run(args) -> None:
    policy_rollout.run(args)
