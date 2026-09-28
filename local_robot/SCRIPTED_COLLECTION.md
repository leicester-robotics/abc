# Scripted episodes and small DiT test

All code and outputs are local additions; upstream robot and trainer files are
unchanged. The collector never runs a learned policy.

## Status of this session

Artifacts: `local_robot/scripted_sessions/20260928T015734Z/`.

- Synthetic HDF5 → existing exporter → 8 training / 2 validation episodes passed.
- Small DiT trained for 200 steps on **synthetic data**, saved at 100 and 200,
  reloaded 200, and trained/saved step 201. Audits check all three image views,
  text embeddings, finite loss/gradients, and updates to head and DINO parameters.
- **No physical episodes collected.** The attempt stopped during sensor startup
  on motor communication faults, before any scripted motion.
- Operator-approved disable was sent, but disabled status was not acknowledged.
  The operator was told to switch off motor power.
- Hardware work stopped at the user's request because another person is using
  the robot. Subsequent changes have only been tested without hardware.

## Motion and failure behavior

Capture both six-joint starting poses before enable. Reuse the existing local
`lift_both.Session` to hold measured arm positions during gripper calibration.
Keep the calibrated gripper commands constant. Use the profile's six arm-joint
targets for the top pose.

Each episode records a one-second bottom hold, smooth ascent, one-second top
hold, smooth descent, and one-second final hold. Quintic time scaling gives
zero endpoint velocity and acceleration; commands stay on a straight joint-space
path, at most 0.25 rad/s and 0.5 rad/s². Commands are at least 1/30 second apart;
delays stretch motion without catch-up bursts. The final target is the saved
supported pose. Measured top and return errors must be at most 0.03 rad.

Camera SDK operations run in separate processes and become ready before motors
enable. This follows a camera-only diagnostic that measured a 377 ms pause in
another Python thread. Camera process isolation has not been verified on the
physical robot. Joint sampling continues in its own local thread; stale sensor
data, tracking error, queue overflow, and writer failures stop progression.

Every HDF5 file starts unusable. A completed cycle is decoded and checked before
being promoted to usable, including the first cycle before the next nine.
Normal completion stops sensor readers, verifies supported return, and uses the
local shutdown override to disable without a further move. Interruption/fault
holds for operator recovery; it does not attempt an unobserved automatic return.
Repeat interrupts cannot bypass recovery. Unconfirmed hardware hold/disable
requires the operator to support the arms and use the hardware power control.

## Commands for a future authorized hardware session

The current robot is shared: **do not run these while it is in use**. Use a fresh
session path. Show the exact command before any robot control execution.

The robot SDK comes from the existing Joint Studio Python 3.11 environment;
camera/recording packages are isolated in `local_robot/hardware_deps`:

```bash
uv pip install --python /home/tarik/code/yam/control/single_arm/.venv/bin/python \
  --target local_robot/hardware_deps --no-deps \
  h5py opencv-python pyrealsense2 pyzmq imageio-ffmpeg

ROBOT_PROFILE=local_yam_config \
PYTHONPATH=/home/tarik/code/abc/local_robot/hardware_deps:/home/tarik/code/abc \
/home/tarik/code/yam/control/single_arm/.venv/bin/python -u \
  -m local_robot.scripted_collection --session local_robot/scripted_sessions/NEW_SESSION
```

The normal VLA launcher and shutdown behavior are unchanged.

## Export and training (no hardware access)

After ten physical recordings pass validation:

```bash
.venv/bin/python -m local_robot.scripted_export --session local_robot/scripted_sessions/NEW_SESSION
.venv/bin/python -u -m local_robot.scripted_train --session local_robot/scripted_sessions/NEW_SESSION
.venv/bin/python -u -m local_robot.scripted_train --session local_robot/scripted_sessions/NEW_SESSION --resume
```

Export uses three stacked views at 30 fps, absolute 14-dimensional states/actions,
seed 123 and an 8/2 episode split. Normalization is computed from training episodes
only. Review video is `export/all_episodes.mp4`.

DiT: hidden size 256, depth 4, heads 8, chunk 30, batch 1, no compilation.
The full DINO vision architecture starts randomly; CLIP text uses the existing
normal path. Training logs every 10 steps, validates/saves every 100. JSON audit
files record every optimizer step. The resume command strictly reloads model,
optimizer and scheduler through the existing trainer and saves step 201.

To exercise the entire export/training path without hardware, choose a separate
session directory and pass `--synthetic` to `scripted_export` first. Synthetic
frames and states are only software fixtures. Repeated physical episodes and
their validation split also do not establish policy generalization.

## Hardware-free verification

```bash
PYTHONPATH=/home/tarik/code/abc/local_robot/hardware_deps:/home/tarik/code/abc \
/home/tarik/code/yam/control/single_arm/.venv/bin/python -m pytest local_robot/tests -q
.venv/bin/python -m unittest local_robot.tests.test_scripted_train -q
```

The hardware environment skips the torch-only unit test; the second command
runs it in the training environment. No test accesses motors or cameras.
