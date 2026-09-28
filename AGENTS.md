# Agent instructions

- We use a YAM robot that is already set up and connected.
- Assume the robot starts in its base position with correct, safe
  support when connecting or starting the control stack.
- Before running any robot control command, show the exact command.
- Preserve the original forked code. Keep necessary code changes
  separate from the original implementation.
- Keep responses clear and concise.

## Station setup

- Use `ROBOT_PROFILE=local_yam_config` (defined in
  `deploy/robot/local_profiles.py`).
- CAN buses: `can_l_foll` = left arm, `can_r_foll` = right arm. Udev rules
  in `/etc/udev/rules.d/` name them by adapter serial and bring them up at
  1 Mbit/s automatically; `local_robot/99-can-up.rules` is the backup copy.
- If a bus is DOWN, run `sudo bash deploy/scripts/reset_all_can.sh`.