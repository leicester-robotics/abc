# Local robot

YAM station tools and local deployment overrides. Run commands from the repository root.

## Setup

- Arms: `can_l_foll` (left), `can_r_foll` (right), 1 Mbit/s.
- Deployment profile: `ROBOT_PROFILE=local_yam_config`.
- Lift environment: `../yam/control/single_arm`, selected by the launcher.
- Other Python tools require their SDK, camera, or policy dependencies in the active environment.

## Lift both arms

```bash
./local_robot/lift_both.sh --check    # Disabled feedback + path check
./local_robot/lift_both.sh           # Move both arms
```

Default: **5 cm up, 1 second pause, return; 5 seconds each way**.
Options: `--execute --height 0.05 --seconds 5`.

Grippers calibrate first. Arms hold afterward; support them, then type `disable`.
Ctrl+C cancels movement and attempts to hold.

The return target is the pose recorded before motor enable. Exact physical return
remains unresolved; acceptance allows 1 cm position and 0.03 rad joint error.
Grippers retain their calibrated opening.

Checks cover joint limits, self-collision, tracking, feedback age, and timing.
Bases must be upright. Desk objects and collisions between arms are not modelled.

## Files

| File | Purpose |
|---|---|
| [lift_both.sh](lift_both.sh) | Select environment, print command, launch lift. |
| [lift_both.py](lift_both.py) | Plan motion, calibrate, control both arms, hold, and disable. |
| [read_joints.py](read_joints.py) | Read joint positions for supplied CAN channels; briefly enables motors. |
| [snap_cameras.py](snap_cameras.py) | Save RGB frames from all RealSense cameras; `--out`, `--tag`. |
| [gripper_identify.py](gripper_identify.py) | Close/open one gripper and capture images; arm joints receive zero torque and need support. |
| [99-can-up.rules](99-can-up.rules) | Bring named follower CAN interfaces up at 1 Mbit/s; requires existing naming rules. |
| [serve_local.py](serve_local.py) | Serve ABC-DiT (bf16) or ABC-VLA (Gemma layer streaming, optional int8); change prompts through terminal input. |
| [vla_offload.py](vla_offload.py) | VLA inference that fits the 8 GB GPU: resident plus host-streamed Gemma layers. |
| [bench_serve_local.py](bench_serve_local.py) | Startup, latency and VRAM of a `serve_local` policy on recorded frames (no robot). |
| [serve_bf16.py](serve_bf16.py) | Original DiT-only bf16 server; still works as `--server-module`. |
| [deploy_policy.py](deploy_policy.py) | Deployment entry point using the local rollout; upstream CLI arguments. |
| [policy_rollout.py](policy_rollout.py) | Wait for camera frames before the first reset; confirm the first action with ENTER via the key listener (`--no-record`); optional speed-limited reset (`LOCAL_RESET_SPEED`). |
| [run_policy.py](run_policy.py) | Preflight, background policy server (or reuse), foreground deployment, cleanup. |
| [set_prompt.py](set_prompt.py) | Send a new prompt to the server started by `run_policy`. |
| [tests/test_lift_both.py](tests/test_lift_both.py) | Offline lift and safety regression tests. |

Camera snapshots default to `local_robot/snaps/`; gripper captures use a channel subfolder.

## Run the policy (one command)

```bash
# DiT (cache/bottles_75k.pt)
uv run python -m local_robot.run_policy --check --debug                         # preflight, print commands
uv run python -m local_robot.run_policy --prompt='pick up the cube' --debug     # arms not driven
uv run python -m local_robot.run_policy --prompt='pick up the cube'             # live: arms move
# VLA (cache/vla_abc130k_200000_v2.pt)
uv run python -m local_robot.run_policy --policy=vla --check --debug
uv run python -m local_robot.run_policy --policy=vla --prompt='pick up the cube' --debug
uv run python -m local_robot.run_policy --policy=vla --prompt='pick up the cube'
# Second terminal: new prompt from the next chunk
uv run python -m local_robot.set_prompt "put the cube in the bowl"
```

Safety:

- In the live run, press `b` (stop and go home) before `c`: `c`/`j` shut down by moving the arms to zero in 2 s.
- The first reset after start is a gentle move (`--reset-speed`, default 0.25 rad/s). Confirm the first-action safety table with ENTER only if the deltas look small.
- Keep the GPU exclusive. Other GPU processes slow inference, and then the robot pauses between chunks (`inference is still pending`).

Keys: `a` start, `b` stop and home, `c`/`j` shut down, ENTER confirms the safety table, Ctrl-C aborts.

1. Preflight is read-only. It checks for other robot sessions, CAN buses UP (a warning under `--debug`), RealSense devices not held by another process (for example `yam-harness serve`) and on USB3, the checkpoint, the port and free VRAM. `--check` stops here and prints both commands.
2. The policy kind comes from the checkpoint (`--policy dit|vla` picks the cache checkpoint; `--checkpoint-path` overrides it). The launcher starts `local_robot.serve_local` with the checkpoint, prompt, diffusion steps and RTC prefix taken from its own flags, and gives the robot side the same checkpoint and flags. `--server-args` may not repeat these. If a policy server already answers on `--port`, the launcher reuses it and never stops it. Server logs are in `data/logs/server-*.log`.
3. The launcher runs `local_robot.deploy_policy` in the same terminal and stops the server it started, and any orphaned deploy nodes, on exit.

| Policy | Server | ms/chunk | GPU | Lead, chunk |
|---|---|---|---|---|
| DiT | bf16 | 106-113 | 4.1 GiB | 7, 16 |
| DiT `--fast-inference` | bf16, compiled | 94 | 4.1 GiB | 7, 16 |
| VLA, GPU free (7.2 GiB or more free) | bf16, 30/34 Gemma layers resident | 116 | 6.7 GiB | 7, 16 |
| VLA, GPU shared | `--offload.quantize=all` (int8), chosen automatically | 160-190 | 2.4-4.3 GiB | 10, 20 |

Other defaults: `--diffusion-steps=5` (applied by the server), `--rtc-prefix-length=4`, `--no-record`. The RTC prefix plus `--execute-chunk-dim` must not exceed the model chunk of 30. An explicit `--rtc-inference-lead-steps`, `--execute-chunk-dim` or `--server-args='--offload.quantize=...'` overrides the choice. Use `--compress-images` for JPEG observations and `--no-rtc` for blocking chunks. Any other flag goes to `deploy_policy` (`--init-q`, `--record`, `--verbose`).

`--fast-inference` (DiT only) compiles both samplers at server startup. Startup takes 345 s on a cold inductor cache and 50-65 s once it is cached, versus 9 s without compiling. The launcher waits up to 1800 s. The ~13 ms gain is not needed: the uncompiled round trip already fits the RTC window with about 100 ms to spare. It is off by default.

`set_prompt` works only with a server the launcher started. The launcher relays each prompt to the server's stdin through `/tmp/abc-$USER/prompt-<port>.sock`. For a reused server, type the prompt in the server's own terminal. Recordings keep the launch `--prompt`.

Upstream `reset()` sends `init_q` as one follower `interp`: a linear 2 s ramp from wherever the arm is. With `--reset-speed` (default 0.25 rad/s), the rollout splits that move into several 2 s ramps through waypoints. For example, the base pose to `init_q` becomes 2 ramps of 0.5 rad each. `--reset-speed=0` restores upstream behavior. Running `deploy_policy` directly keeps upstream behavior unless `LOCAL_RESET_SPEED` is set.

## Policy (manual)

```bash
uv run python -m local_robot.serve_local --policy.checkpoint-path=cache/bottles_75k.pt --policy.diffusion-steps=5 --policy.rtc-prefix-length=4
uv run python -m local_robot.serve_local --policy.checkpoint-path=cache/vla_abc130k_200000_v2.pt --policy.diffusion-steps=5 --policy.rtc-prefix-length=4
ROBOT_PROFILE=local_yam_config uv run python -m local_robot.deploy_policy --help
```

Start the server in a separate terminal. Connect deployment with `--remote-host=localhost`. Typed server prompts apply to the next action chunk. `local_robot.serve_bf16` (DiT only) still works the same way.

## Tests

```bash
uv run --no-sync --project ../yam/control/single_arm python -m pytest local_robot/tests -q
```
