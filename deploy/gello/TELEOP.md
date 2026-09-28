# Dual GELLO simulation teleoperation

Run all commands from the ABC repository root. This migrates the working passive
leader workflow from Gontrol. Gravity compensation is unfinished and remains in
Gontrol; this path never enables leader torque or opens physical follower buses.

## Current station quickstart

The measured `left.relative.json`, `right.json`, and `joint-envelope.json` were
copied into `deploy/gello/calibrations/` and are intended to be committed with the
repository. They are specific to this station. The left USB adapter
is FTBEO6Y3, the right FTBEO6Y6; both use 2 Mbps and IDs 1–7. After a power cycle,
mode change, or torque enable resets the encoder counters, recapture calibration
before using these saved references. Begin near the calibrated resting pose.

Install the lightweight simulation dependencies (no training stack required):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r deploy/gello/requirements.txt
source .venv/bin/activate
mkdir -p deploy/gello/calibrations
```

Hardware-free smoke test:

```sh
python -m deploy.gello.sim.gello --mock --headless --fast --seconds 3
```

If the Gontrol encoder monitor is already running, it remains compatible; use
`--socket` below without starting a second monitor. Otherwise start this in one
terminal:

```sh
python -m deploy.gello.control.monitor \
  --left-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6Y3-if00-port0 \
  --left-baud 2000000 \
  --right-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6Y6-if00-port0 \
  --right-baud 2000000 --output deploy/gello/calibrations/live-encoders.json
```

Then open the simulation in a second terminal:

```sh
python -m deploy.gello.sim.gello --socket \
  --left deploy/gello/calibrations/left.relative.json \
  --right deploy/gello/calibrations/right.json \
  --limits deploy/gello/calibrations/joint-envelope.json
```

Space enables tracking, X holds, Q quits. To connect directly to the USB leaders,
stop the monitor and omit `--socket`. A graphical desktop is required for hardware
leader teleoperation. The simulator requires passive leaders with torque off.

## Reuse of existing ABC code

- `abc_sim.env.MuJoCoYAMEnv` supplies indexing, state projection, action application,
  and physics stepping. The small adapter in `sim/sim.py` loads the original preview
  scene and runs at 30 Hz with 16 substeps. Gripper commands now use ABC's existing
  0.0475 m normalization (the old preview used 0.0495 m actuator travel).
- The original task-free cell and YAM meshes are bundled so this preview does not
  need the larger downloadable ABC task assets. See [PROVENANCE.md](PROVENANCE.md).
- `deploy/robot` already supplies the ZMQ launcher, YAM followers, camera capture,
  HDF5 collection and recording export. Reuse those for the real-robot phase.
- Its current `GelloLeaderNode` assumes 4 Mbps, startup offset guessing, fixed
  trigger limits and force feedback on by default. The migrated measured calibration
  and read-only monitor are needed for this station; changing robot-profile IDs
  alone does not integrate them. The real-robot launcher is not wired to this path yet.

The retained controller maps six joints 1:1, normalizes triggers, clips to the
model and optional measured envelope, and limits changes to 0.5 units/second.
Stale or unsynchronized samples latch a hold until explicitly re-armed.

## Validation

```sh
python -m pip install pytest
python -m pytest deploy/gello/tests -q
```

The tests cover calibration/recovery, monitor failures, local socket transport,
fault/re-arm behavior, bounds, physics motion and recording round-trip. These are
hardware-free checks; live leader tracking still needs a station test.

## Install and discover

From the repository root, with Python 3.10+ installed:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r deploy/gello/requirements.txt
source .venv/bin/activate
mkdir -p deploy/gello/calibrations
python -m deploy.gello.control.gello_cli scan
```

Use persistent `/dev/serial/by-id/...` paths. If permission is denied, grant
access locally (after each reconnect), e.g.:

```sh
sudo setfacl -m u:$USER:rw /dev/ttyUSB0 /dev/ttyUSB1
```

The USB adapter and motors may have separate power supplies. Empty discovery
means **no responses**, not proof that the motor chain is absent or configured.
Default discovery tries 57600, 1000000, 2000000 and 115200 baud; override with
`--baudrates 9600 2000000 3000000 4000000` if needed. Discovery only sends pings;
it changes neither IDs nor motor baud rates. Unknown models are reported but
refused for calibration/ID writes until their control tables are verified.

The [upstream YAM passive setup](https://github.com/wuphilipp/gello_software/blob/main/configs/yam_passive.yaml)
uses the same serial leader / CAN follower split. Its
[configuration generator](https://github.com/wuphilipp/gello_software/blob/main/scripts/generate_yam_config.py)
uses 57600 baud and nominal YAM joint signs `[1, -1, -1, -1, 1, 1]`.
This routine measures signs instead of assuming your build matches that orientation.

Supported models: XL330-M077/M288, XC330-M181/M288/T181/T288 and
XM430-W210/W350, using the shared X-series control table. References:
[ROBOTIS XC330-T288](https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/),
[ROBOTIS XL330-M288](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/),
[Dynamixel SDK](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/).

## Continuous encoder debugging

`python -m deploy.gello.control.monitor` uses independent serial workers targeting
100 Hz fresh group reads per arm, prints at 100 Hz, and atomically replaces the
latest-status JSON at 5 Hz. Output is raw encoder degrees in the format
`Right: [J1,J2,J3,J4,J5,J6,J7] Left: [J1,J2,J3,J4,J5,J6,J7]`.
Use `--read-hz`, `--print-hz`, and `--json-hz` to change the rates. An incomplete
or errored group read invalidates that arm; stale data older than 200 ms prints
`NA`. Faster terminal refreshes may repeat the latest sample: JSON sequence,
age, measured read rate, and timing statistics distinguish those repeats.

These are best-effort host rates, not hard real-time guarantees. JSON records
maximum observed interval jitter and operation duration. Terminal timings measure
write/flush completion, not visible screen refresh; file replacement is not an
fsync durability guarantee. USB buffering and baud rate can limit fresh reads.
The monitor sends no motor-setting writes. Run it in shared tmux, and stop with
Ctrl-C before direct USB calibration or another USB reader uses these adapters.

```sh
.venv/bin/python -m deploy.gello.control.monitor \
  --left-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6Y3-if00-port0 \
  --left-baud 2000000 \
  --right-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6Y6-if00-port0 \
  --right-baud 2000000 --output deploy/gello/calibrations/live-encoders.json
```

Check timestamps before using the file: an abrupt process termination can leave
its last snapshot behind. The monitor marks `running: false` on normal exit.

The current station uses 2 Mbps on both chains, return delay register 9 set to
zero on all fourteen encoders, and USB latency timers set to 1 ms. USB timer
settings may reset when adapters reconnect. The original right-chain baud rate
was 57,600. Configuration changes were verified with torque off and logged under
`deploy/gello/calibrations/`. Even with these settings, measured seven-encoder reads
exceed 1 ms; the monitor does not certify a 1 ms maximum. Each arm reports
`read_latency_within_1ms`, cumulative `read_errors`, and timing violation counts. The
`usb_reads` field separately measures nonblocking host serial read calls, including
a count of calls over 1 ms. A short USB read call only means available bytes were
retrieved quickly; it does not mean a fresh complete frame arrived within 1 ms.
`read_duration_ms` includes the request, all seven replies, and decoding.

## Guided calibration through the monitor

Keep the monitor running. In a second terminal, start the calibration wizard:

```sh
.venv/bin/python -m deploy.gello.control.calibrate_stream \
  --side left --output deploy/gello/calibrations/left.json
```

Repeat with `--side right --output deploy/gello/calibrations/right.json` afterward.
The default simulator target is `[0, 0, 0, 1.5708, 0, 0]` radians: joint 4
at its upper stop. The reference viewer uses the same shared profile with
`python -m deploy.gello.sim.teleop --gello-home`. Press H in that viewer to
return both simulated arms to this reference. Joint 4 defaults to a negative
direction check (J), away from the stop. Explicit `--home` or `--negative-joints`
options override those defaults.
The wizard has 13 prompts: home, six direction checks with neutral returns between
joints, then trigger endpoints. Joint min/max sweeps are not required. After the
last direction check, keep the arm wherever comfortable, fully close/open the
trigger, and press Enter. Trigger extrema survive dropped reads and update live.
There is no final return-to-neutral step.

Version 2 calibration maps each arm joint with
`robot_home + sign * (continuous_encoder_position - leader_home)`.
One degree on a leader gives one degree of requested robot motion; the simulator
clips only at its own model limits. Version 2 files have `raw_min` and `raw_max`
set to null. These references use continuous torque-off encoder counts: after a
motor power cycle, torque enable, or mode change resets those counters, recapture
the home reference before using the saved mapping. Version 1 bounded calibrations
remain readable for compatibility.

For this resting pose, `--negative-joints 4` tests joint 4 away from its physical
home stop in the negative simulator direction (viewer key J). The sign calculation
accounts for the reversed test; the neutral pose and limits are unchanged.
Each prompt explains what to move and waits for Enter. Follow direction checks
with the MuJoCo reference viewer: L/R selects the arm, 1–6 selects the joint,
K increases its angle, and J reverses it. The comfortable leader reference pose
maps to the specified simulator home; it need not have identical geometry.

The calibration app reads a local Unix socket at `/tmp/gontrol-encoders-<uid>.sock`
(default; both apps accept `--socket`). Only the monitor opens the USB ports.
The socket supplies new states independently of the 5 Hz JSON file. Slow clients
cannot block encoder acquisition. The wizard pauses on stale, failed, or disconnected streams and reconnects.
During direction checks it retains accepted stages and requires neutral before
retrying interrupted movement. During trigger capture it retains min/max values,
reconnects in place, and accepts Enter without restarting the sweep.
A changed arm identity stops recovery. It preserves a `.progress.jsonl` journal
next to its output every second and after accepted steps. To recover a terminated
wizard, add `--resume` to the same command; its configuration must match the
latest journal record. Encoder revolution alignment is reestablished only after
you confirm neutral. Finished files
are created exclusively and never overwritten. Ctrl-C stops calibration only.

## Optional joint safety envelope

After both calibrations, capture all allowed joint ranges in one pass:

```sh
.venv/bin/python -m deploy.gello.control.record_limits \
  --left deploy/gello/calibrations/left.relative.json \
  --right deploy/gello/calibrations/right.json \
  --output deploy/gello/calibrations/joint-envelope.json
```

Start near the saved home poses to establish encoder turns, then sweep the desired
ranges on both arms. The app prints all min/max values and retains observed limits
through dropped reads. Press Enter once to save; no final neutral pose is needed.
The envelope is bound to the calibration files by fingerprints. It stores robot
joint-angle bounds; it does not scale motion or certify collision-free Cartesian
workspace. Persistent min/max progress is written beside the output file.

Run simulated teleoperation through the same monitor, with optional bounds:

```sh
python -m deploy.gello.sim.gello --socket \
  --left deploy/gello/calibrations/left.relative.json \
  --right deploy/gello/calibrations/right.json \
  --limits deploy/gello/calibrations/joint-envelope.json
```

The simulator starts at the calibrated robot home. Space enables, X holds, Q quits.
It clamps 1:1 targets to the intersection of the measured envelope and model limits,
then applies its speed limit. Stale data disarms teleoperation; recovery needs Space.
Continuous relative mapping handles crossings at 0/360 degrees; initialize near
home so the first encoder revolution can be identified. The monitor remains the
only USB owner. No physical follower or leader torque commands are sent.

## Capture raw joint poses

The read-only pose command prints encoder angles in radians and degrees, ordered
base through wrist and then trigger. These are raw encoder positions, not
calibrated YAM angles; the trigger is also an angle here, not normalized aperture.
From the repository root, for this station's left leader:

```sh
.venv/bin/python -m deploy.gello.control.poses \
  --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6Y3-if00-port0 \
  --baudrate 2000000 --output deploy/gello/calibrations/left-pose.jsonl
```

For the right leader use adapter `FTBEO6Y6` and `--baudrate 2000000`.
Add `--count 100 --interval 0.1` for a sequence. Output files are created
exclusively and flushed after each pose; existing files are never overwritten.
Communication failures exit with an error instead of saving invalid angles.

## Assign IDs on the unconfigured leader

Keep the configured leader's IDs. Each separate bus can use IDs **1–7**:
1 = base joint, 2–6 = successive joints toward the wrist, 7 = trigger.
Record which USB serial number belongs to each physical side; do not infer
left/right from ttyUSB numbering or adapter enumeration order.

For each motor on the unconfigured leader:

1. Power off, disconnect the chain, and connect **only the motor being assigned**.
2. Power it on and scan its port to obtain its current ID and model number.
3. Run the command below with the observed values. It prompts you to confirm
   physical isolation, checks model and torque-off state, writes only the ID,
   then verifies the new ID. It does not enable torque or move a motor.
4. Power off before changing wiring. Repeat, then reconnect the complete chain
   and scan for seven distinct IDs in the intended physical order.

```sh
python -m deploy.gello.control.gello_cli set-id \
  --port /dev/serial/by-id/YOUR_ADAPTER --baudrate 57600 \
  --old-id 1 --new-id 7 --model 1240
```

The values above are an **example**, not a hardware prescription. A scan cannot
prove isolation: multiple motors with the same ID can respond indistinguishably.
Never run an ID write against a connected chain with duplicate IDs.

## Calibrate each leader

First inspect positive joint directions and the home pose using the existing
MuJoCo keyboard viewer (`python -m
deploy.gello.sim.teleop`). Select joints with 1–6 and jog positive with K.
The simulator's home arm angles are `[0, 1.047, 1.047, 0, 0, 0]` radians.

From the repository root:

```sh
mkdir -p deploy/gello/calibrations
python -m deploy.gello.control.gello_cli calibrate \
  --side left --port /dev/serial/by-id/LEFT_ADAPTER --baudrate 57600 \
  --output deploy/gello/calibrations/left.json
python -m deploy.gello.control.gello_cli calibrate \
  --side right --port /dev/serial/by-id/RIGHT_ADAPTER --baudrate 57600 \
  --output deploy/gello/calibrations/right.json
```

The guided routine disables leader torque after you support the arm and press
Enter. It continuously reads encoders while you:

1. Match the simulator home pose, trigger open.
2. Move each joint individually in the simulator's positive direction to learn
   its sign and check motor ordering.
3. Slowly sweep every joint and trigger through the full **usable** range.
4. Capture fully closed and fully open trigger positions.

No powered stop-seeking occurs. Limits are measured software limits; nothing
writes motor position limits to EEPROM. The arm mapping uses 1:1 angular
motion about the home reference, followed by clipping to the MuJoCo limits.
It does not stretch a short leader sweep over the entire follower range.
Trigger endpoints map to closed=0/open=1 regardless of encoder direction.

At least 0.1 rad of movement per joint is required. This implementation accepts
usable travel **less than one full revolution** per encoder to avoid ambiguous
absolute pose after power cycling. Keep movement below half a revolution per
sample. Multi-turn mechanisms need a separate homing procedure before use.
Calibration records the USB path, baud, ordered IDs, models, signs, home and
travel. Recalibrate after mechanical changes or changing motor homing offsets.
Existing calibration files are never overwritten; choose a new name to recalibrate.
Each confirmed step is also appended to `<output>.progress.jsonl` with its prompt,
raw/unwrapped encoder values, measured extrema, session ID and device identity.
This journal preserves intermediate captures for inspection/recovery; it is not
a finished calibration and is not automatically loaded on restart. Wrong-joint
or too-small movements can be retried in the same session. Up to two brief
read failures are retried within a 200 ms sampling-gap bound; persistent failure
stops calibration and leaves the journal intact.

## Teleoperate MuJoCo

From the ABC repository root:

```sh
python -m deploy.gello.sim.gello \
  --left deploy/gello/calibrations/left.json \
  --right deploy/gello/calibrations/right.json
```

The viewer starts disarmed. **Space** enables tracking, **X** holds the simulated
arms, **Q** or closing the viewer quits. Leader torque must already be off.
Arm commands approach the leader at up to 0.5 rad/s; trigger commands at up to
0.5 normalized aperture/s. Samples older than 200 ms, inter-arm skew above
100 ms, malformed samples, communication errors or travel beyond calibration
stop tracking. Recovery requires Space; a failed serial reader requires restart.
Limits and slew limiting do not provide collision avoidance.

For a hardware-free test:

```sh
python -m deploy.gello.sim.gello --mock --headless --fast \
  --seconds 3 --record /tmp/gello-smoke.npz
```

`--mock` is explicit and never used as a fallback for broken hardware. It
produces synthetic motion through the same mapping, controller and physics.

Add `--seconds 60 --record /tmp/my-episode.npz` to a viewer run to capture a
**motion diagnostic**, not a finished training dataset. Recordings contain
14-D `state` before the action, the bounded `action`, `next_state`, unclipped
`leader_target`, `armed`, wall/simulation/sample times and JSON metadata with
calibration. Invalid leader targets/timestamps are NaN and marked disarmed.
Recordings preserve interruption/failure status and refuse overwrites.
No camera images are recorded; physical RealSense capture and the upstream
training episode format remain a later integration. Recording/headless runs
are bounded to ten minutes to bound in-memory storage.
