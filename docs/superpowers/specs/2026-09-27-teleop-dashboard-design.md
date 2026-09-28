# Headless GELLO teleop dashboard

## Outcome and scope

Run a dashboard on the robot station and open it through an SSH tunnel. Moving
the connected GELLO arms drives YAM arms in MuJoCo. The same page shows raw
GELLO readings, mapped commands, simulation state, available physical YAM
telemetry, and live RealSense images. The operator can explicitly enable
physical teleop and return to simulation from the dashboard.

The user approved this direction with “continue” after the proposed design.
This document is the written specification for review before implementation.

Startup is simulation only. Creating the dashboard must not initialize the
existing YAM follower controller: its startup can calibrate the gripper and
move the arm. Physical commissioning is performed by the operator using the
dashboard controls after simulation and mapping are verified.

Recording, policy inference, model training, and remote multi-user control
are outside this change. “ABC/YAM data” means the deployment stack's robot
observations and commands; no learned policy is required for teleop.

## Existing code and station evidence

- `abc_minimal/viz_policy.py` already uses Viser and `ViserMujocoScene` for
  browser rendering of MuJoCo state.
- `deploy/robot/leaders/` contains Dynamixel communication and GELLO mapping.
  Its automatic offset calibration assumes a calibration pose, so a new
  session must not silently interpret an arbitrary starting pose as calibrated.
- `deploy/robot/followers/yam_follower.py` provides observation normalization
  and a gripper calibration fix. Its boot and shutdown motion routines are
  unsuitable for a simulation-mode switch.
- `deploy/robot/cameras/realsense.py` provides RGB capture by camera serial.
- The station exposes CAN interfaces `can_l_foll` and `can_r_foll`, three
  RealSense D405 devices, and these serial adapters:
  - `usb-FTDI_USB__-__Serial_Converter_FTBEO6Y3-if00-port0`
  - `usb-FTDI_USB__-__Serial_Converter_FTBEO6Y6-if00-port0`
- Device nodes are accessible outside the execution sandbox. Their absence
  inside the sandbox is not evidence that hardware is disconnected.
- `/home/tarik/code/abc/.venv` contains MuJoCo, Viser, mjviser, RealSense,
  Dynamixel, and i2rt packages. The adjacent checkout contains YAM mesh assets;
  this checkout does not yet contain them.

USB discovery does not establish left/right identity, servo IDs, joint signs,
encoder offsets, camera placement, gripper type, or hardware health. The
implementation must discover what it can without movement, expose the mapping,
and require a verified station configuration before physical control.

## Architecture

Use one Viser server bound to `127.0.0.1:8080`. Reuse mjviser to display a
MuJoCo scene and Viser panels for controls, tables, and camera images. Rendering
happens in the browser, so no X server, desktop session, or X forwarding is
required. A standalone web frontend would offer more layout freedom but add a
second interface and build pipeline; it is unnecessary for this dashboard.

Add a focused `deploy/dashboard/` package and a launch entrypoint. Keep these
responsibilities separate:

1. Station configuration and discovery: stable device paths, arm assignments,
   calibration, camera serials, rates, and control limits.
2. GELLO readers: raw samples, calibrated targets, acquisition timestamps,
   validity, and device errors. Simulation use does not enable force feedback.
3. Simulation: model ownership, actuator mapping, physics stepping, and state
   snapshots. Drive actuators and step physics rather than overwrite joint
   positions each frame.
4. Camera workers: independently acquire images and publish latest frames.
5. Physical robot adapter and control supervisor: initialization, observations,
   command validation, mode transitions, and watchdog.
6. Dashboard: present snapshots and submit control requests to the supervisor.

Each hardware device has one owner. Use bounded latest-sample buffers with
monotonic acquisition timestamps. Camera capture and browser updates must not
block the robot loop. Mode changes and commands are serialized by the control
supervisor; GUI callbacks must not directly command hardware.

Default targets are 100 Hz GELLO acquisition, 30 Hz command updates, 30 Hz scene
updates, 10 Hz telemetry display, and 15 Hz displayed camera frames. MuJoCo
substeps follow its model timestep. Show measured rates and sample ages.

## Dashboard behavior

The main view contains the two YAM arms in the existing empty station scene.
Orbit, zoom, and a reset-view control remain available. Simulation stays active
while physical teleop is enabled so the operator can compare target behavior.
An unavailable arm remains visibly disconnected; never substitute synthetic
hardware readings. Provide an explicitly labeled hardware-free demo option
for development and testing.

Panels show:

- Current mode, device health, controller owner, and faults.
- Per GELLO: adapter, servo IDs, raw encoder counts, encoder angles,
  calibration offsets/signs, six mapped joint angles, normalized gripper,
  timestamp, and sample age.
- Per simulated arm: target and actual joint position/velocity, gripper state,
  and target error. Label simulated values and units explicitly.
- Per physical YAM: available raw motor feedback and normalized joint
  position/velocity/effort, gripper, command, errors, and sample age.
- Per RealSense: serial, configured placement, RGB image, capture rate, and
  error status. Offer a depth preview when supported; RGB is the default.

Show physical telemetry in simulation mode using a verified non-actuating
source, such as existing observation publications or passive CAN reception.
Never call a controller constructor merely to populate telemetry. If the
powered robot does not emit passive feedback, display “unavailable until robot
connection” with its reason. Live physical observations become available when
the operator explicitly connects the robot.

Controls include calibration/mapping status, simulation reset, Connect robot,
Enable robot teleop, and Return to simulation. Simulation reset never changes a
physical target. Unknown calibration permits raw inspection but disables
mapped tracking and physical enable. Support a single configured arm as well
as both arms; side assignments must be explicit.

## Physical control and failure behavior

Use these states:

- **Simulation only:** no physical controller initialized; GELLO drives sim.
- **Connecting:** entered only by Connect robot. Explain that connection can
  energize motors and calibrate the gripper before the operator clicks it.
- **Robot holding:** initialization succeeds and the robot holds its measured
  pose; GELLO continues driving sim. Show alignment error per joint.
- **Robot teleop:** enabled only by the owning browser with valid configuration,
  fresh leader/robot samples, and aligned poses.
- **Fault:** reject further GELLO commands, report the cause, and require an
  explicit recovery before another enable.

For initial conservative defaults, require every arm joint to be within
0.15 rad of its mapped target and normalized gripper error within 0.1 at enable.
Reject nonfinite values, invalid shapes, out-of-range targets, samples older
than 200 ms, and invalid calibration. Apply a 0.5 rad/s arm target slew limit
and a 0.5/s normalized gripper limit. These are software defaults, not a
collision-avoidance guarantee. Station limits can narrow them.

The browser that enables teleop owns the control lease. Require an application
heartbeat and revoke the lease after 500 ms without it. A disconnect, stale
hardware sample, or control fault revokes teleop. Reconnection never re-enables
it automatically. The watchdog runs with the physical controller, independently
of camera and UI updates.

Return to simulation stops accepting GELLO targets and holds the last measured
physical pose while simulation continues. Clearly show “simulation / robot
holding” because the motors remain energized. If fresh feedback is lost, use
the last validated hold target and report a fault. If communication itself is
lost, report that stopping cannot be confirmed. A separate release/disconnect
control describes that torque release can let an unsupported arm fall.

Do not reuse the existing follower shutdown move-to-zero. Shutdown must cancel
teleop, stop worker activity, and perform explicit controller cleanup without
a homing move. Inspect and test the underlying driver's stop/watchdog behavior;
if it cannot enforce the required behavior, keep physical enable unavailable
and report the precise blocker rather than claim successful physical support.

## Configuration, launch, and access

Provide one documented launch command with optional configuration, port, arm
selection, and demo mode. Persist calibration and device mapping separately
from runtime mode; never persist an enabled teleop state. Resolve model assets
through an explicit path or the existing asset mechanism, with actionable
errors for missing meshes. Avoid installing the training stack just to run
the dashboard.

Print the actual listening address, log location, and SSH instructions. Fail
clearly if the requested port is occupied rather than silently choose a port.
From the operator's computer:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8080:127.0.0.1:8080 tarik@100.95.170.113
```

Open `http://localhost:8080`. All dashboard traffic uses this tunnel. The
station address comes from this session's SSH connection; users with an SSH
host alias can substitute it. Document a persistent launch and clean shutdown
method for a dashboard that should outlive an SSH terminal.

## Verification and acceptance

Automated tests must cover command mapping and gripper conversion, simulation
actuation, calibration validity, mode transitions, alignment rejection,
staleness, nonfinite/out-of-range commands, slew limits, control ownership,
heartbeat expiry, and shutdown. Fake hardware must record constructor calls
and commands so tests prove simulation startup does not initialize or command
the physical controller. Test browser loss independently of scene updates.

Run the real MuJoCo model headlessly and confirm physics advances and responds
to deterministic targets. Verify the browser scene and telemetry panels with
the installed Viser version. Test camera failure without interrupting sim or
control, and verify missing-device status rather than fabricated values.

On this station, verify USB identities, readable GELLO samples, and camera
streams without enabling the physical follower. Leave the delivered dashboard
running in simulation mode when feasible. Report exactly which streams and
checks succeeded. Physical motion acceptance requires the operator to verify
mapping/calibration and use the control buttons; do not enable it automatically
as part of deployment testing.

Success means the user can tunnel to the running dashboard, move a calibrated
GELLO and observe MuJoCo respond, inspect measured data and camera streams, and
explicitly control whether the physical robot follows. Any unverified physical
capability is called out in the handoff.
