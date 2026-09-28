# Headless GELLO Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a running SSH-accessible dashboard for GELLO-driven MuJoCo simulation, raw arm telemetry, RealSense streams, and explicitly enabled physical teleop.

**Architecture:** A Viser server displays MuJoCo state and device snapshots. Independent acquisition workers feed bounded buffers; a serialized control supervisor owns physical commands and a browser control lease. Simulation startup never constructs a physical controller.

**Tech Stack:** Python, NumPy, MuJoCo, mjviser, Viser, Dynamixel SDK, pyrealsense2, i2rt, ZMQ, standard-library unittest.

**Spec:** [Approved design](../specs/2026-09-27-teleop-dashboard-design.md)

## Global Constraints

- Startup is simulation only.
- Use one Viser server bound to `127.0.0.1:8080`.
- Default targets are 100 Hz GELLO acquisition, 30 Hz command updates, 30 Hz scene updates, 10 Hz telemetry display, and 15 Hz displayed camera frames.
- Require every arm joint to be within 0.15 rad of its mapped target and normalized gripper error within 0.1 at enable.
- Reject samples older than 200 ms. Revoke the browser lease after 500 ms without an application heartbeat.
- Apply a 0.5 rad/s arm target slew limit and a 0.5/s normalized gripper limit.
- Do not reuse the existing follower shutdown move-to-zero.
- Preserve acquisition timestamps; unavailable hardware must never display synthetic readings.
- Verify station mapping and calibration before physical control; do not enable physical motion during deployment testing.
- Use the installed environment through `/home/tarik/code/abc/.venv/bin/python` initially; document portable dependency and asset setup too.
- Existing checkout is a worktree. Verify its identity and instructions before execution; keep changes here.

## Review Focus

1. One arm connected or disconnected mid-session: healthy simulation and cameras continue; physical tracking faults. Covered by Tasks 2 and 4.
2. Worker restart with old queued data: acquisition timestamps remain old and cannot authorize enable. Covered by Tasks 1 and 4.
3. Two browsers or a hidden tab: only the owner can renew the lease; heartbeat throttling returns the robot to hold. Covered by Tasks 4 and 5.
4. Bad configuration or missing assets: actionable startup errors and no motor initialization. Covered by Tasks 1 and 3.
5. Partial initialization or shutdown failure: resources close, no homing command occurs, and uncertain physical state is reported. Covered by Tasks 2, 4, and 6.

## File map and shared data

Create `deploy/dashboard/` with `__init__.py`, `__main__.py`, `config.py`,
`samples.py`, `gello.py`, `cameras.py`, `simulation.py`, `yam.py`, `control.py`,
`ui.py`, and `app.py`. Add tests under `tests/dashboard/`, package markers for
unittest discovery, and `deploy/dashboard/README.md`. Update
`deploy/robot/README.md` with the dashboard entrypoint. Add a lightweight
dashboard dependency extra to `pyproject.toml` if required by the actual import
graph. Keep unrelated robot scripts unchanged.

`ArmSample` contains `acquired_at: float`, `position: ndarray` of length 7,
optional length-7 `velocity`/`effort`, and `raw: dict`. Arm positions use radians
for six joints and a normalized gripper. `LeaderSample` additionally carries
encoder counts, encoder radians, and `calibrated: bool`; its mapped position
is absent until calibration is valid. `CameraSample` contains acquisition time,
RGB array, and optional depth preview. Buffers store immutable copies and a
separate health/error value; reading a buffer never refreshes its timestamp.

## Task 1: Configuration and sample contracts

**Files:** `config.py`, `samples.py`, package markers, `tests/dashboard/test_config.py`, `tests/dashboard/test_samples.py`.

**Interfaces:** `load_config(path: Path) -> DashboardConfig`; `LatestSample.publish(sample)`, `LatestSample.read()`, `LatestSample.set_error(message: str)`; sample types above. Configuration contains per-side stable serial path, servo IDs, signs, offsets, calibration validity, CAN channel, gripper type, camera serial/label, limits, and model path.

- [ ] Write tests for duplicate side/serial/servo assignments, missing offsets, invalid signs, nonfinite calibration, and unknown gripper types. Assert invalid configuration raises `ValueError`; missing calibration remains explicitly uncalibrated.
- [ ] Write snapshot tests: `publish(sample_at_10); read_at_11.acquired_at == 10`; modifying the input array cannot modify the stored sample. Assert no enabled mode exists in serialized config.
- [ ] Run `python -m unittest tests.dashboard.test_config tests.dashboard.test_samples -v`; confirm failures arise from absent implementation.
- [ ] Implement strict config loading, atomic calibration saving, and bounded thread-safe snapshot storage. Defaults match Global Constraints; CLI overrides never infer side identity from enumeration order.
- [ ] Run the same tests and confirm they pass; commit this task's files as `feat: add dashboard configuration and sample contracts`.

## Task 2: GELLO and camera acquisition

**Files:** `gello.py`, `cameras.py`, `tests/dashboard/test_gello.py`, `tests/dashboard/test_cameras.py`.

**Interfaces:** `GelloReader(config, buffer).start()/close()`; `map_encoder(counts: ndarray, config) -> ndarray`; `CameraWorker(config, buffer).start()/close()`; `discover_devices() -> dict` reports serial adapters and RealSense identities without initializing followers.

- [ ] Write mapping tests asserting counts `[2048]*7` produce zero encoder radians, signs and offsets apply before gripper conversion, and configured gripper endpoints map to 0 and 1. Uncalibrated samples expose counts but no usable mapped position.
- [ ] Write fake-device tests asserting missing one servo invalidates the entire arm sample, a disconnected arm does not stop its peer, no torque-enable/current command occurs, and a read failure marks the sample stale/error without changing its acquisition time.
- [ ] Write camera tests asserting one stalled or absent camera leaves other workers running; partial pipeline startup closes opened resources; unsupported depth leaves RGB available with an explicit depth status.
- [ ] Run `python -m unittest tests.dashboard.test_gello tests.dashboard.test_cameras -v`; confirm expected failures.
- [ ] Implement read-only Dynamixel acquisition using the existing connection wrapper and checked sync reads. Verify per-servo availability and signed 32-bit positions. Use bounded read timeouts. Provide explicit calibration capture from the documented pose; save offsets atomically, never auto-calibrate on process start.
- [ ] Implement independent RealSense workers using supported profiles, RGB conversion, optional depth colorization, and bounded frame waits. Discovery records serials; placement labels remain configurable.
- [ ] Run the task tests and confirm they pass; commit as `feat: stream GELLO samples and RealSense frames`.

## Task 3: Headless MuJoCo tracking

**Files:** `simulation.py`, `tests/dashboard/test_simulation.py`.

**Interfaces:** `Simulation(model_path: Path)` exposes `model`, `data`; `set_target(side: str, target: ndarray)`, `step(elapsed: float)`, `snapshot(side: str) -> ArmSample`, `reset()`.

- [ ] Write a MuJoCo test that maps all named arm/gripper actuators, sets a reachable target, steps 0.5 seconds, and asserts joint position changes toward the target while remaining finite. Assert gripper endpoints use actuator range and finger equality, rather than treating normalized gripper values as radians.
- [ ] Add tests for missing meshes with actionable path errors, one active arm, stale leader input holding the last simulation target, and bounded catch-up after a long scheduling pause.
- [ ] Run `python -m unittest tests.dashboard.test_simulation -v`; confirm expected failures.
- [ ] Implement named joint/actuator lookup and physics stepping using the model timestep. Initialize from the scene home keyframe. Resolve explicit asset paths without changing the adjacent checkout; reuse local YAM assets or the existing download mechanism as needed.
- [ ] Run the tests with the real empty station model headlessly; confirm they pass; commit as `feat: drive headless YAM simulation from joint targets`.

## Task 4: Physical adapter and control supervisor

**Files:** `yam.py`, `control.py`, `tests/dashboard/test_control.py`, `tests/dashboard/test_yam.py`.

**Interfaces:** `YamAdapter.connect()`, `observe() -> ArmSample`, `command(target: ndarray)`, `hold(target: ndarray)`, `close()`; `ControlSupervisor.request(action: str, client_id: str)`, `heartbeat(client_id: str, now: float)`, `tick(now: float)`, `status() -> dict`. Inject a clock and adapter factory in tests. Actions are connect, enable, simulation, recover, and release.

- [ ] Write fake-adapter tests asserting simulation startup makes zero adapter constructions and commands; connecting holds measured pose without a GELLO target; enabling requires calibrated configuration and fresh samples.
- [ ] Pin boundaries in tests: reject joint error 0.151 rad, gripper error 0.101, age 0.201 s, NaN/Inf, and out-of-range commands. At `dt=0.1`, command change is at most 0.05 per joint and gripper.
- [ ] Test lease expiry at age 0.501 s, nonowner heartbeats, stale queued samples after restart, disconnect during teleop, and reconnected browsers remaining disabled. Return-to-simulation holds measured pose; stale observations use last validated hold target and fault.
- [ ] Test partial adapter failure and shutdown: no `move_joints`/home command, no fresh tracking commands after revocation, cleanup attempted for every opened resource, and failure to communicate reported as unconfirmed stop.
- [ ] Run `python -m unittest tests.dashboard.test_control tests.dashboard.test_yam -v`; confirm expected failures.
- [ ] Audit installed i2rt constructor, calibration, state timestamps, command thread, CAN timeouts, and close behavior. Implement explicit-connect adapter using the repository's calibration fix. Acquisition timestamps must represent fresh driver feedback, not repeated reads of cached observations.
- [ ] Implement the serialized supervisor and independent 30 Hz worker/watchdog. Before enabling physical controls, prove the driver supports the freshness and hold behavior; otherwise keep enable disabled with the exact reason. This is a reported hardware blocker, not a passing physical acceptance test.
- [ ] Add passive observation support using existing ZMQ follower observation topics; validate source timestamps and layout. When absent, show unavailable until explicit connection. Never launch an existing follower merely for telemetry.
- [ ] Run task tests and confirm they pass; commit as `feat: gate physical teleop with alignment and watchdog checks`.

## Task 5: Viser dashboard and browser lease

**Files:** `ui.py`, `app.py`, `__main__.py`, `tests/dashboard/test_ui.py`, `tests/dashboard/test_app.py`.

**Interfaces:** `DashboardUI(server, supervisor).update(snapshot: dict)`; `run(config: DashboardConfig, demo: bool = False) -> int`; CLI `python -m deploy.dashboard --config PATH --port 8080` with `--demo` and `--model-path` overrides.

- [ ] Write UI binding tests asserting each button submits a supervisor request, simulation reset cannot command YAM, and missing calibration/lease support disables physical enable. Assert raw, mapped, simulated, and measured data have distinct labels and units.
- [ ] Write integration tests for occupied-port failure, hardware-free demo labeling, startup without follower imports/construction, and camera/UI delays not blocking supervisor ticks.
- [ ] Inspect the installed Viser extension mechanism for a browser-origin heartbeat. Implement a browser timer at 100 ms routed over the same server connection with authenticated client identity; do not substitute server-generated timestamps or generic websocket pings. Add a small frontend extension file only if needed by this installed API. If unsupported, keep physical enable disabled and report the required extension as a blocker.
- [ ] Test two browser clients: nonowner cannot renew or enable over the owner, heartbeat silence expires after 500 ms, and a hidden/throttled tab transitions to hold. A browser disconnect callback revokes immediately.
- [ ] Run `python -m unittest tests.dashboard.test_ui tests.dashboard.test_app -v`; confirm expected failures for absent behavior, then implement scene setup, tables, camera panels, controls, and update scheduling.
- [ ] Bind to loopback, reject port fallback, print actual URL, and support orderly signal shutdown. Show calibration instructions, connection effects, alignment errors, and robot-holding status in the UI.
- [ ] Run the tests and inspect the real dashboard in a browser if browser tooling is available. Confirm scene interaction, image updates, ownership, and heartbeat behavior; report any unavailable browser validation.
- [ ] Commit as `feat: add browser dashboard for simulation and teleop`.

## Task 6: Station bring-up and handoff

**Files:** `deploy/dashboard/README.md`, `deploy/robot/README.md`, `pyproject.toml` if needed, station configuration example with discovered identifiers, `tests/dashboard/test_shutdown.py`.

- [ ] Write a shutdown integration test asserting SIGTERM closes cameras/serial handles, revokes teleop, attempts controller cleanup, and never homes the robot. Run it failing, implement missing cleanup, then rerun passing.
- [ ] Verify current hardware visibility outside the sandbox as needed. Discover actual servo IDs and camera serials without movement. Record known identities; keep uncertain side mapping, signs, gripper type, and calibration explicitly unverified.
- [ ] Provide lightweight dependency instructions, model asset configuration, CLI help, calibration steps, and a persistent `tmux` launch plus graceful shutdown command. Keep machine-specific config separate from portable defaults.
- [ ] Run `python -m unittest discover -s tests/dashboard -v`, `python -m deploy.dashboard --help`, and `git diff --check`; fix failures within scope.
- [ ] Start the real dashboard in simulation mode. Verify HTTP access, actual port, readable encoder samples, camera updates, and MuJoCo stepping. Do not click Connect robot or Enable robot teleop during autonomous checks.
- [ ] If calibration requires the operator to place the GELLO in its known pose, leave raw readings and cameras live, show the calibration control, and explain this remaining action. Never fabricate offsets to claim successful tracking.
- [ ] Review the complete diff against the approved spec, including driver limits and browser lease. Commit documentation and verification fixes as `docs: document station dashboard setup and verification`.
- [ ] Handoff: report running process/log, URL, each verified device/stream, tests, any physical-control blocker, and the exact local tunnel command:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8080:127.0.0.1:8080 tarik@100.95.170.113
```

## Plan self-review

The six tasks cover the approved spec's simulation, device acquisition,
telemetry, controls, failure handling, launch, and acceptance sections. Shared
interfaces use one position convention and acquisition clock. Each Review
Focus item has tests assigned above. Physical telemetry availability, driver
freshness, browser heartbeat support, and operator calibration are explicit
verification gates; they cannot be silently replaced by success claims.
