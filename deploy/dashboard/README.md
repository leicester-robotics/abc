# GELLO / YAM dashboard

Browser view of a headless MuJoCo simulation, both GELLO encoder streams,
RealSense RGB/depth previews, and available ABC follower observations.

## This station

Start on the station, from `/home/tarik/code/abc-gello`:

```bash
/home/tarik/code/abc/.venv/bin/python -m deploy.dashboard \
  --config deploy/dashboard/configs/station.json
```

The station configuration uses port **8081** because an existing SSH tunnel
occupies 8080. It records these discovered devices:

| Device | Identity |
| --- | --- |
| Left GELLO | FTBEO6Y6 (`ttyUSB0`), IDs 1–7 at 2 Mbps |
| Right GELLO | FTBEO6Y3 (`ttyUSB1`), IDs 1–7 at 2 Mbps |
| RealSense D405 cameras | 427622271843, 427622273082, 427622272514 |

Saved calibration from `origin/mhasek/abc_gello` commit `d4abb33` supplies
joint signs, home angles, and gripper endpoints. The saved side labels are
reversed on this station: each calibration stays with its USB adapter, while
the operator-confirmed side mapping is left Y6 / right Y3. Both arms use
joint signs `[1, -1, 1, 1, 1, 1]`. Verify the mapping in simulation before physical
commissioning. Motor baud rates remain unchanged at 2 Mbps.

The simulator starts and resets at `[0, 0, 0, 1.5708, 0, 0]` with the gripper
half open. Saved relative calibration aligns encoders to the nearest revolution
of the saved home on connection; keep the leader near that pose when reconnecting.
Camera labels remain neutral until placements are assigned.

### Open it from your computer

Run this in a **local terminal on your laptop/desktop**, outside the station's
SSH session:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 18081:127.0.0.1:8081 tarik@100.95.170.113
```

Leave it running and open **http://127.0.0.1:18081** on that computer. A successful
`ssh -N` command normally prints nothing and stays running. Substitute your usual
SSH host alias if needed. The first port (`18081`) is on your computer; the final
port (`8081`) is on the station. If the local port is occupied, change only the
first port and use the new port in your browser. A tunnel launched *inside* the
station's SSH session will not make your computer's localhost reach the station.
VS Code Remote SSH users can instead forward station port 8081 in the Ports tab.

### Keep it running

```bash
tmux new-session -d -s gello-dashboard \
  'cd /home/tarik/code/abc-gello && /home/tarik/code/abc/.venv/bin/python -m deploy.dashboard --config deploy/dashboard/configs/station.json >> outputs/dashboard/server.log 2>&1'
```

Create `outputs/dashboard/` first if needed. View output with
`tail -f outputs/dashboard/server.log`. Machine-readable device health is in
`outputs/dashboard/status.json`. To stop gracefully, attach with
`tmux attach -t gello-dashboard` and press Ctrl-C. This allows camera and serial
handles to close. A later launch starts in simulation mode again.

Fault details appear above the tabs with the original error, time, control stage,
plain-language explanation, and recovery instructions. Later failed button clicks
do not overwrite the recorded fault. The same record is available in status.json.

## Dashboard layout

- **Simulation:** robot view, preview/reset, and physical-control status.
- **Cameras:** live RGB/depth previews; this tab enables image transmission.
- **Arm data:** wide panel with GELLO / MuJoCo / Physical YAM tabs and Left / Right selectors.
- **Teleop settings:** live GELLO/alignment margins, heartbeat deadline, speeds, and force-feedback strength. Apply validates the whole edit; active control belongs to one browser. Settings are per session and reset on restart.
- **Setup:** calibration, adapter identities, and physical-control lock details.

## Use simulation

This station loads its saved calibration automatically. Move the right GELLO to
control the right simulated arm. The left GELLO controls the left simulated arm. Absolute-zero capture buttons are disabled for imported calibrations
so they cannot overwrite the saved mapping.


1. Open the dashboard and inspect raw encoder readings under **Arm data**.
2. Keep both GELLO arms still and click **Preview GELLO motion from current pose**.
3. Move either GELLO. Its changes in joint angles drive the corresponding
   simulated arm from its home pose. Gripper preview starts at half-open.
4. Orbit/zoom the 3D scene, inspect target response, and watch camera streams.

This preview is a relative mapping for simulation. It does not establish an
absolute hardware zero. Stale or incomplete encoder samples never update sim
commands; the simulated arm holds its last target. Reset clears the preview
mapping so you can capture a new starting pose.

On an uncalibrated station, put the GELLO in its documented zero pose (six joints at
zero and gripper reference at 0.357 rad), then use **Capture absolute zero pose**.
Verify joint signs and gripper endpoints in simulation. Do not capture an
arbitrary resting pose as an absolute zero. Calibration is saved in the supplied
configuration file, but robot control state is never saved.

## Physical control status

Physical control requires a verified isolated i2rt copy with the included lifecycle patch.
Prepare it once, then launch with its path:

```bash
/home/tarik/code/abc/.venv/bin/python -m deploy.dashboard.driver_setup outputs/dashboard/sdk-v3
DASHBOARD_SDK_DIR="$PWD/outputs/dashboard/sdk-v3" PYTHONPATH="$PWD/outputs/dashboard/sdk-v3" /home/tarik/code/abc/.venv/bin/python -m deploy.dashboard --config deploy/dashboard/configs/station.json
```

The patch verifies motor disable acknowledgments, drains fault-clear responses,
joins command threads before release, and retains ownership after partial startup.
The shared installed SDK is unchanged. An unverified SDK keeps physical control locked.

1. Support both arms and clear the grippers, then check the confirmation in Robot control.
2. **Connect robot** calibrates gripper endpoints, moves to the saved home
   `[0, 0, 0, 1.5708, 0, 0, 0.5]` at up to 0.25 rad/s, and holds.
3. **Enable teleop** gradually aligns to GELLO, then follows at up to 0.5 rad/s.
   Keep the leaders still during alignment and the controlling browser visible.
4. **Simulation only** stops following and holds the physical arms.
   **Release robot** disables their motors.

Force feedback defaults on and activates only during teleop. GELLO compliant hold
also defaults on: a current-limited spring resists displacement and moves its
anchor after the configured yield distance (default 6 degrees). It holds within
the available 75 mA combined current cap and can sag under heavier loads. It
is not gravity compensation. Hold strength and yield distance are adjustable
in Teleop settings; zero strength disables hold. The gripper is excluded.
Faults, stale data, or stopping teleop release leader hold. It reflects measured
motor effort, including gravity and friction, with a 75 mA per-motor hardware cap,
15 mA/Nm default reflection gain, a 150 mA/s current ramp, a 100 ms freshness deadline, and a motor bus watchdog. The serial
reader owns all leader writes and restores mode/current limits after disabling.
This has offline coverage; physical haptic behavior still needs operator validation.

GELLO joint targets allow 5 degrees of overtravel and clamp to the robot limits;
physical feedback has a separate 0.03 rad stop-error allowance.
The controller holds on stale feedback, loss of the 1 second browser lease, or a
moving leader during alignment. Startup does not enable following automatically.
Gripper calibration holds the arm joints and checks both endpoints against the
station's observed 4.5–6.0 rad travel range (historical station-shadow reports).
The measured closed stop maps to zero; no 5% overtravel offset is added to calibration.

Physical YAM telemetry comes from complete motor batches while connected, or
existing follower ZMQ observations while disconnected.

## Dependencies and assets

This station already has the required environment and YAM meshes in the adjacent
`abc` checkout. `configs/station.json` names that mesh path explicitly. On a new
machine, create a small environment rather than installing the training stack:

```bash
python3 -m venv .venv-dashboard
.venv-dashboard/bin/pip install -r deploy/dashboard/requirements.txt
.venv-dashboard/bin/python -m deploy.dashboard --demo \
  --asset-dir /path/to/i2rt_yam/assets
```

Run from the repo root. Obtain the YAM model assets using the repository's asset
preparation instructions; `--asset-dir` points to the folder containing the STL
files. `--model-path` selects a compatible bimanual YAM scene. `--demo` opens no
hardware and labels its generated motion. i2rt is needed for physical
controller integration; passive observation uses ZMQ.

Use `python -m deploy.dashboard --discover` to list camera and USB identities.
The server binds only to loopback. All dashboard and image traffic uses the one
SSH-forwarded port. No display server or X forwarding is required.

## Device access and timing

If serial access is denied:

```bash
sudo setfacl -m u:tarik:rw /dev/ttyUSB0 /dev/ttyUSB1
```

For lower USB buffering latency:

```bash
echo 1 | sudo tee \
  /sys/bus/usb-serial/devices/ttyUSB0/latency_timer \
  /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
```

The GELLO worker targets 400 Hz; physics runs independently at 100 Hz and the
viewer publishes at up to 60 Hz. Measured Y3 read batches after setting 1 ms USB
latency: median 0.98 ms, p95 1.83 ms over 100 batches at 2 Mbps. In-service rates
vary with load and are displayed under Arm data. Both adapters now provide live sync reads at approximately 380 Hz.

Camera previews are sent only while a client has Cameras selected. Capture
continues at 30 fps. This reduces tunnel traffic when using the simulation.

These settings may reset after reconnect/reboot. The reader uses sync reads when
available and falls back to checked per-servo reads when sync transfers fail.
Encoder packet timeouts retry on the existing serial connection. Teleop can use
the last complete sample for up to 200 ms; partial batches never refresh its age.
A missed batch zeros active leader feedback. Hardware and haptic-write errors
remain fatal. Raw telemetry reports read mode and acquisition duration. It retains the oldest
sample acquisition time so a slow batch cannot look artificially fresh.

## Verification

```bash
/home/tarik/code/abc/.venv/bin/python -m unittest discover -s tests/dashboard -v
```

The real-browser test is opt-in and needs Playwright plus Chrome:

```bash
DASHBOARD_BROWSER_TESTS=1 python -m unittest tests.dashboard.test_browser -v
```

It tests browser-origin heartbeat updates, hidden-tab expiry, separate clients,
and disconnect cleanup without physical hardware. Simulation tests require the
YAM meshes; set `DASHBOARD_TEST_ASSETS` to their directory on another machine.
