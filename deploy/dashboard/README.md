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

Left/right GELLO identity was verified by moving only the left base joint.
Camera placement, joint signs, gripper configuration, and absolute calibration
still require operator verification. Camera labels are neutral until assigned.
The configuration deliberately has no calibrated offsets and no verified
physical mapping. The existing motor baud rates have not been changed.

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

## Use simulation

1. Open the dashboard and inspect raw encoder readings under **Joint data**.
2. Keep both GELLO arms still and click **Preview GELLO motion from current pose**.
3. Move either GELLO. Its changes in joint angles drive the corresponding
   simulated arm from its home pose. Gripper preview starts at half-open.
4. Orbit/zoom the 3D scene, inspect target response, and watch camera streams.

This preview is a relative mapping for simulation. It does not establish an
absolute hardware zero. Stale or incomplete encoder samples never update sim
commands; the simulated arm holds its last target. Reset clears the preview
mapping so you can capture a new starting pose.

For absolute mapping, put the GELLO in its documented zero pose (six joints at
zero and gripper reference at 0.357 rad), then use **Capture absolute zero pose**.
Verify joint signs and gripper endpoints in simulation. Do not capture an
arbitrary resting pose as an absolute zero. Calibration is saved in the supplied
configuration file, but robot control state is never saved.

## Physical control status

**Physical connection and teleop are locked in this release.** The installed
i2rt driver has these verified lifecycle problems:

- Its `MotorChainRobot.close()` calls a chain `close()` that only clears a running
  flag; it does not confirm motor-off or close the CAN interface.
- Its factory can energize motors before returning an object, leaving no object
  for the caller to clean up if initialization fails.
- Its gripper calibration imports the utility inside the constructor, bypassing
  the repository's module-level calibration patch.

The dashboard displays the physical-control buttons and the lock reason. The
production adapter rejects connection before constructing the controller.
Removing the UI lock alone cannot energize hardware. Fixing and validating the
installed driver lifecycle is required before physical commissioning; do not
remove the backend guard to bypass it.

The supervisor has tested alignment checks (0.15 rad / 0.1 gripper), freshness
checks (200 ms), target slew limits (0.5 rad/s and 0.5 gripper units/s), a per-browser
lease (500 ms), and fault/hold transitions. Tests use fake adapters and do not
establish physical safety. Its browser heartbeat uses the Viser client socket;
hiding the controlling tab stops the heartbeat. No YAM controller is initialized
by simulation, camera capture, calibration, or observation subscription.

Physical YAM telemetry subscribes to existing `follower_left_obs` and
`follower_right_obs` ZMQ streams without starting a controller. If no observation
publisher exists, the panel correctly shows unavailable. It does not display
simulated values as physical readings.

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
hardware and labels its generated motion. i2rt is only needed for future physical
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

These settings may reset after reconnect/reboot. The reader uses sync reads when
available and falls back to checked per-servo reads when sync transfers fail.
Raw telemetry reports read mode and acquisition duration. It retains the oldest
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
