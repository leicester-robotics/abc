"""Run the policy on this station with one command.

Checks the station, starts the policy server in the background (or reuses a
compatible one already listening on the port), then runs local_robot.deploy_policy
in this terminal. Change the prompt from another terminal with
local_robot.set_prompt; it is relayed to the server's stdin.

  uv run python -m local_robot.run_policy --prompt='pick up the cube' --debug
  uv run python -m local_robot.run_policy --policy=vla --prompt='pick up the cube'

The policy kind (DiT or VLA) is read from the checkpoint and picks the RTC
lead and chunk defaults; on a shared GPU the VLA switches to int8 layers.
Unknown arguments go to deploy_policy (e.g. --init-q, --record, --verbose);
--server-args goes to the server module.
"""

import argparse
import ctypes
import getpass
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = "local_yam_config"
DEFAULT_PROMPT = "throw plastic bottles in bin"
CONTROL_DT_MS = 33.33
PROMPT_CONFIRM_PATTERN = "[prompt] now:"
OTHER_SESSION_PATTERN = re.compile(
    r"local_robot\.(deploy_policy|run_policy)|deploy[./]deploy_policy|deploy[./]dagger"
)
GPU_HOG_MIB = 256
CHECKPOINTS = {"dit": "cache/bottles_75k.pt", "vla": "cache/vla_abc130k_200000_v2.pt"}
FAST_ROLLOUT = (7, 16)
"""(lead steps, execute chunk) when a chunk takes ~110 ms (DiT, VLA bf16 alone)."""
SLOW_ROLLOUT = (10, 20)
"""(lead steps, execute chunk) for the int8 VLA (~165-190 ms per chunk)."""
MIN_FREE_VRAM_GIB = {"dit": 4.5, "vla": 2.9}
VLA_BF16_FREE_GIB = 7.2
"""Free VRAM that keeps 30 of 34 VLA Gemma layers resident in bf16."""
MANAGED_SERVER_FLAGS = {
    "--policy.checkpoint-path",
    "--policy.prompt",
    "--policy.diffusion-steps",
    "--policy.rtc-prefix-length",
    "--policy.fast-inference",
    "--policy.no-fast-inference",
    "--policy-type",
    "--port",
}
REALSENSE_USB_VENDOR = "8086"
PR_SET_CHILD_SUBREAPER = 36


def prompt_socket_path(port: int) -> Path:
    return Path(f"/tmp/abc-{getpass.getuser()}/prompt-{port}.sock")


class PreflightError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(f"[run_policy] {message}", flush=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--policy",
        choices=sorted(CHECKPOINTS),
        help=f"Shortcut for --checkpoint-path: {CHECKPOINTS}.",
    )
    parser.add_argument(
        "--checkpoint-path", help=f"Default: {CHECKPOINTS['dit']} (or --policy's)."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--no-rtc", action="store_true")
    parser.add_argument("--rtc-prefix-length", type=int, default=4)
    parser.add_argument(
        "--rtc-inference-lead-steps",
        type=int,
        help=f"Default {FAST_ROLLOUT[0]}; {SLOW_ROLLOUT[0]} for an int8 or shared-GPU VLA.",
    )
    parser.add_argument(
        "--execute-chunk-dim",
        type=int,
        help=f"Default {FAST_ROLLOUT[1]}; {SLOW_ROLLOUT[1]} for an int8 or shared-GPU VLA.",
    )
    parser.add_argument(
        "--fast-inference",
        action="store_true",
        help="DiT only: torch.compile the samplers at server startup (takes minutes).",
    )
    parser.add_argument("--compress-images", action="store_true")
    parser.add_argument(
        "--reset-speed",
        type=float,
        default=0.25,
        help="Max joint speed (rad/s) of the move to init_q; 0 = upstream 2 s move.",
    )
    parser.add_argument(
        "--profile", default=os.environ.get("ROBOT_PROFILE", DEFAULT_PROFILE)
    )
    parser.add_argument("--server-module", default="local_robot.serve_local")
    parser.add_argument(
        "--server-args", default="", help="Extra server args (one string)."
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Fail instead of reusing a policy server already on --port.",
    )
    parser.add_argument(
        "--min-free-vram-gib",
        type=float,
        help=f"Default per policy: {MIN_FREE_VRAM_GIB}.",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        help="Seconds to wait for the server; default 600, 1800 with --fast-inference.",
    )
    parser.add_argument("--log-dir", default="data/logs")
    parser.add_argument(
        "--check", action="store_true", help="Run preflight, print commands, exit."
    )
    args, deploy_extra = parser.parse_known_args(argv)
    args.debug = "--debug" in deploy_extra
    if args.checkpoint_path is None:
        args.checkpoint_path = CHECKPOINTS[args.policy or "dit"]
    if args.server_timeout is None:
        args.server_timeout = 1800.0 if args.fast_inference else 600.0
    args.kind = args.policy
    args.offload_args = []
    return args, deploy_extra


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _processes():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        if raw:
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            yield (
                int(entry.name),
                ppid,
                raw.replace(b"\0", b" ").decode(errors="replace").strip(),
            )


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "?"
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def _check_other_sessions() -> None:
    procs = list(_processes())
    parents = {pid: ppid for pid, ppid, _ in procs}
    own = {os.getpid()}
    pid = os.getpid()
    while pid in parents and pid > 1:
        pid = parents[pid]
        own.add(pid)
    others = [
        (pid, cmd)
        for pid, _ppid, cmd in procs
        if pid not in own
        and OTHER_SESSION_PATTERN.search(cmd)
        and " uv run " not in f" {cmd}"
    ]
    if others:
        listing = "\n".join(f"  pid {pid}: {cmd}" for pid, cmd in others)
        raise PreflightError(
            "Another robot session is running; it holds the cameras and the IPC "
            f"sockets in /tmp/abc-{getpass.getuser()}. Stop it first (c/j or Ctrl-C "
            f"in its terminal):\n{listing}"
        )


def _check_can(profile, debug: bool) -> None:
    for robot in profile.robots.values():
        channel = robot.follower.channel
        result = subprocess.run(
            ["ip", "-br", "link", "show", channel],
            capture_output=True,
            text=True,
            check=False,
        )
        fields = result.stdout.split()
        state = fields[1] if len(fields) > 1 else "MISSING"
        if state == "UP":
            continue
        message = (
            f"CAN {channel} is {state}. Fix: sudo bash deploy/scripts/reset_all_can.sh"
        )
        if debug:
            _log(f"WARNING: {message} (ignored in --debug)")
        else:
            raise PreflightError(message)


def _realsense_video_nodes() -> set[str]:
    nodes = set()
    for dev in Path("/sys/class/video4linux").glob("video*"):
        path = (dev / "device").resolve()
        while path != path.parent and not (path / "idVendor").exists():
            path = path.parent
        vendor = path / "idVendor"
        if vendor.exists() and vendor.read_text().strip() == REALSENSE_USB_VENDOR:
            nodes.add(f"/dev/{dev.name}")
    return nodes


def _check_cameras_free() -> None:
    nodes = _realsense_video_nodes()
    users = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            opened = {os.readlink(fd) for fd in (entry / "fd").iterdir()}
        except OSError:
            continue
        if opened & nodes:
            users[int(entry.name)] = sorted(opened & nodes)
    if users:
        listing = "\n".join(
            f"  pid {pid} ({', '.join(devs)}): {_cmdline(pid)}"
            for pid, devs in users.items()
        )
        raise PreflightError(
            f"RealSense cameras are in use; stop these first:\n{listing}"
        )


def _check_cameras(profile) -> None:
    import pyrealsense2 as rs

    found = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        usb = "?"
        if device.supports(rs.camera_info.usb_type_descriptor):
            usb = device.get_info(rs.camera_info.usb_type_descriptor)
        found[serial] = usb
    problems = []
    for name, camera in profile.cameras.items():
        usb = found.get(camera.serial)
        if usb is None:
            problems.append(f"camera {name} ({camera.serial}) not found")
        elif not usb.startswith("3"):
            problems.append(
                f"camera {name} ({camera.serial}) is on USB {usb}; use a USB3 port/cable"
            )
    if problems:
        raise PreflightError(
            "; ".join(problems) + f". Enumerated: {found or 'none'}. Replug the camera."
        )


def _probe_policy_server(port: int, timeout: float = 3.0) -> bool:
    """Same handshake as WebsocketClientPolicy: connect, receive metadata."""
    import websockets.sync.client
    from websockets.exceptions import WebSocketException

    from deploy.client import msgpack_numpy

    try:
        with websockets.sync.client.connect(
            f"ws://localhost:{port}",
            compression=None,
            max_size=None,
            open_timeout=timeout,
        ) as conn:
            return isinstance(msgpack_numpy.unpackb(conn.recv(timeout=timeout)), dict)
    except (OSError, WebSocketException, ValueError, TypeError):
        return False


def _port_in_use(port: int) -> bool:
    """Bind test; a bare TCP connect makes the websocket server log a traceback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def _port_owner(port: int) -> int | None:
    result = subprocess.run(
        ["ss", "-ltnpH", f"sport = :{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"pid=(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def _gpu_state() -> tuple[float, list[tuple[int, int]]]:
    free = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    procs = []
    for line in apps.stdout.strip().splitlines():
        pid, used = (field.strip() for field in line.split(","))
        procs.append((int(pid), int(used)))
    return int(free.stdout.split()[0]) / 1024, procs


@dataclass
class Plan:
    reuse_pid: int | None
    """Pid of a compatible server already on the port, else None (start one)."""


def _check_server_args(args) -> None:
    clashes = [
        token
        for token in shlex.split(args.server_args)
        if token.split("=")[0].replace("_", "-") in MANAGED_SERVER_FLAGS
    ]
    if clashes:
        raise PreflightError(
            f"--server-args must not set {clashes}; use run_policy's own --policy, "
            "--checkpoint-path, --prompt, --diffusion-steps, --rtc-prefix-length, "
            "--fast-inference and --port so server and robot agree."
        )


def _policy_kind(args) -> str:
    from deploy.policy.selector import sniff_policy_kind

    checkpoint = Path(args.checkpoint_path)
    if not checkpoint.is_file():
        raise PreflightError(f"Checkpoint not found: {checkpoint.resolve()}")
    kind = sniff_policy_kind(args.checkpoint_path)
    if args.policy and args.policy != kind:
        raise PreflightError(f"--policy={args.policy} but {checkpoint} is a {kind} checkpoint")
    if kind == "vla" and args.server_module == "local_robot.serve_bf16":
        raise PreflightError("local_robot.serve_bf16 serves DiT only; use local_robot.serve_local")
    if kind == "vla" and args.fast_inference:
        raise PreflightError("--fast-inference is supported for DiT only")
    return kind


def _choose_rollout(args, free_gib: float, others: list, reuse_pid) -> None:
    """Fill the lead/chunk defaults; on a shared GPU run the VLA with int8 layers."""
    reason = "dit, ~110 ms/chunk" if args.kind == "dit" else "vla bf16, free GPU, ~116 ms/chunk"
    slow = False
    if args.kind == "vla":
        quantize = re.search(r"--offload\.quantize[= ](\w+)", args.server_args)
        shared = bool(others) or (reuse_pid is None and free_gib < VLA_BF16_FREE_GIB)
        if shared and reuse_pid is None and quantize is None:
            args.offload_args = ["--offload.quantize=all"]
            _log(
                f"GPU is shared ({free_gib:.1f} GiB free, {len(others)} other GPU "
                f"process(es); VLA bf16 wants {VLA_BF16_FREE_GIB} GiB free): "
                "switching the VLA to --offload.quantize=all (int8, ~165-190 ms/chunk)"
            )
        int8 = bool(args.offload_args) or (
            quantize is not None and quantize.group(1) != "none"
        )
        slow = int8 or shared
        if int8:
            reason = "vla int8, ~165-190 ms/chunk"
        elif shared:
            reason = "vla bf16, shared GPU"
            _log(
                "WARNING: VLA bf16 on a shared GPU streams most Gemma layers "
                "(~450 ms/chunk); expect 'inference is still pending' pauses"
            )
    lead, chunk = SLOW_ROLLOUT if slow else FAST_ROLLOUT
    if args.rtc_inference_lead_steps is None:
        args.rtc_inference_lead_steps = lead
    if args.execute_chunk_dim is None:
        args.execute_chunk_dim = chunk
    _log(
        f"Rollout: lead {args.rtc_inference_lead_steps}, chunk {args.execute_chunk_dim} "
        f"(defaults for {reason}: lead {lead}, chunk {chunk})"
    )


def _check_rollout(args) -> None:
    from abc_minimal.config import DiTConfig, VLAModelConfig

    model_chunk = (DiTConfig() if args.kind == "dit" else VLAModelConfig()).chunk_length
    chunk = args.execute_chunk_dim
    if args.no_rtc:
        if not 1 <= chunk <= model_chunk:
            raise PreflightError(
                f"--execute-chunk-dim={chunk} must be in 1..{model_chunk} (model chunk)"
            )
        return
    lead, prefix = args.rtc_inference_lead_steps, args.rtc_prefix_length
    if not 1 <= prefix <= lead < chunk:
        raise PreflightError(
            "RTC needs 1 <= --rtc-prefix-length <= --rtc-inference-lead-steps "
            f"< --execute-chunk-dim (got {prefix}, {lead}, {chunk})"
        )
    if prefix + chunk > model_chunk:
        raise PreflightError(
            f"--rtc-prefix-length + --execute-chunk-dim = {prefix + chunk} exceeds the "
            f"{args.kind} chunk length {model_chunk}"
        )


def preflight(args, profile) -> Plan:
    _check_other_sessions()
    _check_can(profile, args.debug)
    _check_cameras_free()
    _check_cameras(profile)
    _check_server_args(args)
    args.kind = _policy_kind(args)

    reuse_pid = None
    if _port_in_use(args.port):
        owner = _port_owner(args.port)
        owner_text = f"pid {owner}: {_cmdline(owner)}" if owner else "unknown process"
        if not _probe_policy_server(args.port):
            raise PreflightError(
                f"Port {args.port} is taken by a non-policy process ({owner_text}). "
                "Stop it or pass --port."
            )
        if args.no_reuse:
            raise PreflightError(
                f"A policy server already runs on port {args.port} ({owner_text}); "
                "drop --no-reuse to use it, or pass another --port."
            )
        reuse_pid = owner or -1
        _log(f"Reusing policy server on port {args.port} ({owner_text})")
        if owner and args.checkpoint_path not in _cmdline(owner):
            _log(f"WARNING: its command line does not mention {args.checkpoint_path}")
    if prompt_socket_path(args.port).exists() and reuse_pid is None:
        prompt_socket_path(args.port).unlink()

    free_gib, apps = _gpu_state()
    others = [
        (pid, used) for pid, used in apps if pid != reuse_pid and used >= GPU_HOG_MIB
    ]
    listing = "\n".join(
        f"  pid {pid} ({used} MiB): {_cmdline(pid)}" for pid, used in others
    )
    _choose_rollout(args, free_gib, others, reuse_pid)
    _check_rollout(args)
    min_free = args.min_free_vram_gib or MIN_FREE_VRAM_GIB[args.kind]
    if reuse_pid is None and free_gib < min_free:
        raise PreflightError(
            f"Only {free_gib:.1f} GiB VRAM free; the {args.kind} server needs "
            f"{min_free:.1f} GiB. GPU processes:\n{listing or '  (none)'}\n"
            "Stop them, or keep a running server on the port to reuse it."
        )
    if others:
        _log(f"WARNING: other GPU processes may slow inference:\n{listing}")
    return Plan(reuse_pid=reuse_pid)


# ---------------------------------------------------------------------------
# Policy server and prompt relay
# ---------------------------------------------------------------------------


def server_command(args) -> list[str]:
    command = [
        sys.executable,
        "-m",
        args.server_module,
        f"--policy.checkpoint-path={args.checkpoint_path}",
        f"--policy-type={args.kind}",
        f"--policy.diffusion-steps={args.diffusion_steps}",
        f"--policy.prompt={args.prompt}",
        f"--port={args.port}",
    ]
    if not args.no_rtc:
        command.append(f"--policy.rtc-prefix-length={args.rtc_prefix_length}")
    if args.fast_inference:
        command.append("--policy.fast-inference")
    return command + args.offload_args + shlex.split(args.server_args)


class PolicyServer:
    def __init__(self, args, log_path: Path):
        self.args = args
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._stdin_lock = threading.Lock()

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen(
                server_command(self.args),
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )

    def wait_ready(self) -> None:
        start = time.monotonic()
        next_note = start + 60.0
        while time.monotonic() - start < self.args.server_timeout:
            if time.monotonic() >= next_note:
                next_note += 60.0
                _log(
                    f"Server still starting ({time.monotonic() - start:.0f} s of "
                    f"{self.args.server_timeout:.0f} s)"
                )
            if self.proc.poll() is not None:
                raise PreflightError(
                    f"Policy server exited with code {self.proc.returncode}. "
                    f"Last log lines ({self.log_path}):\n{self.tail(20)}"
                )
            if _probe_policy_server(self.args.port):
                _log(f"Policy server ready in {time.monotonic() - start:.0f} s")
                for line in self.tail(10).splitlines():
                    if "warmup" in line or "startup" in line:
                        _log(line.strip())
                return
            time.sleep(1.0)
        raise PreflightError(
            f"Policy server not ready after {self.args.server_timeout:.0f} s; "
            f"see {self.log_path}"
        )

    def tail(self, lines: int) -> str:
        try:
            return "\n".join(
                self.log_path.read_text(errors="replace").splitlines()[-lines:]
            )
        except OSError:
            return ""

    def send_prompt(self, prompt: str) -> str:
        if self.proc is None or self.proc.poll() is not None:
            return "error: policy server is not running"
        offset = self.log_path.stat().st_size
        with self._stdin_lock:
            try:
                self.proc.stdin.write(prompt.encode() + b"\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                return "error: policy server stdin is closed"
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with open(self.log_path, "rb") as log:
                log.seek(offset)
                if PROMPT_CONFIRM_PATTERN.encode() in log.read():
                    return f"ok: server applies {prompt!r} from the next chunk"
            time.sleep(0.05)
        return f"sent {prompt!r} (no confirmation in {self.log_path})"

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        for sig, wait_s in (
            (signal.SIGINT, 10.0),
            (signal.SIGTERM, 5.0),
            (signal.SIGKILL, 5.0),
        ):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        _log(f"Policy server stopped (exit {self.proc.returncode})")


class PromptRelay:
    """Unix socket that forwards one prompt line per connection to the server."""

    def __init__(self, path: Path, server: PolicyServer):
        self.path = path
        self.server = server
        self.sock: socket.socket | None = None

    def start(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(self.path))
        self.sock.listen()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(5.0)
                try:
                    prompt = conn.makefile("r", encoding="utf-8").readline().strip()
                    if not prompt:
                        reply = "error: empty prompt"
                    else:
                        reply = self.server.send_prompt(prompt)
                        print(f"\n[run_policy] prompt -> {prompt!r}", flush=True)
                    conn.sendall(reply.encode() + b"\n")
                except OSError:
                    pass

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Robot side and cleanup
# ---------------------------------------------------------------------------


def deploy_command(args, deploy_extra: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "local_robot.deploy_policy",
        f"--checkpoint-path={args.checkpoint_path}",
        f"--policy-type={args.kind}",
        "--remote-host=localhost",
        f"--port={args.port}",
        f"--diffusion-steps={args.diffusion_steps}",
        f"--execute-chunk-dim={args.execute_chunk_dim}",
        f"--prompt={args.prompt}",
    ]
    if not args.no_rtc:
        command += [
            "--rtc",
            f"--rtc-prefix-length={args.rtc_prefix_length}",
            f"--rtc-inference-lead-steps={args.rtc_inference_lead_steps}",
        ]
    if args.compress_images:
        command.append("--compress-images")
    if "--record" not in deploy_extra:
        command.append("--no-record")
    return command + deploy_extra


def _become_subreaper() -> None:
    """Orphaned deploy nodes (they call setsid) are re-parented to us for cleanup."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _stop_strays(keep: set[int]) -> int:
    def strays():
        return [
            pid
            for pid, ppid, _ in _processes()
            if ppid == os.getpid() and pid not in keep
        ]

    found = strays()
    for sig in (signal.SIGINT, signal.SIGKILL):
        for pid in strays():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 3.0
        while strays() and time.monotonic() < deadline:
            time.sleep(0.1)
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return len(found)


def _banner(args, command, server, reuse_pid, log_path) -> None:
    if reuse_pid is not None:
        server_line = (
            f"reusing pid {reuse_pid} on :{args.port} (prompt: type in its terminal)"
        )
        prompt_line = "not relayed for a reused server; type in the server's terminal"
    else:
        server_line = f"{args.server_module} pid {server.proc.pid} on :{args.port}, log {log_path}"
        prompt_line = (
            'uv run python -m local_robot.set_prompt "new prompt"  (next chunk)'
        )
    rtc = (
        "off"
        if args.no_rtc
        else f"prefix {args.rtc_prefix_length}, lead {args.rtc_inference_lead_steps} steps "
        f"(~{args.rtc_inference_lead_steps * CONTROL_DT_MS:.0f} ms)"
    )
    reset = f"{args.reset_speed} rad/s max" if args.reset_speed > 0 else "upstream 2 s"
    mode = "DEBUG: arms are not driven" if args.debug else "LIVE: arms will move"
    compress = "on" if args.compress_images else "off"
    env = f"ROBOT_PROFILE={args.profile} LOCAL_RESET_SPEED={args.reset_speed}"
    extras = [*args.offload_args] + (["fast inference"] if args.fast_inference else [])
    lines = [
        f"profile     {args.profile}   [{mode}]",
        f"checkpoint  {args.checkpoint_path}  ({args.kind}"
        + "".join(f", {extra}" for extra in extras)
        + ")",
        f"prompt      {args.prompt!r}",
        f"server      {server_line}",
        f"rollout     RTC {rtc}, chunk {args.execute_chunk_dim}, "
        + f"compress {compress}, reset {reset}",
        "keys        a/b start | stop + home,  c/j shut down,  "
        + "ENTER confirms the first-action safety check,  Ctrl-C abort",
        f"new prompt  {prompt_line}",
        f"command     {env} {shlex.join(command)}",
    ]
    rule = "=" * 72
    print("\n".join([rule, *lines, rule]), flush=True)


def main(argv=None) -> int:
    args, deploy_extra = _parse_args(argv)
    os.chdir(REPO_ROOT)
    os.environ["ROBOT_PROFILE"] = args.profile
    import deploy.robot.config as robot_config

    if args.profile not in robot_config.PROFILES:
        _log(f"Unknown profile {args.profile!r}; see deploy/robot/local_profiles.py")
        return 2
    profile = robot_config.PROFILES[args.profile]

    try:
        plan = preflight(args, profile)
    except PreflightError as error:
        _log(f"PREFLIGHT FAILED: {error}")
        return 2
    _log("Preflight OK")
    command = deploy_command(args, deploy_extra)
    if args.check:
        if plan.reuse_pid is None:
            print(f"server:  {shlex.join(server_command(args))}")
        print(f"robot:   ROBOT_PROFILE={args.profile} {shlex.join(command)}")
        return 0

    stdin_tty = sys.stdin.isatty()
    saved_tty = termios.tcgetattr(sys.stdin) if stdin_tty else None
    log_path = Path(args.log_dir) / f"server-{time.strftime('%Y%m%d-%H%M%S')}.log"
    server = PolicyServer(args, log_path)
    relay = None
    robot = None
    _become_subreaper()

    def _forward(signum, _frame):
        # Terminal Ctrl-C already reaches deploy_policy (same process group).
        if signum != signal.SIGINT and robot is not None and robot.poll() is None:
            robot.send_signal(signal.SIGINT)
        elif robot is None:
            raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _forward)

    exit_code = 1
    try:
        if plan.reuse_pid is None:
            _log(f"Starting policy server: {shlex.join(server_command(args))}")
            _log(f"Server log: {log_path}")
            server.start()
            server.wait_ready()
            relay = PromptRelay(prompt_socket_path(args.port), server)
            relay.start()
        _banner(args, command, server, plan.reuse_pid, log_path)
        env = {
            **os.environ,
            "ROBOT_PROFILE": args.profile,
            "LOCAL_RESET_SPEED": str(args.reset_speed),
        }
        robot = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
        exit_code = robot.wait()
    except PreflightError as error:
        _log(f"FAILED: {error}")
        exit_code = 2
    except KeyboardInterrupt:
        _log("Interrupted")
        exit_code = 130
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if relay is not None:
            relay.close()
        server_pid = server.proc.pid if server.proc else None
        strays = _stop_strays({server_pid} if server_pid else set())
        server.stop()
        if saved_tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved_tty)
        if strays:
            _log(f"Stopped {strays} stray process(es)")
        _log(
            f"Done (robot exit {exit_code}). Server log: {log_path if server_pid else '-'}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
