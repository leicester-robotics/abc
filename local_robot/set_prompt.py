"""Change the live prompt of a policy server started by local_robot.run_policy.

  uv run python -m local_robot.set_prompt "pick up the cube"
  uv run python -m local_robot.set_prompt        # one prompt per line until EOF

Applies from the next action chunk. Recordings keep the launch --prompt.
"""

import argparse
import socket
import sys

from local_robot.run_policy import prompt_socket_path


def send(prompt: str, port: int) -> str:
    path = prompt_socket_path(port)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(10.0)
        try:
            sock.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            raise SystemExit(
                f"No run_policy launcher owns a server on port {port} ({path}). "
                "For a server started on its own, type the prompt in its terminal."
            )
        sock.sendall(prompt.encode() + b"\n")
        return sock.makefile("r", encoding="utf-8").readline().strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.prompt:
        reply = send(" ".join(args.prompt), args.port)
        print(reply)
        return 0 if reply.startswith(("ok", "sent")) else 1
    for line in sys.stdin:
        if line.strip():
            print(send(line.strip(), args.port), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
