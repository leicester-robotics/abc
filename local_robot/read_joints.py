"""Read YAM joint positions once per CAN bus without commanding motion.

Mirrors the one-shot state read at the start of i2rt's get_yam_robot: motors
are enabled to get a feedback frame, but no torque or position command is sent.
"""

import argparse

import numpy as np
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, ReceiveMode

ARM_MOTORS = [
    [0x01, "DM4340"],
    [0x02, "DM4340"],
    [0x03, "DM4340"],
    [0x04, "DM4310"],
    [0x05, "DM4310"],
    [0x06, "DM4310"],
]
GRIPPER_MOTOR = [0x07, "DM4310"]


def read_bus(channel: str, with_gripper: bool) -> None:
    motors = ARM_MOTORS + ([GRIPPER_MOTOR] if with_gripper else [])
    n = len(motors)
    chain = DMChainCanInterface(
        motors,
        [0.0] * n,
        [1] * n,
        channel,
        motor_chain_name=f"read_{channel}",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
    )
    try:
        states = chain.read_states()
    finally:
        chain.close()
    pos = np.array([s.pos for s in states])
    print(f"{channel}: {n} motors responded")
    print(f"  pos (rad): {np.array2string(pos, precision=4, suppress_small=True)}")
    print(
        f"  pos (deg): {np.array2string(np.rad2deg(pos), precision=1, suppress_small=True)}"
    )
    print(f"  error codes: {[s.error_code for s in states]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("channels", nargs="+")
    parser.add_argument("--no-gripper", action="store_true")
    args = parser.parse_args()
    for channel in args.channels:
        try:
            read_bus(channel, with_gripper=not args.no_gripper)
        except Exception as error:
            print(f"{channel}: FAILED ({type(error).__name__}: {error})")


if __name__ == "__main__":
    main()
