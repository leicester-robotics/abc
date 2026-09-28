"""Close then open one arm's gripper and grab camera frames at each end.

Arm joints get zero torque (as in the repo's patched gripper calibration) and
the run aborts if any arm joint drifts. Only the gripper is driven, with the
same low stall torque the calibration uses.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd, ReceiveMode

MOTORS = [
    [0x01, "DM4340"],
    [0x02, "DM4340"],
    [0x03, "DM4340"],
    [0x04, "DM4310"],
    [0x05, "DM4310"],
    [0x06, "DM4310"],
    [0x07, "DM4310"],
]
GRIPPER = 6
TORQUE = 0.2
MAX_SECONDS = 2.0
ARM_DRIFT_LIMIT = 0.05


def start_cameras() -> dict[str, rs.pipeline]:
    pipelines = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        pipeline.start(config)
        pipelines[serial] = pipeline
    for _ in range(20):
        for pipeline in pipelines.values():
            pipeline.wait_for_frames()
    return pipelines


def grab(pipelines: dict[str, rs.pipeline], out: Path, tag: str) -> None:
    for serial, pipeline in pipelines.items():
        for _ in range(3):
            frames = pipeline.wait_for_frames()
        image = np.asanyarray(frames.get_color_frame().get_data())
        cv2.imwrite(str(out / f"{tag}_{serial}.jpg"), image)


def drive_gripper(
    chain: DMChainCanInterface, direction: int, arm0: np.ndarray
) -> float:
    torques = np.zeros(len(MOTORS))
    torques[GRIPPER] = direction * TORQUE
    start, last, stable = time.time(), None, 0
    while time.time() - start < MAX_SECONDS:
        chain.set_commands(torques=torques)
        time.sleep(0.05)
        pos = np.array([s.pos for s in chain.read_states()])
        if np.any(np.abs(pos[:GRIPPER] - arm0) > ARM_DRIFT_LIMIT):
            chain.set_commands(torques=np.zeros(len(MOTORS)))
            raise RuntimeError(f"arm joints drifted: {pos[:GRIPPER] - arm0}")
        if last is not None and abs(pos[GRIPPER] - last) < 0.01:
            stable += 1
            if stable >= 6:
                break
        else:
            stable = 0
        last = pos[GRIPPER]
    return float(pos[GRIPPER])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("channel")
    parser.add_argument("--out", type=Path, default=Path("local_robot/snaps"))
    args = parser.parse_args()
    out = args.out / args.channel
    out.mkdir(parents=True, exist_ok=True)

    pipelines = start_cameras()
    n = len(MOTORS)
    chain = DMChainCanInterface(
        MOTORS,
        [0.0] * n,
        [1] * n,
        args.channel,
        motor_chain_name=f"grip_{args.channel}",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
    )
    try:
        chain.commands = [MotorCmd(torque=0.0) for _ in MOTORS]
        chain.start_thread()
        pos0 = np.array([s.pos for s in chain.read_states()])
        print(f"{args.channel}: start gripper={pos0[GRIPPER]:.3f}")
        grab(pipelines, out, "0_start")
        closed = drive_gripper(chain, +1, pos0[:GRIPPER])
        grab(pipelines, out, "1_closed")
        print(f"{args.channel}: closed gripper={closed:.3f}")
        opened = drive_gripper(chain, -1, pos0[:GRIPPER])
        grab(pipelines, out, "2_open")
        print(f"{args.channel}: open gripper={opened:.3f}")
    finally:
        chain.set_commands(torques=np.zeros(n))
        time.sleep(0.1)
        chain.close()
        for pipeline in pipelines.values():
            pipeline.stop()
    print(f"frames saved to {out}")


if __name__ == "__main__":
    main()
