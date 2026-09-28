"""Save one RGB frame from every connected RealSense camera."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def snap(serial: str, out: Path, warmup: int = 20) -> Path:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        for _ in range(warmup):
            frames = pipeline.wait_for_frames()
        image = np.asanyarray(frames.get_color_frame().get_data())
    finally:
        pipeline.stop()
    path = out / f"{serial}.jpg"
    cv2.imwrite(str(path), image)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("local_robot/snaps"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    out = args.out / args.tag if args.tag else args.out
    out.mkdir(parents=True, exist_ok=True)
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        print(snap(serial, out))


if __name__ == "__main__":
    main()
