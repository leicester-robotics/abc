"""Protocol 2 GELLO encoder reader; never enables torque or commands motion.

Control table references: ROBOTIS XC330-T288 / XL330-M288 e-manual.
Only identified, explicitly supported models may be read or written.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager

import numpy as np

# Shared X-series table: ID=7, torque=64, present position=132 (4096 ticks/turn).
MODELS = {
    1190: "XL330-M077",
    1200: "XL330-M288",
    1210: "XC330-M181",
    1220: "XC330-M288",
    1230: "XC330-T181",
    1240: "XC330-T288",
    1030: "XM430-W210",
    1020: "XM430-W350",
}
TICKS_PER_TURN = 4096


class DynamixelBus(AbstractContextManager):
    def __init__(self, port: str, baudrate: int = 57600):
        from dynamixel_sdk import PacketHandler, PortHandler

        self.port_name = port
        self.baudrate = baudrate
        self.port = PortHandler(port)
        self.packet = PacketHandler(2.0)
        try:
            if not self.port.openPort() or not self.port.setBaudRate(baudrate):
                raise OSError(f"Cannot open {port} at {baudrate}")
            # Fail if another process owns this port; never kill other processes.
            if hasattr(self.port, "ser"):
                self.port.ser.exclusive = True
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.port.ser is not None:
            self.port.closePort()

    def __exit__(self, *exc):
        self.close()

    def _check(self, result, error):
        if result != 0 or error != 0:
            raise OSError(
                f"{self.port_name}: {self.packet.getTxRxResult(result)}; "
                f"{self.packet.getRxPacketError(error)}"
            )

    def ping(self, motor_id):
        model, result, error = self.packet.ping(self.port, motor_id)
        self._check(result, error)
        return model

    def scan(self):
        """Read-only broadcast discovery. Duplicate IDs cannot be reliably detected."""
        devices, result = self.packet.broadcastPing(self.port)
        # The SDK returns COMM_RX_TIMEOUT when no motors answer.
        if result == -3001:
            return {}
        self._check(result, 0)
        return {int(i): int(info[0]) for i, info in devices.items()}

    def verify(self, ids):
        if (
            len(ids) != 7
            or len(set(ids)) != 7
            or any(type(i) is not int or not 0 <= i <= 252 for i in ids)
        ):
            raise ValueError("Expected seven unique IDs, ordered base to wrist, then gripper")
        models = [self.ping(i) for i in ids]
        unknown = set(models) - MODELS.keys()
        if unknown:
            raise ValueError(f"Unsupported model numbers {unknown}; verify the control table first")
        return models

    def read_torque(self, motor_id):
        value, result, error = self.packet.read1ByteTxRx(self.port, motor_id, 64)
        self._check(result, error)
        return value

    def relax(self, ids):
        self.verify(ids)
        for motor_id in ids:
            result, error = self.packet.write1ByteTxRx(self.port, motor_id, 64, 0)
            self._check(result, error)
            if self.read_torque(motor_id) != 0:
                raise OSError(f"Torque remains enabled on motor {motor_id}")

    def read(self, ids):
        """Fresh complete frame only; no cached values on partial communication failure."""
        from dynamixel_sdk import GroupSyncRead

        started = time.monotonic()
        bus = self

        class CheckedGroupRead(GroupSyncRead):
            def rxPacket(self):
                # SDK GroupSyncRead discards motor error bytes; preserve failure semantics.
                self.last_result = False
                for motor_id in self.data_dict:
                    data, result, error = self.ph.readRx(self.port, motor_id, self.data_length)
                    try:
                        bus._check(result, error)
                    except OSError as exc:
                        raise OSError(f"Motor {motor_id} position read failed: {exc}") from exc
                    self.data_dict[motor_id] = data
                self.last_result = True
                return 0

        group = CheckedGroupRead(self.port, self.packet, 132, 4)
        try:
            for motor_id in ids:
                if not group.addParam(motor_id):
                    raise ValueError(f"Duplicate/invalid motor ID {motor_id}")
            self._check(group.txRxPacket(), 0)
            ticks = []
            for motor_id in ids:
                if not group.isAvailable(motor_id, 132, 4):
                    raise OSError(f"Missing position from motor {motor_id}")
                raw = group.getData(motor_id, 132, 4)
                ticks.append(raw if raw < 2**31 else raw - 2**32)
            return np.asarray(ticks, dtype=float) * (2 * np.pi / TICKS_PER_TURN), started
        finally:
            group.clearParam()

    def assign_id(self, old_id, new_id, expected_model, *, isolated=False):
        """EEPROM write for ONE physically isolated motor; never a whole duplicate-ID chain."""
        if not isolated:
            raise ValueError("Physically isolate one powered motor before assigning its ID")
        if any(type(i) is not int or not 0 <= i <= 252 for i in (old_id, new_id)):
            raise ValueError("IDs must be integers in 0..252")
        if expected_model not in MODELS:
            raise ValueError("Unsupported model; refusing a control-table write")
        found = self.scan()
        if found != {old_id: expected_model} or self.ping(old_id) != expected_model:
            raise ValueError(f"Expected only ID {old_id}, model {expected_model}; found {found}")
        if self.read_torque(old_id):
            raise ValueError("Motor torque must already be disabled before assigning an ID")
        if old_id == new_id:
            return
        result, error = self.packet.write1ByteTxRx(self.port, old_id, 7, new_id)
        self._check(result, error)
        time.sleep(0.1)
        if self.ping(new_id) != expected_model or self.scan() != {new_id: expected_model}:
            raise OSError("ID write could not be verified; rescan before doing anything else")
