"""Validated live teleop settings, in operator-facing units."""
from dataclasses import asdict, dataclass, replace
import math
import numpy as np

# key: (label, minimum, maximum, step)
FIELDS = {
    'leader_margin_deg': ('GELLO limit margin (degrees)', .5, 10., .5),
    'alignment_deg': ('Arm alignment tolerance (degrees)', .5, 10., .1),
    'gripper_alignment_percent': ('Gripper alignment tolerance (%)', 1., 15., 1.),
    'heartbeat_seconds': ('Browser heartbeat timeout (seconds)', .5, 2., .1),
    'teleop_speed': ('Teleop joint speed (rad/s)', .1, 2., .1),
    'alignment_speed': ('Alignment joint speed (rad/s)', .05, .5, .05),
    'feedback_percent': ('Force feedback strength (%)', 0., 200., 5.),
    'hold_percent': ('GELLO hold strength (%)', 0., 100., 5.),
    'hold_yield_deg': ('GELLO hold yield distance (degrees)', 1., 15., .5),
}

@dataclass(frozen=True)
class TeleopSettings:
    leader_margin_deg: float = 5.
    alignment_deg: float = float(np.rad2deg(.03))
    gripper_alignment_percent: float = 3.
    heartbeat_seconds: float = 1.
    teleop_speed: float = .5
    alignment_speed: float = .25
    feedback_percent: float = 100.
    hold_percent: float = 100.
    hold_yield_deg: float = 6.

    def updated(self, values):
        if not isinstance(values,dict) or set(values)-set(FIELDS):
            raise ValueError('Unknown settings field')
        cleaned={}
        for name,value in values.items():
            label,low,high,_=FIELDS[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{label} must be between {low:g} and {high:g}')
            cleaned[name]=float(value)
        return replace(self,**cleaned)

    def alignment_tolerance(self):
        return np.array([np.deg2rad(self.alignment_deg)]*6+[self.gripper_alignment_percent/100.])

    def as_dict(self):return asdict(self)
