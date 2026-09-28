"""Current-limited spring that yields to deliberate leader displacement.

This does not estimate gravity or distinguish hand force from another load.
It can support a resting pose only within the available motor current.
"""
import numpy as np


class CompliantHold:
    def __init__(self):self.reset()

    def reset(self):
        self.anchor=None
        self.previous=None
        self.position=None
        self.previous_time=None

    def current(self, raw, now, strength_percent, yield_degrees):
        raw=np.asarray(raw,dtype=float)
        if raw.shape!=(7,) or not np.isfinite(raw).all():raise ValueError('Invalid leader hold position')
        if strength_percent<=0:
            self.reset();return np.zeros(7)
        if self.previous is None or now-self.previous_time>.1:
            self.anchor=raw.copy();self.position=raw.copy();self.previous=raw.copy();self.previous_time=now
            return np.zeros(7)
        dt=now-self.previous_time
        if dt<=0:return np.zeros(7)
        delta=(raw-self.previous+np.pi)%(2*np.pi)-np.pi
        self.position+=delta
        velocity=delta/dt
        width=np.deg2rad(yield_degrees)
        # Beyond the spring travel, move its anchor with the hand. On release,
        # the new anchor remains; this never pulls back to the original pose.
        displacement=self.position-self.anchor
        self.anchor+=np.where(np.abs(displacement)>width,displacement-np.clip(displacement,-width,width),0.)
        limit=75.*strength_percent/100.
        force=limit/width*(self.anchor-self.position)-2.*velocity
        force=np.clip(force,-limit,limit)
        force[-1]=0.  # Gripper remains free to open/close.
        self.previous=raw.copy();self.previous_time=now
        return force
