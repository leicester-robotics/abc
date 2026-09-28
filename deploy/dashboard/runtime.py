"""Simulation follows the newest GELLO samples independently of browser/image work."""
from __future__ import annotations
import queue
import threading
import time
from pathlib import Path
import mujoco
import numpy as np
from .config import save_config
from .gello import calibration_offsets


class SimulationRunner:
    rate = 100.

    def __init__(self, sim, config, leaders, actions, control_status, demo=False, config_path=None):
        self.sim=sim; self.config=config; self.leaders=leaders; self.actions=actions
        self.control_status=control_status; self.demo=demo; self.config_path=config_path
        self.lock=threading.RLock();self.stop=threading.Event();self.thread=None
        self.origins={};self.previous=time.monotonic();self.started=self.previous
        self.rate_hz=0.
        self.sim_status='Choose motion preview, or calibrate the absolute zero pose.'
        self.setup_status='Absolute calibration and verified mapping are required for physical control.'

    def _actions(self, now):
        while not self.actions.empty():
            action,value=self.actions.get_nowait()
            try:
                if action=='reset':
                    self.sim.reset();self.origins.clear()
                    self.sim_status='Simulation reset; capture a new preview pose.'
                elif action=='preview':
                    self.sim.reset();self.origins.clear()
                    for arm in self.config.arms:
                        sample=self.leaders[arm.side].read()
                        if sample is not None and not self.leaders[arm.side].error and 0<=now-sample.acquired_at<=.2:
                            self.origins[arm.side]=(sample.radians.copy(),self.sim.snapshot(arm.side).position.copy())
                    self.sim_status='Relative preview: '+(', '.join(self.origins) or 'no fresh GELLO data')+'. Simulation only.'
                elif action in ('calibrate','verify'):
                    if self.control_status()['mode']!='simulation':
                        raise ValueError('Release physical robot before changing calibration or mapping')
                    side=value if action=='calibrate' else value[0]
                    arm=next(a for a in self.config.arms if a.side==side)
                    if action=='calibrate':
                        sample=self.leaders[side].read()
                        if sample is None or self.leaders[side].error or not 0<=now-sample.acquired_at<=.2:
                            raise ValueError('Fresh GELLO sample required')
                        arm.offsets=calibration_offsets(sample.counts,arm.signs)
                        arm.mapping_verified=False;self.origins.pop(side,None)
                    else:arm.mapping_verified=bool(value[1])
                    if self.config_path:save_config(Path(self.config_path),self.config)
                    self.setup_status=f'{side}: {action} saved. Verify directions in simulation.'
            except Exception as error:self.setup_status=str(error)

    def tick(self, now):
        with self.lock:
            self._actions(now)
            for arm in self.config.arms:
                sample=self.leaders[arm.side].read()
                if self.leaders[arm.side].error or sample is None or not 0<=now-sample.acquired_at<=.2:continue
                if arm.side in self.origins:
                    raw0,home=self.origins[arm.side]
                    target=home+np.asarray(arm.signs)*(sample.radians-raw0)
                    target[-1]=np.clip(.5+(sample.radians[-1]-raw0[-1])*arm.signs[-1]/np.diff(arm.gripper_range)[0],0,1)
                    self.sim.set_target(arm.side,target)
                else:self.sim.follow(arm.side,sample,now)
            if self.demo:
                for side in ('left','right'):
                    self.sim.set_target(side,np.array([.25*np.sin(now-self.started),.8,1.,-.5,0.,0.,.5]))
                self.sim_status='DEMO: synthetic sine-wave commands; no hardware connected.'
            elapsed=max(now-self.previous,0.)
            self.sim.step(elapsed);self.previous=now
            if elapsed>0:self.rate_hz=1./elapsed if self.rate_hz==0 else .1/elapsed+.9*self.rate_hz

    def snapshot(self):
        with self.lock:
            return {'simulation':{s:self.sim.snapshot(s) for s in ('left','right')},
                    'targets':{s:v.copy() for s,v in self.sim.targets.items()},
                    'sim_status':self.sim_status,'setup_status':self.setup_status,'scene_rate':self.rate_hz}

    def copy_render_state(self, data):
        with self.lock:mujoco.mj_copyData(data,self.sim.model,self.sim.data)

    def start(self):
        def loop():
            while not self.stop.is_set():
                now=time.monotonic()
                try:self.tick(now)
                except Exception as error:
                    self.setup_status=f'Simulation update failed: {error}'
                self.stop.wait(max(0.,1/self.rate-(time.monotonic()-now)))
        self.thread=threading.Thread(target=loop,name='mujoco-simulation',daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3.)
            if self.thread.is_alive():raise RuntimeError('Simulation worker did not stop')
