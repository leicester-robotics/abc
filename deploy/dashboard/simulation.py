"""CPU MuJoCo physics, with named actuator mappings for the bimanual scene."""
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .samples import ArmSample


class Simulation:
    def __init__(self, model_path: Path, asset_dir: Path | None = None):
        model_path=Path(model_path).resolve()
        root=ET.parse(model_path).getroot()
        compiler=root.find('compiler')
        meshdir=Path(asset_dir).resolve() if asset_dir else model_path.parent/compiler.get('meshdir','')
        if not meshdir.is_dir():
            raise FileNotFoundError(f'Missing model assets: {meshdir}. Set --asset-dir to the YAM assets directory.')
        compiler.set('meshdir',str(meshdir))
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model)
        self.actuator_ids={}
        self.qpos_ids={}
        self.dof_ids={}
        for side in ('left','right'):
            names=[f'{side}_joint{i}' for i in range(1,7)]+[f'{side}_gripper']
            ids=np.array([self.model.actuator(n).id for n in names])
            joints=self.model.actuator_trnid[ids,0]
            self.actuator_ids[side]=ids
            self.qpos_ids[side]=self.model.jnt_qposadr[joints]
            self.dof_ids[side]=self.model.jnt_dofadr[joints]
        self.targets={}
        self._remainder=0.
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model,self.data,self.model.key('home').id)
        mujoco.mj_forward(self.model,self.data)
        self._remainder=0.
        for side in self.actuator_ids:
            self.targets[side]=self.snapshot(side).position.copy()

    def set_target(self, side: str, target: np.ndarray):
        target=np.asarray(target,dtype=float)
        if target.shape!=(7,) or not np.isfinite(target).all():
            raise ValueError('Target must contain seven finite values')
        ids=self.actuator_ids[side]
        limits=self.model.actuator_ctrlrange[ids]
        normalized=target.copy()
        normalized[:6]=np.clip(target[:6],limits[:6,0],limits[:6,1])
        normalized[-1]=np.clip(target[-1],0,1)
        self.targets[side]=normalized
        control=normalized.copy()
        control[-1]=np.interp(normalized[-1],[0.,1.],limits[-1])
        self.data.ctrl[ids]=control

    def follow(self, side, sample, now):
        if sample is not None and sample.calibrated and sample.position is not None and 0<=now-sample.acquired_at<=.2:
            self.set_target(side,sample.position)

    def step(self, elapsed: float):
        self._remainder+=min(max(elapsed,0.),.05)
        dt=self.model.opt.timestep
        steps=int((self._remainder+1e-10)/dt)
        if steps:
            mujoco.mj_step(self.model,self.data,nstep=steps)
            self._remainder-=steps*dt

    def snapshot(self, side):
        ids=self.actuator_ids[side]
        position=self.data.qpos[self.qpos_ids[side]].copy()
        velocity=self.data.qvel[self.dof_ids[side]].copy()
        limits=self.model.actuator_ctrlrange[ids[-1]]
        width=limits[1]-limits[0]
        position[-1]=(position[-1]-limits[0])/width
        velocity[-1]/=width
        return ArmSample(time.monotonic(),position,velocity,raw={'simulation_time':self.data.time})
