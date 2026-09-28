"""Headless dashboard lifecycle. Device/control loops never wait for the UI."""
from __future__ import annotations
import json
import queue
import signal
import socket
import threading
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .cameras import CameraWorker
from .config import save_config
from .control import ControlSupervisor
from .gello import GelloReader, calibration_offsets
from .samples import LatestSample
from .simulation import Simulation
from .yam import PassiveYamWorker, YamAdapter


def ensure_port_free(port):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',port))


def make_workers(config, leaders, cameras, physical, demo=False):
    if demo:return []
    return ([GelloReader(a,leaders[a.side]) for a in config.arms]
            +[CameraWorker(c,cameras[c.serial]) for c in config.cameras]
            +[PassiveYamWorker(a.side,physical[a.side]) for a in config.arms])


def shutdown_resources(supervisor, workers, server):
    errors=[]
    for resource in [supervisor,*workers,server]:
        try:
            if resource is server:resource.stop()
            else:resource.close()
        except Exception as error:errors.append(str(error))
    if errors:raise RuntimeError('; '.join(errors))


def run(config, demo=False, config_path=None, status_path=None, stop_event=None):
    import viser
    from mjviser import ViserMujocoScene
    from .ui import DashboardUI
    ensure_port_free(config.port)
    sim=Simulation(Path(config.model_path),Path(config.asset_dir) if config.asset_dir else None)
    leaders={a.side:LatestSample() for a in config.arms}
    cameras={c.serial:LatestSample() for c in config.cameras}
    physical={a.side:LatestSample() for a in config.arms}
    supervisor=ControlSupervisor(config.arms,leaders,YamAdapter)
    workers=make_workers(config,leaders,cameras,physical,demo)
    server=viser.ViserServer(host='127.0.0.1',port=config.port)
    if server.get_port()!=config.port:
        server.stop();raise OSError('Requested port occupied; refusing fallback port')
    stop=stop_event or threading.Event()
    old_signals={}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM,signal.SIGINT):
            old_signals[signum]=signal.signal(signum,lambda *_:stop.set())
    try:
        actions=queue.Queue()
        scene=ViserMujocoScene(server,sim.model,num_envs=1)
        scene.camera_tracking_enabled=False
        server.scene.remove_by_name('/fixed_bodies/world/table_plane')
        ui=DashboardUI(server,supervisor,config,actions,demo)
        for worker in workers:worker.start()
        supervisor.start()
        origins={};previous=time.monotonic();start=previous;last_ui=0.;last_status=0.
        sim_status='Absolute calibration required, or choose simulation-only motion preview.'
        print(f'Dashboard: http://127.0.0.1:{config.port} (simulation only)',flush=True)
        print(f'Tunnel from your computer: ssh -N -o ExitOnForwardFailure=yes -L {config.port}:127.0.0.1:{config.port} tarik@100.95.170.113',flush=True)
        while not stop.is_set():
            now=time.monotonic()
            while not actions.empty():
                action,value=actions.get_nowait()
                try:
                    if action=='reset':sim.reset();origins.clear()
                    elif action=='preview':
                        sim.reset();origins.clear()
                        for arm in config.arms:
                            sample=leaders[arm.side].read()
                            if sample is not None and not leaders[arm.side].error and 0<=now-sample.acquired_at<=.2:
                                origins[arm.side]=(sample.radians.copy(),sim.snapshot(arm.side).position.copy())
                        sim_status='Relative simulation preview: '+(', '.join(origins) or 'no fresh GELLO data')+'. Physical calibration unchanged.'
                    elif action in ('calibrate','verify'):
                        if supervisor.status()['mode']!='simulation':raise ValueError('Release physical robot before changing calibration or mapping')
                        side=value if action=='calibrate' else value[0]
                        arm=next(a for a in config.arms if a.side==side)
                        if action=='calibrate':
                            sample=leaders[side].read()
                            if sample is None or leaders[side].error or now-sample.acquired_at>.2:raise ValueError('Fresh GELLO sample required')
                            arm.offsets=calibration_offsets(sample.counts,arm.signs)
                            arm.mapping_verified=False
                            origins.pop(side,None)
                        else:arm.mapping_verified=bool(value[1])
                        if config_path:save_config(Path(config_path),config)
                        ui.setup_status.content=f'{side}: {action} saved. Verify directions in simulation before physical teleop.'
                except Exception as error:ui.setup_status.content=str(error)
            leader_samples={side:buf.read() for side,buf in leaders.items()}
            for arm in config.arms:
                sample=leader_samples[arm.side]
                if leaders[arm.side].error or sample is None or not 0<=now-sample.acquired_at<=.2:continue
                if arm.side in origins:
                    raw0,home=origins[arm.side]
                    target=home+np.asarray(arm.signs)*(sample.radians-raw0)
                    target[-1]=np.clip(.5+(sample.radians[-1]-raw0[-1])*arm.signs[-1]/np.diff(arm.gripper_range)[0],0,1)
                    sim.set_target(arm.side,target)
                else:sim.follow(arm.side,sample,now)
            if demo:
                for side in ('left','right'):
                    q=np.array([.25*np.sin(now-start),.8,1.,-.5,0.,0.,.5])
                    sim.set_target(side,q)
                sim_status='DEMO: synthetic sine-wave commands; no hardware connected.'
            sim.step(now-previous);previous=now
            scene.update_from_mjdata(sim.data)
            if now-last_ui>=1/15:
                snapshot=dict(control=supervisor.status(),leaders=leader_samples,
                    simulation={s:sim.snapshot(s) for s in ('left','right')},
                    physical={s:b.read() for s,b in physical.items()},
                    cameras={s:b.read() for s,b in cameras.items()},
                    errors={s:b.error for s,b in leaders.items()},
                    camera_errors={s:b.error for s,b in cameras.items()},sim_status=sim_status)
                ui.update(snapshot);last_ui=now
                if status_path and now-last_status>=1:
                    status={'mode':snapshot['control']['mode'],'error':snapshot['control']['error'],
                        'port':config.port,'simulation_time':float(sim.data.time),'demo':demo,
                        'leaders':{s:{'age':now-v.acquired_at if v else None,'error':leaders[s].error,
                                      'counts':v.counts.tolist() if v is not None else None} for s,v in leader_samples.items()},
                        'cameras':{s:{'age':now-v.acquired_at if v else None,'error':cameras[s].error,
                                      'shape':list(v.rgb.shape) if v else None} for s,v in snapshot['cameras'].items()}}
                    status_path=Path(status_path);status_path.parent.mkdir(parents=True,exist_ok=True)
                    tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(status_path)
                    last_status=now
            stop.wait(max(0.,1/30-(time.monotonic()-now)))
    finally:
        try:shutdown_resources(supervisor,workers,server)
        finally:
            for signum,handler in old_signals.items():signal.signal(signum,handler)
    return 0
