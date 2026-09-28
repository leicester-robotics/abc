"""Headless dashboard lifecycle. Device/control loops never wait for the UI."""
from __future__ import annotations
import json
import queue
import signal
import socket
import threading
import time
from pathlib import Path
from .cameras import CameraWorker
from .control import ControlSupervisor
from .gello import GelloReader
from .samples import LatestSample
from .simulation import Simulation
from .yam import PassiveYamWorker, YamAdapter


def ensure_port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
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
    sim=Simulation(Path(config.model_path),Path(config.asset_dir) if config.asset_dir else None,home=config.simulation_home)
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
        from .runtime import SimulationRunner
        import mujoco
        runtime=SimulationRunner(sim,config,leaders,actions,supervisor.status,demo,config_path)
        workers.append(runtime)
        runtime.start()
        render_data=mujoco.MjData(sim.model)
        last_ui=0.;last_status=0.
        print(f'Dashboard: http://127.0.0.1:{config.port} (simulation only)',flush=True)
        print(f'Tunnel ON YOUR COMPUTER: ssh -N -o ExitOnForwardFailure=yes -L 18081:127.0.0.1:{config.port} tarik@100.95.170.113',flush=True)
        while not stop.is_set():
            now=time.monotonic()
            runtime.copy_render_state(render_data)
            scene.update_from_mjdata(render_data)
            if now-last_ui>=.1:
                leader_samples={s:b.read() for s,b in leaders.items()}
                snapshot=dict(control=supervisor.status(),leaders=leader_samples,
                    physical={s:b.read() for s,b in physical.items()},
                    cameras={s:b.read() for s,b in cameras.items()},
                    errors={s:b.error for s,b in leaders.items()},
                    camera_errors={s:b.error for s,b in cameras.items()},
                    leader_rates={s:b.rate_hz for s,b in leaders.items()},
                    camera_rates={s:b.rate_hz for s,b in cameras.items()},**runtime.snapshot())
                ui.update(snapshot);last_ui=now
                if status_path and now-last_status>=1:
                    status={'mode':snapshot['control']['mode'],'error':snapshot['control']['error'],
                        'port':config.port,'simulation_time':float(render_data.time),'demo':demo,
                        'simulation_hz':snapshot['scene_rate'],
                        'leaders':{s:{'age':max(0.,time.monotonic()-v.acquired_at) if v else None,'error':leaders[s].error,
                                      'rate_hz':leaders[s].rate_hz,'read_ms':v.raw.get('read_ms') if v else None,
                                      'read_mode':v.raw.get('read_mode') if v else None,
                                      'counts':v.counts.tolist() if v is not None else None} for s,v in leader_samples.items()},
                        'cameras':{s:{'age':max(0.,time.monotonic()-v.acquired_at) if v else None,'error':cameras[s].error,
                                      'shape':list(v.rgb.shape) if v else None,'rate_hz':cameras[s].rate_hz} for s,v in snapshot['cameras'].items()}}
                    status_path=Path(status_path);status_path.parent.mkdir(parents=True,exist_ok=True)
                    tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(status_path)
                    last_status=now
            stop.wait(max(0.,1/60-(time.monotonic()-now)))
    finally:
        try:shutdown_resources(supervisor,workers,server)
        finally:
            for signum,handler in old_signals.items():signal.signal(signum,handler)
    return 0
