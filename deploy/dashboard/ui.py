"""Browser scene controls, live device panels, and per-client heartbeat."""
from __future__ import annotations
import html
import json
import time
import numpy as np
from .yam import PHYSICAL_BLOCKER


def heartbeat_html(button_id):
    # Viser renders button UUIDs as DOM IDs. An iframe executes the timer without
    # modifying Viser's installed frontend. Clicks use its existing client socket.
    script = f'''<script>
const id = {json.dumps(button_id)};
setInterval(() => {{
  if (parent.document.visibilityState !== 'visible') return;
  const button = parent.document.getElementById(id);
  if (button) button.click();
}}, 100);
</script>'''
    return f'<style>[id="{button_id}"]{{display:none!important}}</style><iframe title="connection heartbeat" style="display:none" srcdoc="{html.escape(script, quote=True)}"></iframe>'


def format_arm(label, sample, now):
    if sample is None:
        return f'### {label}\nNo samples available.'
    age = max(0., now-sample.acquired_at)
    text = f'### {label}\n{"STALE" if age > .2 else "LIVE"} · age {age*1000:.0f} ms\n\n'
    if sample.position is not None:
        text += '| Joint | Position (rad) | Velocity (rad/s) | Effort (Nm) |\n|---|---:|---:|---:|\n'
        for i in range(7):
            name = f'J{i+1}' if i<6 else 'Gripper (normalized)'
            velocity = '—' if sample.velocity is None else f'{sample.velocity[i]:.3f}'
            effort = '—' if sample.effort is None else f'{sample.effort[i]:.3f}'
            text += f'| {name} | {sample.position[i]:.4f} | {velocity} | {effort} |\n'
    if getattr(sample, 'counts', None) is not None:
        text += '\n**Raw encoder counts:** `' + ', '.join(str(int(x)) for x in sample.counts) + '`\n'
        text += '\n**Encoder radians:** `' + ', '.join(f'{x:.3f}' for x in sample.radians) + '`\n'
        if not sample.calibrated:
            text += '\nAbsolute joint mapping is uncalibrated.\n'
    if sample.raw:
        text += '\n```json\n'+json.dumps(sample.raw,default=lambda x: np.asarray(x).tolist(),indent=2)+'\n```\n'
    return text


class DashboardUI:
    def __init__(self, server, supervisor, config, action_queue, demo=False):
        self.server=server
        self.supervisor=supervisor
        self.config=config
        self.actions=action_queue
        self.demo=demo
        self.clients={}
        self.latest={}
        server.gui.configure_theme(control_layout='floating', control_width='large', dark_mode=True,
                                   show_logo=False, show_share_button=False, brand_color=(41,181,163))
        server.gui.add_markdown('# GELLO · YAM\n**Headless teleoperation station**')
        self.status=server.gui.add_markdown('Starting simulation…')
        self.control_buttons={}
        if PHYSICAL_BLOCKER:
            server.gui.add_markdown('**Physical control unavailable:** '+PHYSICAL_BLOCKER)
        with server.gui.add_folder('Robot control', expand_by_default=True):
            server.gui.add_markdown('Simulation starts without energizing YAM. **Connect** can energize motors and calibrate the gripper. Return to simulation keeps the robot holding position.')
            for action,label in [('connect','Connect robot'),('enable','Enable robot teleop'),
                                  ('simulation','Return to simulation'),('recover','Clear fault'),
                                  ('release','Release motors / disconnect (support arms first)')]:
                button=server.gui.add_button(label, color='red' if action=='release' else None)
                button.on_click(lambda event, a=action: supervisor.request(a,event.client_id))
                self.control_buttons[action]=button
        with server.gui.add_folder('Simulation', expand_by_default=True):
            button=server.gui.add_button('Preview GELLO motion from current pose')
            button.on_click(lambda _: self.actions.put(('preview',None)))
            server.gui.add_markdown('Preview maps your current GELLO pose to the simulated home pose. **Simulation only**; it does not calibrate the physical robot.')
            server.gui.add_button('Reset simulation').on_click(lambda _: self.actions.put(('reset',None)))
            server.gui.add_button('Reset view').on_click(lambda _: self.reset_view())
            self.sim_status=server.gui.add_markdown('Awaiting GELLO mapping.')
        self.images={};self.depth_images={};self.camera_status={};self.camera_times={}
        with server.gui.add_folder('RealSense live streams',expand_by_default=True):
            for cam in config.cameras:
                with server.gui.add_folder(f'{cam.label} · {cam.serial}',expand_by_default=True):
                    self.camera_status[cam.serial]=server.gui.add_markdown('Connecting…')
                    self.images[cam.serial]=server.gui.add_image(np.zeros((240,320,3),dtype=np.uint8),label='RGB',format='jpeg',jpeg_quality=75)
                    if cam.depth:
                        self.depth_images[cam.serial]=server.gui.add_image(np.zeros((240,320,3),dtype=np.uint8),label='Depth',format='jpeg',jpeg_quality=70)
        self.leader_text={};self.sim_text={};self.robot_text={}
        with server.gui.add_folder('Joint data',expand_by_default=False):
            for side in ('left','right'):
                with server.gui.add_folder(side.capitalize(),expand_by_default=True):
                    self.leader_text[side]=server.gui.add_markdown('GELLO: awaiting samples')
                    self.sim_text[side]=server.gui.add_markdown('MuJoCo: initializing')
                    self.robot_text[side]=server.gui.add_markdown('Physical YAM: unavailable until observation source or explicit connection')
        with server.gui.add_folder('Calibration and station mapping',expand_by_default=False):
            server.gui.add_markdown('Use raw readings and simulation to verify joint directions. For absolute calibration, place all six joints at the documented GELLO zero pose and the gripper at its reference pose (0.357 rad). Capturing an arbitrary pose here gives an incorrect robot mapping.')
            for arm in config.arms:
                with server.gui.add_folder(f'{arm.side} · {arm.device.split("_")[-1]}',expand_by_default=True):
                    server.gui.add_markdown(f'IDs {arm.servo_ids} · {arm.baudrate/1e6:g} Mbps\n\nSigns {arm.signs} · CAN `{arm.channel}` · gripper `{arm.gripper_type}`')
                    server.gui.add_button(f'Capture {arm.side} absolute zero pose').on_click(lambda _,s=arm.side:self.actions.put(('calibrate',s)))
                    verify=server.gui.add_checkbox(f'{arm.side}: side, signs, gripper and CAN mapping verified',initial_value=arm.mapping_verified)
                    verify.on_update(lambda event,s=arm.side:self.actions.put(('verify',(s,event.target.value))))
            self.setup_status=server.gui.add_markdown('Physical teleop requires verified mapping and absolute calibration for each configured arm.')
        @server.on_client_connect
        def connected(client):
            client.camera.position=(-.8,0,1.75)
            client.camera.look_at=(.45,0,.95)
            client.camera.up_direction=(0,0,1)
            beat=client.gui.add_button('Connection heartbeat',order=9999)
            @beat.on_click
            def receive(event):
                if event.client_id==client.client_id:
                    supervisor.heartbeat(str(client.client_id),time.monotonic())
            client.gui.add_html(heartbeat_html(beat._impl.uuid),order=10000)
            self.clients[client.client_id]=client
        @server.on_client_disconnect
        def disconnected(client):
            self.clients.pop(client.client_id,None)
            supervisor.disconnect(str(client.client_id))

    def reset_view(self):
        for client in self.server.get_clients().values():
            client.camera.position=(-.8,0,1.75)
            client.camera.look_at=(.45,0,.95)

    def update(self,snapshot):
        now=time.monotonic();self.latest=snapshot
        state=snapshot['control'];mode=state['mode']
        title={'simulation':'SIMULATION ONLY','holding':'SIMULATION / ROBOT HOLDING','teleop':'PHYSICAL TELEOP ENABLED','fault':'FAULT / TELEOP DISABLED','connecting':'CONNECTING ROBOT'}[mode]
        self.status.content=f'## {title}\n'+('**DEMO — synthetic simulation motion**\n\n' if self.demo else '')+f'Controller: {state["owner"] or "none"}\n\n'+html.escape(state['error'])
        ready=not PHYSICAL_BLOCKER and bool(self.config.arms) and all(a.calibrated and a.mapping_verified for a in self.config.arms)
        self.control_buttons['connect'].disabled=self.demo or not ready or mode!='simulation'
        self.control_buttons['enable'].disabled=self.demo or not ready or mode!='holding'
        self.control_buttons['release'].disabled=mode=='simulation'
        self.control_buttons['recover'].disabled=mode!='fault'
        self.sim_status.content=snapshot['sim_status']
        for side in self.leader_text:
            leader=snapshot['leaders'].get(side)
            self.leader_text[side].content=format_arm('GELLO · mapped + raw',leader,now)+'\n'+html.escape(snapshot['errors'].get(side,''))
            self.sim_text[side].content=format_arm('MuJoCo · simulated',snapshot['simulation'][side],now)
            actual=state['observations'].get(side) or snapshot['physical'].get(side)
            self.robot_text[side].content=format_arm('Physical YAM · measured',actual,now)
        for serial,handle in self.images.items():
            sample=snapshot['cameras'][serial]
            error=snapshot['camera_errors'].get(serial,'')
            if sample is None:
                self.camera_status[serial].content=html.escape(error or 'Waiting for camera')
                continue
            age=now-sample.acquired_at
            self.camera_status[serial].content=f'{"STALE" if age>.5 else "LIVE"} · {age*1000:.0f} ms · {sample.depth_status}\n'+html.escape(error)
            if sample.acquired_at!=self.camera_times.get(serial):
                handle.image=sample.rgb
                if sample.depth is not None and serial in self.depth_images:self.depth_images[serial].image=sample.depth
                self.camera_times[serial]=sample.acquired_at
