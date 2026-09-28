"""Single-owner teleop supervisor; GUI callbacks only enqueue requests."""
from __future__ import annotations
import queue
import threading
import time
import numpy as np
from .settings import TeleopSettings


LOW = np.array([-2.617, 0., 0., -1.57, -1.57, -2.09, 0.])
HIGH = np.array([3.13, 3.65, 3.13, 1.57, 1.57, 2.09, 1.])


def checked(sample, now, observation=False, leader_margin_deg=5.):
    if sample is None or not np.isfinite(sample.acquired_at) or not 0 <= now-sample.acquired_at <= .2:
        raise ValueError('Feedback missing or older than 200 ms')
    q = np.asarray(sample.position, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('Expected seven finite joint values')
    # Allow 5 degrees of leader overtravel and 0.03 rad physical stop error.
    # Commands remain clipped to the robot limits.
    tolerance = np.array([.03 if observation else np.deg2rad(leader_margin_deg)]*6+[0.])
    if np.any(q < LOW-tolerance) or np.any(q > HIGH+tolerance):
        bad=np.flatnonzero((q < LOW-tolerance) | (q > HIGH+tolerance))
        details=', '.join(f'{"gripper" if i==6 else "joint "+str(i+1)}={q[i]:.5f} (range {LOW[i]:.3f}..{HIGH[i]:.3f}, tolerance {tolerance[i]:.3f})' for i in bad)
        raise ValueError('Joint target or observation exceeds limits: '+details)
    return np.clip(q, LOW, HIGH)


BROWSER_LEASE_SECONDS = 1.0


class ControlSupervisor:
    def __init__(self, arms, leaders, adapter_factory, clock=time.monotonic, require_support=False, home_on_connect=False, haptics=None, home_pose=None):
        self.settings = TeleopSettings()
        self.settings_error = ''
        self.settings_revision = 0
        self.haptics = haptics or {}
        self.force_feedback = True
        self.home_pose = home_pose
        self._leader_at_alignment = {}
        self.home_on_connect = home_on_connect
        self._alignment_goals = {}
        self._alignment_then = 'teleop'
        self.require_support = require_support
        self._support_confirmed = {}
        self.arms = arms
        self.leaders = leaders
        self.adapter_factory = adapter_factory
        self.clock = clock
        self.mode = 'simulation'
        self.owner = None
        self.error = ''
        self.last_fault = None
        self.adapters = {}
        self.observations = {}
        self.targets = {}
        self.alignment = {}
        self._requests = queue.Queue(maxsize=32)
        self._teleop_stop = threading.Event()
        self._release_uncertain = False
        self._release_error = ''
        self._heartbeats = {}
        self._heartbeat_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._published = {}
        self._generation = 0
        self._previous = clock()
        self._stop = threading.Event()
        self._thread = None
        self._publish()

    def request(self, action, client_id):
        if action == 'simulation' or (action == 'disconnect' and str(client_id) == self.owner):
            self._teleop_stop.set()
            return
        try:
            self._requests.put_nowait((action, str(client_id), self._generation))
        except queue.Full:
            pass  # Stop requests use a separate latched event and cannot be dropped.

    def request_settings(self, values, client_id):
        self.request(('settings',dict(values)),client_id)

    def heartbeat(self, client_id, now):
        with self._heartbeat_lock:
            self._heartbeats[str(client_id)] = now

    def disconnect(self, client_id):
        with self._heartbeat_lock:
            self._heartbeats.pop(str(client_id), None)
        self.request('disconnect', client_id)

    def _lease_valid(self, client, now):
        with self._heartbeat_lock:
            age = now-self._heartbeats.get(client, -float('inf'))
        return 0 <= age <= self.settings.heartbeat_seconds

    def _leader(self, arm, now):
        if not arm.mapping_verified or not arm.calibrated:
            raise ValueError(f'{arm.side}: verify side, signs, and calibration first')
        sample = self.leaders[arm.side].read()
        if self.leaders[arm.side].error and not self.leaders[arm.side].transient_error:
            raise ValueError(f'{arm.side}: {self.leaders[arm.side].error}')
        if sample is None or not sample.calibrated:
            raise ValueError(f'{arm.side}: GELLO unavailable or uncalibrated')
        try:
            return checked(sample, now, leader_margin_deg=self.settings.leader_margin_deg)
        except ValueError as error:
            raise ValueError(f'{arm.side} GELLO: {error}') from error

    def _remember_fault(self, reason, stage=None):
        if self.mode=='fault' and self.last_fault is not None:return
        from datetime import datetime
        self.last_fault={'reason':reason,'time':datetime.now().astimezone().isoformat(timespec='seconds'),
                         'stage':stage or self.mode}

    def _hold(self, now, fault=False, reason=''):
        if fault:self._remember_fault(reason)
        self.owner = None
        self._generation += 1
        self.mode = 'fault' if fault else 'holding' if self.adapters else 'simulation'
        self.error = reason
        if self._release_uncertain:
            self.mode = 'fault'
            self.error = self._release_error or 'Robot release unconfirmed; retry release before recovery'
            return
        for side, adapter in self.adapters.items():
            try:
                sample = adapter.observe()
                try:
                    target = checked(sample, now, observation=True).copy()
                except ValueError:
                    target = self.targets.get(side)
                    self.mode = 'fault'
                    self.error = 'Feedback stale; holding last validated target'
                if target is not None:
                    adapter.hold(target)
                    self.targets[side] = target.copy()
            except Exception as error:
                self.mode = 'fault'
                self.error = f'Stop cannot be confirmed: {error}'

    def _action(self, action, client, now):
        if isinstance(action,tuple) and action[0]=='settings':
            try:
                if self.owner is not None and client!=self.owner:
                    raise ValueError('Only the controlling browser can change teleop settings')
                self.settings=self.settings.updated(action[1])
                self.settings_revision+=1
                self.settings_error=''
            except ValueError as error:self.settings_error=str(error)
            return
        if action in ('haptics_on','haptics_off'):
            if self.owner is not None and client != self.owner:
                raise ValueError('Only the controlling browser can change force feedback')
            self.force_feedback = action == 'haptics_on'
            return
        if action == 'confirm_support':
            self._support_confirmed[client] = now
            return
        if action == 'cancel_support':
            self._support_confirmed.pop(client, None)
            return
        if action in ('simulation', 'disconnect'):
            if action == 'simulation' or client == self.owner:
                self._hold(now)
            return
        if action == 'release':
            if self.owner is not None and client != self.owner:
                raise ValueError('Only the controlling browser can release motors')
            self._hold(now)
            self._close_adapters()
            self.mode = 'simulation'
            self.error = ''
            return
        if action == 'recover':
            if self._release_uncertain:
                raise ValueError('Robot release unconfirmed; retry release')
            if self.mode != 'fault':
                return
            for adapter in self.adapters.values():
                checked(adapter.observe(), now, observation=True)
            self._hold(now)
            return
        if action == 'connect':
            if self.require_support and not 0 <= now-self._support_confirmed.pop(client, -float('inf')) <= 30:
                raise ValueError('Confirm arms supported and grippers clear immediately before connecting')
            if self.adapters or self.mode != 'simulation':
                raise ValueError('Release the current connection before connecting again')
            if not self.arms:
                raise ValueError('No arms configured')
            for arm in self.arms:
                self._leader(arm, now)
                if not arm.channel:
                    raise ValueError('CAN channel is not configured')
            self.mode = 'connecting'
            self._publish()
            try:
                for arm in self.arms:
                    adapter = self.adapter_factory(arm)
                    self.adapters[arm.side] = adapter
                    adapter.connect()
                    try:
                        q = checked(adapter.observe(), self.clock(), observation=True)
                    except Exception as error:
                        raise ValueError(f'{arm.side} YAM startup: {error}') from error
                    adapter.hold(q)
                    self.targets[arm.side] = q.copy()
                self.mode = 'holding'
                self.error = ''
            except Exception as error:
                self._remember_fault(str(error),stage='connecting')
                self._close_adapters()
                self.mode = 'fault'
                raise
            if self.home_on_connect:
                self._begin_alignment(client, self.clock(), follow=False)
            return
        if action != 'enable':
            raise ValueError(f'Unknown control action: {action}')
        self._begin_alignment(client, now, follow=True)

    def _begin_alignment(self, client, now, follow):
        if self.mode != 'holding':
            raise ValueError('Connect and hold the robot before enabling teleop')
        if not self._lease_valid(client, now):
            raise ValueError('Browser heartbeat missing')
        needs_alignment = False
        alignment_goals = {}
        leader_at_alignment = {}
        for arm in self.arms:
            leader_at_alignment[arm.side] = self._leader(arm, now).copy()
            target = leader_at_alignment[arm.side]
            if not follow and self.home_pose is not None:
                from .samples import ArmSample
                target = checked(ArmSample(now,np.asarray(self.home_pose)),now)
            actual = checked(self.adapters[arm.side].observe(), now, observation=True)
            alignment_goals[arm.side] = target.copy()
            delta = np.abs(target-actual)
            self.alignment[arm.side] = delta.tolist()
            self.targets[arm.side] = actual.copy()
            needs_alignment = needs_alignment or np.any(delta > self.settings.alignment_tolerance())
        self.owner = client
        self._leader_at_alignment = leader_at_alignment
        self._alignment_goals = alignment_goals
        self._alignment_then = 'teleop' if follow else 'holding'
        self.mode = 'aligning' if needs_alignment else self._alignment_then
        if self.mode == 'holding':self.owner = None
        self.error = ''

    def tick(self, now):
        dt = min(max(now-self._previous, 0.), .1)
        self._previous = now
        if self._teleop_stop.is_set():
            self._teleop_stop.clear()
            while not self._requests.empty():
                try:
                    self._requests.get_nowait()
                except queue.Empty:
                    break
            self._hold(now)
        if self.mode in ('aligning','teleop') and not self._lease_valid(self.owner, now):
            self._hold(now, reason='Browser disconnected or heartbeat expired')
        for _ in range(32):
            try:
                action, client, generation = self._requests.get_nowait()
            except queue.Empty:
                break
            try:
                if action == 'enable' and generation != self._generation:
                    raise ValueError('Control state changed; enable again explicitly')
                self._action(action, client, now)
            except Exception as error:
                self.error = str(error)
                import logging
                logging.exception('Dashboard robot action %s failed',action)
        now = self.clock()
        for side, adapter in self.adapters.items():
            try:
                sample = adapter.observe()
                self.observations[side] = sample
                checked(sample, now, observation=True)
            except Exception as error:
                self._hold(now, fault=True, reason=f'{side} YAM: {error}')
                break
        if self.mode in ('aligning','teleop'):
            try:
                # Validate both arms before sending either command.
                goals = {a.side: self._leader(a, now) for a in self.arms}
                if self.mode == 'aligning':
                    if any(np.max(np.abs(goals[s]-q))>.05 for s,q in self._leader_at_alignment.items()):
                        self._hold(now, reason='Keep GELLO still during alignment; enable again to restart')
                        self._publish()
                        return
                    goals = self._alignment_goals
                speed = self.settings.alignment_speed if self.mode == 'aligning' else self.settings.teleop_speed
                for side, goal in goals.items():
                    target = self.targets[side]+np.clip(goal-self.targets[side], -speed*dt, speed*dt)
                    self.adapters[side].command(target)
                    self.targets[side] = target.copy()
                    self.alignment[side] = np.abs(goal-self.observations[side].position).tolist()
                if self.mode == 'aligning' and all(np.all(np.asarray(delta)<=self.settings.alignment_tolerance()) for delta in self.alignment.values()):
                    self.mode = self._alignment_then
                    if self.mode == 'holding':self.owner = None
            except Exception as error:
                self._hold(now, fault=True, reason=str(error))
        self._publish()

    def _publish(self):
        from .haptics import reflected_current,HapticTarget
        now=self.clock()
        for arm in self.arms:
            if arm.side not in self.haptics:continue
            current=reflected_current(arm,self.observations.get(arm.side),now,strength=.3*self.settings.feedback_percent/100.) if self.mode=='teleop' and self.force_feedback else None
            observation=self.observations.get(arm.side)
            hold=self.settings.hold_percent if self.mode=='teleop' and observation is not None and 0<=now-observation.acquired_at<=.1 else 0.
            enabled=current is not None or hold>0
            self.haptics[arm.side].publish(HapticTarget(observation.acquired_at if enabled else now,
                np.zeros(7) if current is None else current,enabled=enabled,
                hold_percent=hold,yield_degrees=self.settings.hold_yield_deg))
        import copy
        with self._status_lock:
            self._published = copy.deepcopy(dict(mode=self.mode, owner=self.owner, error=self.error, last_fault=self.last_fault,
                settings=self.settings.as_dict(),settings_error=self.settings_error,settings_revision=self.settings_revision,
                observations=self.observations, targets=self.targets, alignment=self.alignment,force_feedback=self.force_feedback))

    def status(self):
        import copy
        with self._status_lock:
            return copy.deepcopy(self._published)

    def start(self):
        def loop():
            while not self._stop.is_set():
                start = self.clock()
                try:
                    self.tick(start)
                except Exception as error:
                    self._hold(self.clock(), True, f'Controller fault: {error}')
                    self._publish()
                self._stop.wait(max(0., 1/30-(self.clock()-start)))
        self._thread = threading.Thread(target=loop, name='teleop-supervisor', daemon=True)
        self._thread.start()

    def _close_adapters(self):
        errors = []
        for side, adapter in list(self.adapters.items()):
            try:
                adapter.close()
            except Exception as error:
                errors.append(f'{side}: {error}')
            else:
                del self.adapters[side]
        self._release_uncertain = bool(errors)
        if errors:
            self._remember_fault('Robot release not confirmed: '+ '; '.join(errors))
            self.mode = 'fault'
            self.owner = None
            self.error = 'Robot release not confirmed: '+ '; '.join(errors)
            self._release_error = self.error
            raise RuntimeError(self.error)

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15.)
            if self._thread.is_alive():
                raise RuntimeError('Controller shutdown is still pending; physical state unconfirmed')
        self._hold(self.clock())
        self._close_adapters()
        self.mode = 'simulation'
        self._publish()
