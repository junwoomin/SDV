"""Frame-pinned CARLA policy execution. Importable without CARLA or torch.

Observation at F produces a command submitted at F for physics tick F+1.
The previous command which produced state F is stored separately. CARLA alone
integrates the vehicle's dynamics; this module never changes actor transforms.
"""
from collections import deque
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from sdv.collection.metadata import (assert_world_frame, call, control, enum_name,
                                      snapshot_state, vector, rotation)


def driving_settings(config=None):
    settings = dict(agent='roach', checkpoint='roach/log/ckpt_11833344.pth', device='cpu',
                    deterministic=True, policy_ppm=5, control_hz=30, seed=2026)
    supplied = json.loads(os.environ.get('SDV_DRIVING_CONFIG', '{}'))
    if not isinstance(supplied, dict):
        raise ValueError('SDV_DRIVING_CONFIG must be a JSON object')
    settings.update(config or {})
    settings.update(supplied)
    if settings['agent'] not in ('roach', 'existing', 'autopilot', 'manual'):
        raise ValueError('Unknown driving agent: ' + str(settings['agent']))
    if float(settings['policy_ppm']) != 5:
        raise ValueError('The original Roach checkpoint requires policy_ppm=5')
    return settings


def action_control(action):
    """Return canonical controls; action is signed acceleration, right steer."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError('Policy action must contain two finite values, shape (2,)')
    if np.max(np.abs(value)) > 1.000001:
        raise ValueError('Policy action is outside [-1, 1]')
    acceleration, steer = np.clip(value, -1, 1)
    return dict(throttle=float(max(acceleration, 0)), brake=float(max(-acceleration, 0)),
                steer=float(steer), hand_brake=False, reverse=False, manual_gear_shift=False)


def manual_command(path, previous_sequence=-1, now=None, job_id=None):
    """The web joystick is deadman-controlled; stale/replayed files stop the ego."""
    now = time.time() if now is None else now
    try:
        command = json.loads(Path(path).read_text(encoding='utf-8'))
        age = now - float(command['issued_unix'])
        sequence = int(command['sequence'])
        if age < -0.1 or age > 0.5 or sequence < previous_sequence:
            raise ValueError('Manual command expired or replayed')
        if job_id and command.get('job_id') != job_id:
            raise ValueError('Manual command belongs to another job')
        result = {name: float(command[name]) for name in ('throttle', 'brake', 'steer')}
        if not all(math.isfinite(v) for v in result.values()):
            raise ValueError('Non-finite manual command')
        if not (0 <= result['throttle'] <= 1 and 0 <= result['brake'] <= 1 and -1 <= result['steer'] <= 1):
            raise ValueError('Manual command outside control range')
        result.update(hand_brake=False, reverse=False, manual_gear_shift=False)
        return result, sequence, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(throttle=0., brake=1., steer=0., hand_brake=False, reverse=False,
                    manual_gear_shift=False), previous_sequence, str(error)


class RouteTracker:
    """Keep route progress monotonic, bounded and independent of a controller."""
    def __init__(self):
        self.set_plan([])

    def set_plan(self, plan):
        self.plan = list(plan)
        self.xyz = np.asarray([[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                               for wp, _ in self.plan], dtype=np.float64).reshape(-1, 3)
        self.lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.xyz, axis=0), axis=1))]
        self.index = 0
        self.progress_m = 0.
        self.deviation_m = None
        self.arrived = False

    def update(self, location, maximum_advance_m=15.):
        if len(self.xyz) < 2:
            raise RuntimeError('A valid global route with at least two waypoints is required')
        point = np.array([location.x, location.y, location.z])
        # Bounded forward search prevents jumping to a later crossing or a road
        # directly below the ego. Distances include Z.
        end = min(len(self.xyz)-1, int(np.searchsorted(self.lengths, self.progress_m+maximum_advance_m))+1)
        start = max(0, self.index-2)
        a, b = self.xyz[start:end], self.xyz[start+1:end+1]
        ab = b-a
        t = np.clip(np.sum((point-a)*ab, axis=1) / np.maximum(np.sum(ab*ab, axis=1), 1e-12), 0, 1)
        distances = np.linalg.norm(point-(a+t[:, None]*ab), axis=1)
        winner = int(np.argmin(distances))
        segment = start+winner
        self.deviation_m = float(distances[winner])
        reached = self.lengths[segment] + t[winner] * (self.lengths[segment+1]-self.lengths[segment])
        # Reject huge lateral jumps as progress; downstream safety handles them.
        if self.deviation_m <= 5:
            self.progress_m = max(self.progress_m, float(reached))
            self.index = max(self.index, segment)
        self.arrived = bool(self.lengths[-1]-self.progress_m <= 3 and np.linalg.norm(point-self.xyz[-1]) <= 2)

    @property
    def completion(self):
        return min(100., 100*self.progress_m/max(float(self.lengths[-1]), 1e-6))


class DrivingRuntime:
    def __init__(self, world, ego, config=None, fixed_delta_seconds=1/30, collision_sensor=None,
                 lane_sensor=None, traffic_manager=None, bev_settings=None, policy=None,
                 observation_adapter=None, legacy_agent=None):
        import carla
        self.carla = carla
        self.world, self.ego = world, ego
        self._map = world.get_map()
        self.config = driving_settings(config)
        self.mode = self.config['agent']
        self.dt = float(fixed_delta_seconds)
        frequency = float(self.config['control_hz'])
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError('control_hz must be a positive finite frequency')
        ratio = 1/(self.dt*frequency)
        if ratio < 1 or abs(ratio-round(ratio)) > 1e-6:
            raise ValueError('control_hz must divide the synchronous simulator frequency exactly')
        self.control_stride = int(round(ratio))
        self.collision_sensor, self.lane_sensor = collision_sensor, lane_sensor
        self.tm = traffic_manager
        self.legacy_agent = legacy_agent
        self.policy, self.observation = policy, observation_adapter
        self._global_route = []
        self.route = RouteTracker()
        self._local_planner = SimpleNamespace(_waypoints_queue=deque(), target_waypoint=None,
                                             target_road_option=None, _target_speed=None)
        self.future_waypoints = []
        self.last_observation = None
        self.last_policy = {}
        self.last_navigation = {}
        self._last_frame = None
        self._step_count = 0
        self._manual_sequence = -1
        self._failure = None
        self._held_control = dict(throttle=0., brake=1., steer=0., hand_brake=False, reverse=False,
                                  manual_gear_shift=False)
        self._last_raw_action = None
        self._last_prediction = {}
        self._action_source_frame = None
        self._history_ready = False
        self._last_inference_ms = None
        self._seen_collisions = set()
        self._seen_lane_events = set()
        self._speed_sum = 0.
        self._route_deviation_sum = 0.
        self.metrics = dict(steps=0, route_completion_pct=0., collision_count=0,
                            lane_invasion_event_count=0, lane_departure_count=None, offroad_steps=0,
                            lane_departure_note='Lane invasion events include legal lane changes; use evaluator for violations', traffic_light_violation_count=None,
                            average_speed_kmh=0., average_route_deviation_m=0.,
                            traffic_light_violation_note='Use validation stop-line crossing evaluator; not inferred from being near red')
        self.emergency_stop('initialization')
        try:
            world_settings = world.get_settings()
            if not world_settings.synchronous_mode or abs(float(world_settings.fixed_delta_seconds or 0)-self.dt) > 1e-8:
                raise ValueError('DrivingRuntime requires synchronous CARLA with matching fixed_delta_seconds')
            if self.mode == 'roach':
                if self.policy is None:
                    from sdv.driving.policy import RoachPolicy
                    self.policy = RoachPolicy(self.config['checkpoint'], device=self.config['device'],
                                              deterministic=bool(self.config['deterministic']))
                if self.observation is None:
                    from sdv.driving.observation import RoachObservationAdapter
                    self.observation = RoachObservationAdapter.from_carla(
                        world, bev_settings=bev_settings, fixed_delta_seconds=self.dt)
            elif self.mode == 'existing' and self.legacy_agent is None:
                from sdv.collection.carla_engine.main import BehaviorAgent
                self.legacy_agent = BehaviorAgent(ego, behavior='normal')
            if self.legacy_agent is not None:
                if collision_sensor is not None:
                    self.legacy_agent.set_collision_sensor(collision_sensor)
                planner = self.legacy_agent._local_planner
                planner._dt = self.dt*self.control_stride
                planner._args_lateral_dict['dt'] = planner._dt
                planner._args_longitudinal_dict['dt'] = planner._dt
                planner._init_controller()
            if self.mode == 'autopilot' and self.tm is None:
                raise ValueError('CARLA Autopilot requires a TrafficManager')
        except BaseException:
            self.emergency_stop('initialization_failed')
            raise

    def _vehicle_control(self, values):
        for key, lower, upper in (('throttle', 0, 1), ('brake', 0, 1), ('steer', -1, 1)):
            value = float(values.get(key, 0))
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('Invalid VehicleControl ' + key)
        return self.carla.VehicleControl(**{k: v for k, v in values.items() if k in (
            'throttle', 'brake', 'steer', 'hand_brake', 'reverse', 'manual_gear_shift')})

    def emergency_stop(self, reason='stopped'):
        value = self._vehicle_control(dict(throttle=0., brake=1., steer=0., hand_brake=False,
                                          reverse=False, manual_gear_shift=False))
        try:
            if self.tm is not None:
                self.ego.set_autopilot(False, self.tm.get_port())
            else:
                self.ego.set_autopilot(False)
        finally:
            self.ego.apply_control(value)
        return value

    def close(self):
        try:
            if getattr(self.ego, 'is_alive', True):
                self.emergency_stop('closed')
        except RuntimeError:
            pass

    def set_global_plan(self, plan):
        self.route.set_plan(plan)
        if len(self.route.plan) < 2 or float(self.route.lengths[-1]) < 2:
            raise RuntimeError('Route is empty or destination is too close to the spawn')
        self._global_route = self.route.plan
        if self.legacy_agent is not None:
            self.legacy_agent.set_global_plan(self._global_route)
        if self.mode == 'autopilot':
            self.tm.set_path(self.ego, [wp.transform.location for wp, _ in self._global_route])
            self.ego.set_autopilot(True, self.tm.get_port())

    def set_destination(self, destination):
        from sdv.collection.carla_engine.agents.navigation.global_route_planner import GlobalRoutePlanner
        if not hasattr(self, '_global_planner'):
            self._global_planner = GlobalRoutePlanner(self._map, 1.0)
        self.set_global_plan(self._global_planner.trace_route(self.ego.get_location(), destination))

    def done(self):
        return self.route.arrived

    def _update_navigation(self, actor_state):
        transform = actor_state.get_transform()
        velocity = actor_state.get_velocity()
        speed = math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
        self.route.update(transform.location, maximum_advance_m=max(15, speed*self.dt*3))
        remaining = self._global_route[self.route.index:]
        self._local_planner._waypoints_queue = deque(remaining[:100])
        target_index = min(len(remaining)-1, max(1, int(speed*.5)))
        target, option = remaining[target_index]
        self._local_planner.target_waypoint = target
        self._local_planner.target_road_option = option
        self.future_waypoints = [(wp.transform.location.x, wp.transform.location.y) for wp, _ in remaining[1:8]]
        current_wp = self._map.get_waypoint(transform.location)
        onroad = self._map.get_waypoint(transform.location, project_to_road=False,
                                       lane_type=self.carla.LaneType.Driving)
        light = call(self.ego, 'get_traffic_light')
        light_state = enum_name(call(light, 'get_state')) or 'None'
        self.last_navigation = dict(road_option=enum_name(option), distance_to_route_m=self.route.deviation_m,
                                   route_completion_pct=self.route.completion, destination_reached=self.route.arrived,
                                   target_point=vector(target.transform.location),
                                   target_waypoint_world=dict(position=vector(target.transform.location)),
                                   target_speed_kmh=None, traffic_light_state=light_state,
                                   active_driving_agent=self.mode, route_length_m=float(self.route.lengths[-1]),
                                   remaining_distance_m=float(self.route.lengths[-1]-self.route.progress_m))
        self._speed_sum += speed*3.6
        self._route_deviation_sum += self.route.deviation_m
        self.metrics['steps'] += 1
        self.metrics.update(route_completion_pct=self.route.completion,
                            average_speed_kmh=self._speed_sum/self.metrics['steps'],
                            average_route_deviation_m=self._route_deviation_sum/self.metrics['steps'])
        if onroad is None:
            self.metrics['offroad_steps'] += 1
        collisions = list(getattr(self.collision_sensor, 'events', []) or [])
        new_collision = False
        for event in collisions:
            key = (event['frame'], event.get('other_actor_id'))
            if key not in self._seen_collisions:
                self._seen_collisions.add(key)
                self.metrics['collision_count'] += 1
                new_collision = True
        for event in list(getattr(self.lane_sensor, 'events', []) or []):
            if event['frame'] not in self._seen_lane_events:
                self._seen_lane_events.add(event['frame'])
                self.metrics['lane_invasion_event_count'] += 1
        return dict(current_speed=speed*3.6,
                    current_waypoint=current_wp.transform.location if current_wp is not None else transform.location,
                    target_waypoint=target.transform.location,
                    global_waypoints=[(wp.transform.location.x, wp.transform.location.y) for wp, _ in remaining[::5]][:20],
                    current_control=control(self.ego.get_control()), next_control=None,
                    traffic_light=light_state == 'Red', traffic_light_20=light_state == 'Red',
                    traffic_light_distance=None, vehicle_in_front=False,
                    collision=new_collision,
                    ego_vehicle_loc=vector(transform.location), ego_vehicle_rot=rotation(transform.rotation))

    def run_step(self, snapshot):
        started = time.perf_counter()
        frame = int(snapshot.frame)
        previous = control(self.ego.get_control())
        inference_ran = False
        reason = None
        raw_action = self._last_raw_action
        try:
            assert_world_frame(self.world, frame)
            if self._failure:
                raise RuntimeError(self._failure)
            if self._last_frame is not None and frame != self._last_frame+1:
                raise RuntimeError('Non-consecutive CARLA frames invalidate policy history')
            actor_state = snapshot_state(snapshot, self.ego)
            if actor_state is None:
                raise RuntimeError('Ego is absent from WorldSnapshot')
            info = self._update_navigation(actor_state)
            if self.mode == 'roach':
                route_xyz = self.route.xyz[max(0, self.route.index-8):self.route.index+50].tolist()
                self.last_observation = self.observation.observe_carla(self.ego, snapshot, route_xyz, previous)
            due = self._step_count % self.control_stride == 0
            if self.mode == 'roach':
                history_ready = self.last_observation.get('metadata', {}).get('history_ready') is not False
                due = due or (history_ready and not self._history_ready)
                self._history_ready = history_ready
            if due:
                self._action_source_frame = frame
                if self.mode == 'roach':
                    inference_started = time.perf_counter()
                    result = self.policy.predict(self.last_observation)
                    raw_action = np.asarray(result['raw_action'] if isinstance(result, dict) else result)
                    self._last_prediction = dict(result) if isinstance(result, dict) else {}
                    self._held_control = action_control(raw_action)
                    self._last_raw_action = raw_action.tolist()
                    self._last_inference_ms = (time.perf_counter()-inference_started)*1000
                    inference_ran = True
                elif self.mode == 'existing':
                    legacy_control, _legacy_info = self.legacy_agent.run_step()
                    self._held_control = {key: getattr(legacy_control, key) for key in (
                        'throttle', 'brake', 'steer', 'hand_brake', 'reverse', 'manual_gear_shift')}
                    raw_action = [legacy_control.throttle-legacy_control.brake, legacy_control.steer]
                    self._last_raw_action = raw_action
                elif self.mode == 'manual':
                    self._held_control, self._manual_sequence, reason = manual_command(
                        os.environ.get('SDV_MANUAL_CONTROL_FILE', ''), self._manual_sequence,
                        job_id=os.environ.get('SDV_JOB_ID'))
                    if reason:
                        reason = 'manual_deadman: '+reason
                    raw_action = [self._held_control['throttle']-self._held_control['brake'], self._held_control['steer']]
                    self._last_raw_action = raw_action
            # Deadman must still expire between policy control steps.
            if self.mode == 'manual' and not due:
                _, _, stale = manual_command(os.environ.get('SDV_MANUAL_CONTROL_FILE', ''), self._manual_sequence,
                                              job_id=os.environ.get('SDV_JOB_ID'))
                if stale:
                    reason = 'manual_deadman: '+stale
            if info['collision']:
                reason = 'collision'
                self._failure = reason
            elif self.route.deviation_m > float(self.config.get('maximum_route_deviation_m', 8.0)):
                reason = 'route_deviation'
                self._failure = reason
            elif self.route.arrived:
                reason = 'destination_reached'
            elif self.mode == 'roach' and self.last_observation.get('metadata', {}).get('history_ready') is False:
                reason = 'history_warmup'
            assert_world_frame(self.world, frame)
            if reason:
                applied = self.emergency_stop(reason)
            elif self.mode == 'autopilot':
                applied = self.ego.get_control()  # TM owns the upcoming command, unavailable until next tick.
            else:
                applied = self._vehicle_control(self._held_control)
                self.ego.apply_control(applied)
            info['next_control'] = control(applied)
            self._last_frame = frame
            self._step_count += 1
            metadata = dict((self.last_observation or {}).get('metadata', {}))
            self.last_policy = dict(name=self.mode, checkpoint=self.config['checkpoint'] if self.mode == 'roach' else None,
                device=self.config['device'] if self.mode == 'roach' else None, deterministic=self.config['deterministic'],
                checkpoint_metadata=getattr(self.policy, 'metadata', None), prediction=self._last_prediction,
                status='stopped' if reason else 'ready', birdview_shape=[15,192,192] if self.mode == 'roach' else None,
                state_shape=[6] if self.mode == 'roach' else None, policy_input_ppm=5,
                history_frames=metadata.get('history_frames', metadata.get('history_frame_ids')),
                history_ready=metadata.get('history_ready'), observation_metadata=metadata,
                inference_ms=self._last_inference_ms, inference_ran=inference_ran,
                action_source_observation_frame=self._action_source_frame if self.mode != 'autopilot' else None,
                held_action=(self._action_source_frame != frame) if self.mode != 'autopilot' else None,
                raw_action=self._last_raw_action, raw_vehicle_control=dict(self._held_control) if self.mode != 'autopilot' else None,
                applied_control=control(applied), safety_intervened=bool(reason), safety_reason=reason,
                control_latency_ms=(time.perf_counter()-started)*1000, observation_frame=frame,
                command_frame=frame if self.mode != 'autopilot' else None,
                command_effective_from_frame=frame+1 if self.mode != 'autopilot' else None,
                control_hz=float(self.config['control_hz']), simulator_hz=1/self.dt,
                previous_vehicle_control=previous,
                autopilot_command_semantics='Observed control at F; TM owns computation, command-to-physics frame unavailable' if self.mode == 'autopilot' else None)
            return applied, info
        except BaseException as error:
            self._failure = '{}: {}'.format(type(error).__name__, error)
            applied = self.emergency_stop(self._failure)
            self.last_policy = dict(name=self.mode, status='error', error=self._failure, observation_frame=frame,
                                    raw_action=self._last_raw_action, applied_control=control(applied),
                                    previous_vehicle_control=previous, safety_intervened=True,
                                    safety_reason='runtime_failure', control_latency_ms=(time.perf_counter()-started)*1000)
            raise
