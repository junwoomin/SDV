"""Mocked CARLA boundaries: these tests do not establish live driving quality."""
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sdv.collection.driving_runtime import DrivingRuntime, RouteTracker, action_control, manual_command


def vec(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def waypoint(x=0., y=0., z=0.):
    return NS(transform=NS(location=vec(x,y,z), rotation=NS(yaw=0., pitch=0., roll=0.)))


class VehicleControl:
    def __init__(self, throttle=0., brake=0., steer=0., **kwargs):
        self.throttle, self.brake, self.steer = throttle, brake, steer
        self.gear = 1
        self.reverse = self.hand_brake = self.manual_gear_shift = False
        self.__dict__.update(kwargs)


class Observation:
    def __init__(self):
        self.frames = []
        self.ready = True
        self.error = None

    def observe_carla(self, ego, snapshot, route, control):
        if self.error:
            raise self.error
        self.frames.append(snapshot.frame)
        return dict(birdview=np.zeros((15,192,192),np.uint8), state=np.zeros(6,np.float32),
                    metadata=dict(history_frames=list(self.frames),history_ready=self.ready))


class Policy:
    metadata = dict(checkpoint_sha256='test-fixture', status='ready')
    def __init__(self):
        self.calls = 0
        self.action = [.6,-.2]

    def predict(self, observation):
        self.calls += 1
        return dict(raw_action=self.action)


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.delenv('SDV_DRIVING_CONFIG', raising=False)
    monkeypatch.setitem(sys.modules, 'carla', NS(VehicleControl=VehicleControl, LaneType=NS(Driving='Driving')))
    location = vec()
    actor_state = NS(get_transform=lambda:NS(location=location,rotation=NS(yaw=0,pitch=0,roll=0)),
                     get_velocity=lambda:vec(2))
    ego = NS(id=1,is_alive=True,controls=[],autopilot_calls=[], current=VehicleControl(throttle=.8),
             get_location=lambda:location,get_traffic_light=lambda:None)
    ego.get_control = lambda:ego.current
    def apply(value):
        ego.current = value
        ego.controls.append(value)
    ego.apply_control = apply
    ego.set_autopilot = lambda *args:ego.autopilot_calls.append(args)
    carla_map = NS(get_waypoint=lambda *args, **kwargs:waypoint(location.x))
    world = NS(frame=10,get_map=lambda:carla_map,
               get_settings=lambda:NS(synchronous_mode=True,fixed_delta_seconds=1/30))
    def snapshot(frame=None):
        if frame is not None:
            world.frame = frame
        return NS(frame=world.frame,timestamp=NS(elapsed_seconds=world.frame/30),find=lambda actor_id:actor_state)
    world.get_snapshot = snapshot
    obs, policy = Observation(), Policy()
    collision = NS(events=[])
    def make(**config):
        runtime = DrivingRuntime(world, ego, dict(agent='roach',control_hz=30,**config),
            observation_adapter=obs, policy=policy, collision_sensor=collision)
        runtime.set_global_plan([(waypoint(x),'LANEFOLLOW') for x in range(0,101)])
        return runtime
    return NS(make=make,world=world,ego=ego,observation=obs,policy=policy, snapshot=snapshot,
              location=location,collision=collision)


def test_action_control_preserves_acceleration_and_steer_sign():
    assert action_control([.4,-.5]) == dict(throttle=.4,brake=0.,steer=-.5,hand_brake=False,reverse=False,manual_gear_shift=False)
    assert action_control([-.6,.7])['brake'] == .6
    assert action_control([-.6,.7])['throttle'] == 0


def test_navigation_is_serializable_for_actual_collection_commit(runtime_fixture):
    f = runtime_fixture
    runtime = f.make()
    runtime.run_step(f.snapshot())
    # The collection info JSON is the commit marker, and rejects NumPy bool_.
    json.dumps(runtime.last_navigation, allow_nan=False)
    assert type(runtime.last_navigation['destination_reached']) is bool


@pytest.mark.parametrize('action', [[np.nan,0], [np.inf,0], [1.1,0], [.2], [[.2,0]]])
def test_malformed_action_is_rejected(action):
    with pytest.raises(ValueError):
        action_control(action)


def test_roach_controls_actual_actor_and_records_causality(runtime_fixture):
    f=runtime_fixture;r=f.make()
    ctrl, info=r.run_step(f.snapshot())
    assert f.ego.controls[-1] is ctrl
    assert ctrl.throttle == .6 and ctrl.steer == -.2
    assert r.last_policy['observation_frame'] == 10
    assert r.last_policy['command_effective_from_frame'] == 11
    assert r.last_policy['previous_vehicle_control']['brake'] == 1  # initialization brake
    assert r.last_policy['raw_action'] == [.6,-.2]
    assert info['next_control']['steer'] == -.2
    assert f.ego.autopilot_calls[-1] == (False,)


def test_observation_failure_immediately_replaces_previous_throttle(runtime_fixture):
    f=runtime_fixture;r=f.make()
    r.run_step(f.snapshot())
    f.observation.error=ValueError('shape mismatch')
    with pytest.raises(ValueError,match='shape mismatch'):
        r.run_step(f.snapshot(11))
    assert f.ego.current.throttle == 0 and f.ego.current.brake == 1
    assert r.last_policy['status'] == 'error'
    f.observation.error=None
    with pytest.raises(RuntimeError):
        r.run_step(f.snapshot(12))
    assert f.ego.current.brake == 1


def test_nan_policy_action_fail_stops(runtime_fixture):
    f=runtime_fixture;r=f.make();f.policy.action=[np.nan, .2]
    with pytest.raises(ValueError):r.run_step(f.snapshot())
    assert f.ego.current.brake == 1 and f.ego.current.throttle == 0


def test_missing_tick_is_fatal_for_history(runtime_fixture):
    f=runtime_fixture;r=f.make();r.run_step(f.snapshot())
    with pytest.raises(RuntimeError, match='Non-consecutive'):
        r.run_step(f.snapshot(12))
    assert f.ego.current.brake == 1


def test_external_tick_between_snapshot_and_control_fail_stops(runtime_fixture):
    f=runtime_fixture;r=f.make();snap=f.snapshot();f.world.frame=11
    with pytest.raises(RuntimeError, match='Frame'):
        r.run_step(snap)
    assert f.ego.current.brake == 1


def test_history_warmup_preserves_raw_action_and_stores_safety_control(runtime_fixture):
    f=runtime_fixture;r=f.make();f.observation.ready=False
    ctrl,info=r.run_step(f.snapshot())
    assert r.last_policy['raw_vehicle_control']['throttle']==.6
    assert r.last_policy['applied_control']['throttle']==0
    assert r.last_policy['safety_reason']=='history_warmup'
    assert ctrl.brake==1


def test_delayed_collision_is_not_missed_and_latches_stop(runtime_fixture):
    f=runtime_fixture;r=f.make();r.run_step(f.snapshot())
    f.collision.events.append(dict(frame=9,other_actor_id=2))
    _,info=r.run_step(f.snapshot(11))
    assert info['collision'] and f.ego.current.brake==1
    assert r.metrics['collision_count']==1
    with pytest.raises(RuntimeError,match='collision'):
        r.run_step(f.snapshot(12))


def test_low_control_frequency_does_not_skip_history(runtime_fixture):
    f=runtime_fixture
    r=DrivingRuntime(f.world,f.ego,dict(agent='roach',control_hz=15),policy=f.policy,
                     observation_adapter=f.observation)
    r.set_global_plan([(waypoint(x),'LANEFOLLOW') for x in range(100)])
    for frame in range(10,14):r.run_step(f.snapshot(frame))
    assert f.observation.frames==[10,11,12,13]
    assert f.policy.calls==2


def test_manual_deadman_rejects_stale_replay_and_wrong_job(tmp_path):
    path=tmp_path/'manual.json'
    command=dict(issued_unix=100.,sequence=5,job_id='abc',throttle=.6,brake=0.,steer=-.2)
    path.write_text(json.dumps(command))
    ctrl, seq, error=manual_command(path,4,now=100.1,job_id='abc')
    assert error is None and ctrl['throttle']==.6 and seq==5
    for kwargs in (dict(now=101),dict(now=100.1,previous_sequence=6),dict(now=100.1,job_id='xyz')):
        ctrl,_,error=manual_command(path,**kwargs)
        assert error and ctrl['brake']==1 and ctrl['throttle']==0


def test_route_progress_uses_height_and_does_not_jump_crossings():
    route=RouteTracker()
    plan=[(waypoint(x,0,0),'LANEFOLLOW') for x in range(20)]
    plan += [(waypoint(x,0,8),'LANEFOLLOW') for x in range(19,-1,-1)]
    route.set_plan(plan)
    route.update(vec(5,0,0))
    assert route.index==4 or route.index==5
    assert route.progress_m==5
    route.update(vec(5,0,8))
    assert route.progress_m==5 and route.deviation_m==8


def test_route_arrival_requires_destination_and_progress():
    route=RouteTracker();route.set_plan([(waypoint(x),'LANEFOLLOW') for x in range(21)])
    route.update(vec(10));assert not route.arrived
    route.update(vec(20));assert route.arrived and route.completion==100


def test_runtime_rejects_async_world_after_braking(runtime_fixture):
    f=runtime_fixture
    f.world.get_settings=lambda:NS(synchronous_mode=False,fixed_delta_seconds=None)
    with pytest.raises(ValueError,match='synchronous'):
        f.make()
    assert f.ego.current.brake==1 and f.ego.current.throttle==0


def test_matching_traffic_manager_port_used_for_disable(runtime_fixture):
    f=runtime_fixture
    tm=NS(get_port=lambda:8123,set_path=lambda *args:None)
    r=DrivingRuntime(f.world,f.ego,dict(agent='autopilot',control_hz=30),traffic_manager=tm)
    r.set_global_plan([(waypoint(x),'LANEFOLLOW') for x in range(100)])
    assert f.ego.autopilot_calls[-1]==(True,8123)
    r.run_step(f.snapshot())
    assert r.last_policy['command_frame'] is None
    r.close()
    assert f.ego.autopilot_calls[-1]==(False,8123)
    assert f.ego.current.brake==1


def test_history_ready_transition_requires_fresh_inference_even_at_low_hz(runtime_fixture):
    f=runtime_fixture
    r=DrivingRuntime(f.world,f.ego,dict(agent='roach',control_hz=1),policy=f.policy,
                     observation_adapter=f.observation)
    r.set_global_plan([(waypoint(x),'LANEFOLLOW') for x in range(100)])
    f.observation.ready=False
    r.run_step(f.snapshot(10))
    r.run_step(f.snapshot(11))
    assert f.policy.calls==1 and f.ego.current.brake==1
    assert r.last_policy['action_source_observation_frame']==10
    assert r.last_policy['held_action']
    f.observation.ready=True
    r.run_step(f.snapshot(12))
    assert f.policy.calls==2 and f.ego.current.throttle==.6
    assert r.last_policy['action_source_observation_frame']==12
    assert not r.last_policy['held_action']
