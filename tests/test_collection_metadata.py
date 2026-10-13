"""Controlled CARLA-shaped objects, not a claim of simulator validation."""
import json
import threading
import time
from types import SimpleNamespace as NS

import pytest
from sdv.collection.metadata import (MetadataRecorder, aligned_packets, collection_schema,
                                     publish_sensor, atomic_json)


def vec(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def wp(x=0., lane=1):
    return NS(transform=NS(location=vec(x), rotation=NS(yaw=0., pitch=0., roll=0.)),
              road_id=7, section_id=0, lane_id=lane, s=x, is_junction=False,
              junction_id=-1, lane_type='Driving', lane_width=3.5, lane_change='Both',
              left_lane_marking=NS(type='Solid', color='White', width=.12, lane_change='NONE'),
              right_lane_marking=NS(type='Broken', color='White', width=.12, lane_change='Right'))


def fixture():
    point = wp(1)
    ctrl = NS(throttle=.6, steer=.2, brake=0., gear=2, reverse=False, hand_brake=False, manual_gear_shift=False)
    light = NS(id=20, type_id='traffic.traffic_light', state='Red', get_location=lambda: vec(8),
               get_stop_waypoints=lambda: [wp(6)], get_affected_lane_waypoints=lambda: [wp(6)])
    light.get_state = lambda: light.state
    ego = NS(id=1, type_id='vehicle.ego', get_location=lambda: vec(1), get_transform=lambda: point.transform,
             get_control=lambda: ctrl, get_velocity=lambda: vec(3), get_acceleration=lambda: vec(1),
             get_angular_velocity=lambda: vec(z=2), get_speed_limit=lambda: 40., get_traffic_light=lambda: light)
    actor = NS(id=3, type_id='vehicle.other', get_location=lambda: vec(4), get_velocity=lambda: vec(2))
    carla_map = NS(name='Town01', get_waypoint=lambda loc: point)
    world = NS(get_actors=lambda: [ego, light, actor], get_weather=lambda: NS(cloudiness=10.), get_map=lambda: carla_map)
    planner = NS(target_waypoint=wp(5), target_road_option='LEFT', _waypoints_queue=[(wp(5), 'LEFT')], _target_speed=35.)
    agent = NS(_local_planner=planner, _incoming_direction='LEFT', _global_route=[(point, 'LANEFOLLOW'), (wp(5), 'LEFT')])
    snap = NS(frame=42, timestamp=NS(elapsed_seconds=4.2))
    packet = {'frame': 42, 'timestamp_s': 4.2, 'world_transform': {'position': {'x': 1., 'y': 0., 'z': 2.}, 'rotation_deg': {'yaw':0.}}, 'data': [1]}
    return world, ego, agent, snap, {'front_cam_rgb':packet}, light


def test_metadata_uses_native_state_and_distinguishes_online_control(tmp_path):
    world, ego, agent, snap, packets, light = fixture()
    record = MetadataRecorder().capture(world, ego, agent, tmp_path/'scene_1', '00002_a', snap, packets, applied_control=ego.get_control())
    assert record['synchronization']['carla_frame'] == 42
    assert record['synchronization']['timestamp_s'] == 4.2
    assert record['synchronization']['sensors']['front_cam_rgb']['frame'] == 42
    assert record['ego']['acceleration_m_s2']['x'] == 1
    assert record['ego']['angular_velocity_deg_s']['z'] == 2
    assert record['ego']['gear'] == 2
    assert record['road']['lane_id'] == 1
    assert record['road']['left_lane_marking']['type'] == 'Solid'
    assert record['navigation']['road_option'] == 'LEFT'
    assert record['navigation']['target_speed_kmh'] == 35
    assert record['navigation']['local_waypoints_world'][0]['position']['x'] == 5
    assert 'not future' in record['navigation']['source']
    assert 'subsequent' in record['ego']['applied_control_semantics']
    route = json.loads((tmp_path/'scene_1'/record['navigation']['global_route_file']).read_text())
    assert route['online_plan'] is True
    assert route['waypoints'][1]['road_option'] == 'LEFT'
    assert record['environment']['nearby_actors'][0]['distance_m'] == 3


def test_traffic_applicability_visibility_and_stopline_are_not_guessed(tmp_path):
    world, ego, agent, snap, packets, light = fixture()
    recorder = MetadataRecorder()
    record = recorder.capture(world, ego, agent, tmp_path, '00002_a', snap, packets)
    signal = record['traffic']['signals'][0]
    assert signal['affects_ego'] is True
    assert signal['visible_in_camera'] is None
    assert signal['distance_m'] == 7
    assert signal['stop_waypoints_world'][0]['position']['x'] == 6
    assert record['traffic']['stop_line_distance_m'] is None
    assert signal['state_changed_at_frame'] is None
    light.state = 'Green'
    recorder.observe_signals([light], 43)  # between saved frames
    snap.frame = 44
    ego.get_traffic_light = lambda: None
    record = recorder.capture(world, ego, agent, tmp_path, '00003_a', snap, packets)
    assert record['traffic']['affecting_signal_id'] is None
    assert record['traffic']['signals'][0]['affects_ego'] is False
    assert record['traffic']['signals'][0]['state_changed_at_frame'] == 43
    assert record['traffic']['signals'][0]['first_observed_frame'] == 42


def test_absent_api_is_null_not_synthetic_value(tmp_path):
    snap=NS(frame=1, timestamp=NS(elapsed_seconds=.1))
    out=MetadataRecorder().capture(NS(), NS(id=1), NS(), tmp_path, '0', snap, {})
    assert out['ego']['velocity_m_s'] is None
    assert out['road']['lane_id'] is None
    assert out['navigation']['target_speed_kmh'] is None
    assert out['traffic']['signals'] == []
    json.dumps(out, allow_nan=False)


def test_packets_wait_for_exact_frame_and_keep_immutable_reference():
    a, b = NS(packet=None), NS(packet=None)
    old_data = [1]
    publish_sensor(a, NS(frame=3, timestamp=.3), old_data)
    publish_sensor(b, NS(frame=2, timestamp=.2), [2])
    result, missing = aligned_packets({'rgb':a,'seg':b}, 3, timeout_s=0)
    assert result == {} and missing == ['seg']
    timer=threading.Timer(.02, lambda: publish_sensor(b, NS(frame=3, timestamp=.3), [3]))
    timer.start()
    packets, missing = aligned_packets({'rgb':a,'seg':b}, 3, timeout_s=.5)
    timer.join()
    assert missing == []
    publish_sensor(a, NS(frame=4, timestamp=.4), [4])
    assert packets['rgb']['data'] is old_data
    assert packets['rgb']['frame'] == 3


def test_routes_written_once_but_refresh_for_new_destination_and_scene(tmp_path):
    _, _, agent, _, _, _=fixture()
    recorder=MetadataRecorder()
    first=recorder.route_reference(agent,tmp_path/'a')
    before=(tmp_path/'a'/first).stat().st_mtime_ns
    assert recorder.route_reference(agent,tmp_path/'a')==first
    assert (tmp_path/'a'/first).stat().st_mtime_ns==before
    assert recorder.route_reference(agent,tmp_path/'b')==first
    assert (tmp_path/'b'/first).exists()
    agent._global_route=[(wp(99),'RIGHT')]
    assert recorder.route_reference(agent,tmp_path/'b') != first


def test_atomic_commit_and_native_schema_are_explicit(tmp_path):
    data=collection_schema()
    atomic_json(tmp_path/'info/1.json', data)
    assert json.loads((tmp_path/'info/1.json').read_text())==data
    assert not (tmp_path/'info/1.json.tmp').exists()
    assert data['encodings']['depth']['dtype']=='float32'
    assert data['encodings']['bbox']['object_id'] is None
    assert data['coordinates']['bev']['pixels_per_meter']==5
    assert data['navigation']['future_observed_ground_truth'] is False


def test_metric_depth_survives_npy_beyond_png_limit(tmp_path):
    np=pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    from sdv.collection.encoding import depth_outputs
    raw=np.array([[[0,0,0],[0,0,1],[128,0,0],[255,255,255]]],dtype=np.uint8)
    metres, preview=depth_outputs(raw)
    assert metres.dtype==np.float32
    assert metres[0,1]==pytest.approx(1000/(256**3-1),rel=1e-6)
    assert metres[0,2]>500
    assert metres[0,3]==1000
    assert preview.tolist()==[[0,0,255,255]]
    np.save(tmp_path/'depth.npy',metres,allow_pickle=False)
    assert np.array_equal(np.load(tmp_path/'depth.npy',allow_pickle=False),metres)


def test_od_uses_actual_current_semantic_tags():
    np=pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    from sdv.collection.encoding import semantic_visibility_ratio
    assert semantic_visibility_ratio(np.array([[14,15,16,17,18,19]]),'Vehicle')==1
    assert semantic_visibility_ratio(np.array([[10,4]]),'Vehicle')==0
    assert semantic_visibility_ratio(np.array([[12,12]]),'Pedestrian')==1
    assert semantic_visibility_ratio(np.array([[4,4]]),'Pedestrian')==0


def _world_method(name, namespace):
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).parents[1]/'sdv/collection/carla_engine/main.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='World')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    module=ast.Module(body=[method])
    # Python 3.7 AST has no type_ignores; later Python requires it at compile time.
    if hasattr(ast.Module, '_fields') and 'type_ignores' in ast.Module._fields:
        module.type_ignores=[]
    exec(compile(ast.fix_missing_locations(module),'actual-world-method','exec'),namespace)
    return namespace[name]


def test_od_collector_uses_all_vehicle_groups_and_per_camera_calibration():
    np=pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    from sdv.collection.encoding import semantic_visibility_ratio
    function=_world_method('_get_2d_bbs',{'np':np,'MIN_WIDTH':5,'MIN_HEIGHT':5,'semantic_visibility_ratio':semantic_visibility_ratio})
    received=[]
    world=NS(_create_bb_points=lambda bbox:np.ones((8,4)), _world_to_sensor=lambda pts,*args:pts,
             _coords_to_2d_bb=lambda pts,cam:(received.append(cam) or [(0,0),(10,10)]))
    camera={'width':640,'height':480,'calibration':np.eye(3)}
    groups={'vehicles':[1],'trucks':[2],'bicycles':[3],'pedestrians':[]}
    bbs=function(world,None,None,groups,np.full((10,10),14),camera)
    assert len(bbs)==3 and all(row[4]==0 for row in bbs)
    assert all(c is camera for c in received)
    assert function(world,None,None,groups,np.full((10,10),10),camera)==[]
    groups={'vehicles':[],'pedestrians':[4]}
    assert function(world,None,None,groups,np.full((10,10),12),camera)[0][-1]==1


def test_projection_uses_selected_camera_not_first_camera():
    np=pytest.importorskip('numpy')
    function=_world_method('_coords_to_2d_bb',{'np':np})
    points=np.array([[10.,10.,10.,10.],[-1.,1.,1.,-1.],[-1.,-1.,1.,1.]])
    first={'width':100,'height':100,'calibration':np.array([[50,0,50],[0,50,50],[0,0,1]])}
    second={'width':200,'height':120,'calibration':np.array([[100,0,100],[0,100,60],[0,0,1]])}
    world=NS(_sensor_data=first)
    assert function(world,points)==[(45,45),(55,55)]
    assert function(world,points,second)==[(90,50),(110,70)]


def test_external_tick_is_rejected_before_sample_commit():
    from sdv.collection.metadata import assert_world_frame
    world=NS(frame=31)
    world.get_snapshot=lambda:NS(frame=world.frame)
    assert_world_frame(world,31)
    world.frame=32
    with pytest.raises(RuntimeError,match='31.*32'):
        assert_world_frame(world,31)


def test_captured_imu_keeps_measurement_transform_and_alignment(tmp_path):
    world,ego,agent,snap,packets,_=fixture()
    sensor=NS(packet=None)
    measurement=NS(frame=42,timestamp=4.2,transform=NS(location=vec(1,2,3),rotation=NS(yaw=4,pitch=5,roll=6)))
    publish_sensor(sensor,measurement,{'accelerometer':{'x':1}})
    captured,_=aligned_packets({'imu':sensor},42,timeout_s=0)
    publish_sensor(sensor,NS(frame=43,timestamp=4.3),{'accelerometer':{'x':99}})
    packets.update(captured)
    record=MetadataRecorder().capture(world,ego,agent,tmp_path,'frame',snap,packets)
    imu=record['synchronization']['sensors']['imu']
    assert imu['frame']==42 and imu['timestamp_s']==4.2 and imu['aligned'] is True
    assert imu['world_transform']['position']=={'x':1.,'y':2.,'z':3.}
    assert captured['imu']['data']['accelerometer']['x']==1


def test_failed_signal_lookup_is_unknown_not_no_affecting_light(tmp_path):
    world,ego,agent,snap,packets,_=fixture()
    def unavailable():raise RuntimeError('CARLA RPC unavailable')
    ego.get_traffic_light=unavailable
    record=MetadataRecorder().capture(world,ego,agent,tmp_path,'frame',snap,packets)
    assert record['traffic']['affecting_signal_lookup_available'] is False
    assert record['traffic']['signals'][0]['affects_ego'] is None


def test_exact_frame_ring_keeps_late_packets_without_regressing_latest():
    wrapper = NS(packet=None)
    for frame in (41, 43, 42):
        publish_sensor(wrapper, NS(frame=frame,timestamp=frame*.05), [frame])
    assert wrapper.packet['frame'] == 43
    packets, missing = aligned_packets({'rgb':wrapper},42,timeout_s=0)
    assert not missing and packets['rgb']['data'] == [42]
    packets, missing = aligned_packets({'rgb':wrapper},40,timeout_s=0)
    assert not packets and missing == ['rgb']
