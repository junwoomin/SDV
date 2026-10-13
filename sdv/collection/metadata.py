"""Versioned CARLA observations. No CARLA import; missing API values remain null.

Legacy info fields are kept by workflow.py. This extension adds only missing state,
calibration references and synchronization. Navigation is an online planner output;
it is never described as future observed ground truth.
"""
from collections import OrderedDict
import threading
import hashlib
import json
import math
import os
import time
from pathlib import Path


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def call(obj, name, default=None, *args):
    method = getattr(obj, name, None)
    if not callable(method):
        return default
    try:
        return method(*args)
    except (RuntimeError, AttributeError, TypeError):
        return default


def vector(value):
    return {k: number(getattr(value, k, None)) for k in ('x', 'y', 'z')} if value is not None else None


def enum_name(value):
    if value is None:
        return None
    return getattr(value, 'name', str(value).split('.')[-1])


def rotation(value):
    return {k: number(getattr(value, k, None)) for k in ('pitch', 'yaw', 'roll')} if value is not None else None


def control(value):
    if value is None:
        return None
    return {k: getattr(value, k, None) for k in ('throttle', 'steer', 'brake', 'gear', 'reverse', 'hand_brake', 'manual_gear_shift')}


def waypoint(value):
    if value is None:
        return None
    transform = getattr(value, 'transform', None)
    return {'position': vector(getattr(transform, 'location', None)),
            'rotation_deg': rotation(getattr(transform, 'rotation', None)),
            'road_id': getattr(value, 'road_id', None), 'section_id': getattr(value, 'section_id', None),
            'lane_id': getattr(value, 'lane_id', None), 's_m': number(getattr(value, 's', None)),
            'is_junction': getattr(value, 'is_junction', None)}


def lane_marking(value):
    if value is None:
        return None
    return {'type': enum_name(getattr(value, 'type', None)), 'color': enum_name(getattr(value, 'color', None)),
            'lane_change': enum_name(getattr(value, 'lane_change', None)), 'width_m': number(getattr(value, 'width', None))}


def distance(a, b):
    if a is None or b is None:
        return None
    values = [(number(getattr(a, k, None)), number(getattr(b, k, None))) for k in ('x', 'y', 'z')]
    if any(x is None or y is None for x, y in values):
        return None
    return math.sqrt(sum((x-y)**2 for x, y in values))


def snapshot_state(snapshot, actor):
    """Use pinned native state when the WorldSnapshot API is available."""
    if callable(getattr(snapshot, 'find', None)):
        return call(snapshot, 'find', None, getattr(actor, 'id', None))
    # Import-free legacy callers may supply only frame/timestamp, without actors.
    return actor


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def publish_sensor(wrapper, measurement, data):
    """Retain a bounded exact-frame history, including late sensor callbacks."""
    transform = getattr(measurement, 'transform', None)
    packet = {'data': data, 'frame': int(measurement.frame),
              'timestamp_s': float(measurement.timestamp),
              'world_transform': {'position': vector(getattr(transform, 'location', None)),
                                  'rotation_deg': rotation(getattr(transform, 'rotation', None))}}
    if not hasattr(wrapper, '_packet_lock'):
        wrapper._packet_history = OrderedDict()
        wrapper._packet_lock = threading.Lock()
    with wrapper._packet_lock:
        wrapper._packet_history[packet['frame']] = packet
        while len(wrapper._packet_history) > 32:
            wrapper._packet_history.popitem(last=False)
        previous = getattr(wrapper, 'packet', None)
        if previous is None or previous['frame'] <= packet['frame']:
            wrapper.packet = packet


def aligned_packets(sensors, frame, timeout_s=2.0):
    """Wait for F only. Past or future frames are never substituted for F."""
    deadline = time.monotonic() + timeout_s
    while True:
        packets = {}
        for name, sensor in sensors.items():
            lock = getattr(sensor, '_packet_lock', None)
            if lock is not None:
                with lock:
                    packet = sensor._packet_history.get(frame)
            else:
                packet = getattr(sensor, 'packet', None)
            packets[name] = packet
        missing = [name for name, packet in packets.items() if packet is None or packet.get('frame') != frame]
        if not missing:
            return packets, []
        if time.monotonic() >= deadline:
            return {}, missing
        time.sleep(0.002)


def assert_world_frame(world, expected_frame):
    """Reject another client's tick before publishing a supposedly aligned sample."""
    observed = world.get_snapshot().frame
    if observed != expected_frame:
        raise RuntimeError('CARLA Frame이 수집 중 {}에서 {}로 변경되었습니다. 다른 클라이언트의 tick 또는 비동기 실행을 중지하고 다시 수집하세요.'.format(expected_frame, observed))


def collection_schema(bev_settings=None):
    schema = {
        'schema_version': 2,
        'format': 'sdv-native-masks-v2',
        'depth_encoding': 'meters_u8',
        'depth_authoritative_encoding': 'meters_f32',
        'coordinates': {
            'world': {'axes': 'CARLA left-handed: x forward, y right, z up', 'position_unit': 'm', 'rotation_unit': 'deg'},
            'ego': {'axes': 'x forward, y right, z up', 'origin': 'ego actor transform', 'unit': 'm'},
            'camera': {'axes': 'sensor local x forward,y right,z up; optical projection x=y,y=-z,z=x', 'pixel_origin': 'top-left', 'box_format': 'x1,y1,x2,y2,class; pixels'},
            'bev': {'axes': 'u right,v backward; forward is image-up', 'width': 500, 'height': 500,
                    'pixels_per_meter': 5.0, 'ego_pixel': [250.0, 250.0], 'front_m': 50.0, 'back_m': 50.0, 'left_m': 50.0, 'right_m': 50.0,
                    'world_to_pixel': 'ego=R(-yaw)*(world-ego_world); u=250+5*ego_y; v=250-5*ego_x'}},
        'encodings': {
            'roach_observation': {'path': 'roach_observation/{frame}.npz',
                'birdview': {'shape': [15, 192, 192], 'dtype': 'uint8', 'range': [0, 255], 'pixels_per_meter': 5},
                'state': {'shape': [6], 'dtype': 'float32'}, 'profile': 'lrs_loop',
                'present_when': 'Driving Agent is Roach Policy',
                'contract': 'per-frame sdv_metadata.policy.observation_metadata; differs from training YAML'},
            'segmentation': {'authoritative': '{camera}_seg_ids/{frame}.png', 'dtype': 'uint8', 'schema': 'CARLA-0.9.15 semantic IDs 0..28', 'preview': '{camera}_seg/{frame}.png', 'preview_encoding': 'RGB palette stored by OpenCV BGR writer'},
            'depth': {'authoritative': '{camera}_depth_m/{frame}.npy', 'dtype': 'float32', 'unit': 'm', 'range_m': [0.0, 1000.0],
                      'source': 'CARLA raw BGRA: (R+256*G+65536*B)/(16777215)*1000',
                      'preview': '{camera}_depth/{frame}.png', 'preview_encoding': 'uint8 rounded/clipped metres, compatibility only; never use as metric depth when NPY exists'},
            'bbox': {'path': '{camera}_bbs/{frame}.npy', 'dtype': 'float32', 'shape': ['N', 5],
                     'columns': ['x1', 'y1', 'x2', 'y2', 'class_id'], 'coordinates': '2D pixel xyxy', 'classes': {'0': 'Vehicle', '1': 'Pedestrian'}, 'object_id': None},
            'lidar': {'path': 'lidar/{frame}.npy', 'legacy_scale': 2.0, 'note': 'Legacy collector multiplies x,y,z,intensity by 2; divide all columns by 2 for native CARLA values. Preserved for existing research.'}},
        'calibration': {'file': 'sensor_config.yaml', 'extrinsics': 'x,y,z metres and pitch,yaw,roll degrees relative to ego',
                        'intrinsics': 'fx=fy=width/(2*tan(horizontal_fov*pi/360)); cx=width/2; cy=height/2'},
        'metadata': {'path': 'info/{frame}.json', 'extension': 'sdv_metadata', 'commit_marker': 'info JSON is written after all sensor outputs'},
        'navigation': {'global_route': 'routes/{sha256}.json', 'online': True, 'future_observed_ground_truth': False},
    }
    if bev_settings is not None and bev_settings.get('mode') in ('height_aware', 'global_raster'):
        from sdv.bev.schema import channel_schema, SEMANTIC_CLASSES
        from sdv.bev.geometry import BEVGeometry
        geometry = BEVGeometry(**{key: bev_settings[key] for key in ('width', 'height', 'meters_per_pixel', 'anchor_x', 'anchor_y')})
        schema.update(schema_version=3, format='sdv-height-aware-bev-v3')
        schema['coordinates']['bev'] = geometry.metadata()
        schema['encodings']['bev_gt'] = {'path': 'bev_gt/{frame}.npz', 'schema_version': 3,
            'channels': channel_schema(), 'classes': SEMANTIC_CLASSES,
            'role': 'simulator_ground_truth', 'online_input': False,
            'preview': 'bev/{frame}.png', 'source': 'OpenDRIVE geometry and snapshot CARLA actor bounding boxes',
            'legacy_masks': 'das/lane/vehicle_mask/walker_mask are derived binary compatibility exports'}
        schema['bev_generator'] = dict(bev_settings)
        if bev_settings['mode'] == 'global_raster':
            schema.update(schema_version=4, format='sdv-global-png-bev-v4')
            bev = schema['encodings']['bev_gt']
            bev.update(schema_version=4,
                source='User-supplied global PNG map annotations and snapshot CARLA actors; static geometry is not rebuilt',
                metric_road_elevation_available=False,
                validity='validity describes available semantic pixels; elevation_validity independently describes measured road height',
                map_provenance='bev_map in this manifest and per-frame bev_gt metadata')
            bev['channels']['elevation_validity'] = {'dtype': 'uint8', 'range': [0, 1],
                'meaning': 'metric road elevation validity; zero for supplied PNG-only maps'}
            if not bev_settings.get('rotate_with_ego', True):
                schema['coordinates']['bev'].update(orientation='world_x_up_world_y_right',
                    yaw_applied_degrees=0.0)
    return schema


class MetadataRecorder:
    def __init__(self):
        self.signal_states = {}
        self.signal_static = {}
        self._route_object = None
        self._route_ref = None
        self._route_scene = None

    def observe_signals(self, lights, frame):
        """Called every simulator tick; records first observed transition, not interpolation."""
        for light in lights:
            sid = getattr(light, 'id', None)
            state = enum_name(call(light, 'get_state'))
            previous = self.signal_states.get(sid)
            if previous is None:
                self.signal_states[sid] = {'state': state, 'first_observed_frame': frame, 'state_changed_at_frame': None}
            elif state != previous['state']:
                previous.update(state=state, state_changed_at_frame=frame)

    def route_reference(self, agent, scene):
        route = getattr(agent, '_global_route', None)
        if not route:
            return None
        if route is self._route_object and str(scene) == self._route_scene:
            return self._route_ref
        data = {'schema_version': 1, 'coordinate_system': 'world', 'unit': 'm', 'source': 'CARLA GlobalRoutePlanner global route',
                'online_plan': True, 'waypoints': [dict(waypoint(wp), road_option=enum_name(option)) for wp, option in route]}
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
        rel = 'routes/{}.json'.format(digest)
        atomic_json(Path(scene) / rel, data)
        self._route_object, self._route_scene, self._route_ref = route, str(scene), rel
        return rel

    def capture(self, world, ego, agent, scene, frame_id, snapshot, packets, applied_control=None, carla_map=None):
        native_frame = int(snapshot.frame)
        carla_map = carla_map or call(world, 'get_map')
        ego_state = snapshot_state(snapshot, ego)
        ego_tf = call(ego_state, 'get_transform')
        ego_loc = getattr(ego_tf, 'location', None)
        wp = call(carla_map, 'get_waypoint', None, ego_loc) if ego_loc is not None else None
        planner = getattr(agent, '_local_planner', None)
        current_ctrl = call(ego, 'get_control')
        unavailable_signal = object()
        affecting = call(ego, 'get_traffic_light', unavailable_signal)
        signal_lookup_available = affecting is not unavailable_signal
        affecting_id = getattr(affecting, 'id', None)
        actors = list(call(world, 'get_actors', []) or [])
        lights = [a for a in actors if str(getattr(a, 'type_id', '')).startswith('traffic.traffic_light')]
        self.observe_signals(lights, native_frame)
        signals, nearby = [], []
        for actor in actors:
            actor_state = snapshot_state(snapshot, actor)
            actor_tf = call(actor_state, 'get_transform')
            actor_loc = getattr(actor_tf, 'location', None) if actor_tf is not None else call(actor_state, 'get_location')
            dist = distance(ego_loc, actor_loc)
            type_id = str(getattr(actor, 'type_id', ''))
            if dist is None or dist > 60 or getattr(actor, 'id', None) == getattr(ego, 'id', None):
                continue
            if type_id.startswith(('vehicle.', 'walker.pedestrian.')):
                nearby.append({'id': actor.id, 'type_id': type_id, 'world_position': vector(actor_loc), 'distance_m': dist,
                               'velocity_m_s': vector(call(actor_state, 'get_velocity'))})
        for light in lights:
            sid = light.id
            loc = call(light, 'get_location')
            dist = distance(ego_loc, loc)
            if (dist is None or dist > 100) and sid != affecting_id:
                continue
            if sid not in self.signal_static:
                stops = call(light, 'get_stop_waypoints')
                affected = call(light, 'get_affected_lane_waypoints')
                self.signal_static[sid] = {'stop_waypoints_world': None if stops is None else [waypoint(p) for p in stops],
                                          'affected_lane_waypoints_world': None if affected is None else [waypoint(p) for p in affected],
                                          'opendrive_id': call(light, 'get_opendrive_id')}
            record = dict(self.signal_static[sid])
            record.update(self.signal_states[sid])
            record.update(id=sid, affects_ego=(sid == affecting_id) if signal_lookup_available else None,
                          world_position=vector(loc), distance_m=dist, visible_in_camera=None)
            signals.append(record)
        weather = call(world, 'get_weather')
        weather_values = {key: number(getattr(weather, key, None)) for key in (
            'cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle',
            'sun_altitude_angle', 'fog_density', 'fog_distance', 'wetness', 'fog_falloff', 'scattering_intensity',
            'mie_scattering_scale', 'rayleigh_scattering_scale')}
        queued = list(getattr(planner, '_waypoints_queue', []) or [])[:30]
        road = {'road_id': getattr(wp, 'road_id', None), 'section_id': getattr(wp, 'section_id', None),
                'lane_id': getattr(wp, 'lane_id', None), 'lane_type': enum_name(getattr(wp, 'lane_type', None)),
                'lane_width_m': number(getattr(wp, 'lane_width', None)), 'lane_change': enum_name(getattr(wp, 'lane_change', None)),
                'left_lane_marking': lane_marking(getattr(wp, 'left_lane_marking', None)),
                'right_lane_marking': lane_marking(getattr(wp, 'right_lane_marking', None)),
                'is_junction': getattr(wp, 'is_junction', None), 'junction_id': getattr(wp, 'junction_id', None),
                'speed_limit_kmh': number(call(ego, 'get_speed_limit'))}
        return {'schema_version': 2, 'source': 'carla-0.9.15',
                'synchronization': {'sequence_id': '{}/{}'.format(Path(scene).parent.name, Path(scene).name), 'frame_id': str(frame_id), 'carla_frame': native_frame,
                    'timestamp_s': number(getattr(getattr(snapshot, 'timestamp', None), 'elapsed_seconds', None)),
                    'sensors': {name: dict({k: p[k] for k in ('frame', 'timestamp_s', 'world_transform')}, aligned=p['frame'] == native_frame) for name, p in packets.items()},
                    'aligned': all(p['frame'] == native_frame for p in packets.values())},
                'ego': {'world_position': vector(ego_loc), 'rotation_deg': rotation(getattr(ego_tf, 'rotation', None)),
                        'state_source': 'WorldSnapshot' if callable(getattr(snapshot, 'find', None)) else 'legacy live actor',
                        'velocity_m_s': vector(call(ego_state, 'get_velocity')), 'acceleration_m_s2': vector(call(ego_state, 'get_acceleration')),
                        'angular_velocity_deg_s': vector(call(ego_state, 'get_angular_velocity')), 'gear': getattr(current_ctrl, 'gear', None),
                        'reverse': getattr(current_ctrl, 'reverse', None), 'hand_brake': getattr(current_ctrl, 'hand_brake', None),
                        'applied_control': control(applied_control), 'applied_control_semantics': 'command submitted after this frame; affects a subsequent physics tick'},
                'road': road,
                'navigation': {'road_option': enum_name(getattr(planner, 'target_road_option', None)),
                    'incoming_road_option': enum_name(getattr(agent, '_incoming_direction', None)),
                    'target_speed_kmh': number(getattr(planner, '_target_speed', None)),
                    'target_waypoint_world': waypoint(getattr(planner, 'target_waypoint', None)),
                    'local_waypoints_world': [dict(waypoint(p), road_option=enum_name(option)) for p, option in queued],
                    'global_route_file': self.route_reference(agent, scene),
                    'source': 'CARLA global route with monotonic local progress; not future observed ground truth'},
                'traffic': {'affecting_signal_id': affecting_id, 'affecting_signal_lookup_available': signal_lookup_available, 'signals': signals, 'stop_line_distance_m': None,
                    'provenance': 'affects_ego=Vehicle.get_traffic_light; stop waypoints=TrafficLight.get_stop_waypoints; distance=Euclidean ego to pole in metres; camera visibility unknown; no inferred route-stopline match'},
                'environment': {'map_name': getattr(carla_map, 'name', None), 'weather': weather_values,
                                'nearby_actors': nearby, 'nearby_radius_m': 60.0},
                'coordinates_file': 'collection_manifest.json', 'calibration_file': 'sensor_config.yaml'}
