"""CARLA observations for the pinned LRS ``loop.py`` inference profile.

Reference: junwoomin/LRS @ 6eaf6c38450efc540f5fb1c86ed2980dd96a59bc.
This is intentionally the executed offline LRS profile, NOT the different
``roach/config/config_agent.yaml`` training configuration. See ``contract()``.
No simulator physics, policy weights, or visualization BEV enters this module.
"""
from collections import deque
from collections.abc import Mapping
import copy
import math

import numpy as np

from sdv.bev.geometry import BEVGeometry, local_to_world, warp_global, world_to_local


REFERENCE_COMMIT = '6eaf6c38450efc540f5fb1c86ed2980dd96a59bc'
HISTORY_INDICES = (-1, -5, -10, -15)
POLICY_PPM = 5.0
FIXED_DELTA_SECONDS = 1.0 / 30.0
GEOMETRY = BEVGeometry(192, 192, .2, .5, 136.0 / 192.0)
TL_VALUES = {'green': 80, 'yellow': 170, 'red': 255}


class ObservationError(RuntimeError):
    """A policy observation cannot be trusted; caller must apply full brake."""


def contract():
    return {
        'profile': 'lrs_loop', 'reference_commit': REFERENCE_COMMIT,
        'birdview_shape': [15, 192, 192], 'birdview_dtype': 'uint8',
        'birdview_range': [0, 255], 'state_shape': [6], 'policy_ppm': 5,
        'state_fields': ['throttle', 'steer', 'brake', 'lrs_gear', 'ego_velocity_x_m_s', 'ego_velocity_y_m_s'],
        'channel_order': ['road', 'route', 'lane'] +
            ['{}_{}'.format(kind, age) for kind in ('vehicle', 'pedestrian', 'traffic_light') for age in HISTORY_INDICES],
        'history_indices': list(HISTORY_INDICES), 'history_frame_lags': [0, 4, 9, 14],
        'fixed_delta_seconds': FIXED_DELTA_SECONDS,
        'history_seconds': [0, 4/30., 9/30., 14/30.],
        'ego_anchor_pixels': [96, 136], 'orientation': 'ego_forward_up_ego_right_right',
        'normalization': '0..255 retained; original PpoPolicy divides by 255',
        'traffic_light_values': dict(TL_VALUES), 'route_width_pixels': 12,
        'traffic_stop_line_width_pixels': 6,
        'gear_mapping': 'LRS P=0,R=1,N=2,D=3; CARLA automatic forward -> D; reverse -> R; manual neutral -> N',
        'actor_rasterization': 'LRS fixed vehicle 4.69x2m template; pedestrian 9x9 pixels; actual CARLA snapshot center/yaw',
        'history_coordinate_frame': 'all historical world states reprojected into CURRENT ego frame',
        'cold_start': 'unavailable actor history zero; traffic history repeats first state, as LRS',
        'source_config_differences': {
            'config_agent.yaml': {'history_idx': [-16, -11, -6, -1], 'ego_anchor_pixels': [96, 152]},
            'rl_birdview_wrapper.py': 'gear / 5; NOT the executed loop.py categorical gear',
            'loop.py': 'raw categorical gear; newest-first history; move_px=40 from image center',
            'dtype': 'LRS casts masks to float32; adapter retains exact uint8 values until policy conversion',
        },
        'training_contract_equivalence_verified': False,
    }


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ObservationError('{} must be finite'.format(name))
    if not math.isfinite(number):
        raise ObservationError('{} must be finite'.format(name))
    return number


def _vector(value, name):
    if isinstance(value, Mapping):
        value = [value.get(axis) for axis in ('x', 'y', 'z')]
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ObservationError('{} must contain finite xyz'.format(name))
    return result


def vehicle_state(ego, previous_control):
    """Same rotation as LRS vec_global_to_ref, including actual pitch/roll."""
    control = previous_control if isinstance(previous_control, Mapping) else vars(previous_control)
    values = [_finite(control.get(key), key) for key in ('throttle', 'steer', 'brake')]
    if not (0 <= values[0] <= 1 and -1 <= values[1] <= 1 and 0 <= values[2] <= 1):
        raise ObservationError('Previous applied control outside CARLA bounds')
    if bool(control.get('reverse', False)) or _finite(control.get('gear', 0), 'gear') < 0:
        gear = 1
    elif control.get('manual_gear_shift') and control.get('gear') == 0:
        gear = 2
    else:
        gear = 3
    rotation = ego.get('rotation_deg') or {'yaw': ego['yaw'], 'pitch': 0, 'roll': 0}
    roll, pitch, yaw = [math.radians(_finite(rotation.get(key, 0), key)) for key in ('roll', 'pitch', 'yaw')]
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    yaw_matrix = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    pitch_matrix = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    roll_matrix = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    velocity = (yaw_matrix.dot(pitch_matrix).dot(roll_matrix)).T.dot(_vector(ego.get('velocity_m_s'), 'velocity'))
    return np.asarray(values + [gear, velocity[0], velocity[1]], dtype=np.float32)


class RoachObservationAdapter:
    def __init__(self, global_map, fixed_delta_seconds=FIXED_DELTA_SECONDS):
        if not math.isclose(float(fixed_delta_seconds), FIXED_DELTA_SECONDS, rel_tol=0, abs_tol=1e-8):
            raise ObservationError('lrs_loop requires synchronous 30 Hz ticks; history must retain its physical timing')
        if float(global_map.ppm) not in (5., 10.):
            raise ObservationError('Roach map input requires a verified 5 or 10 PPM global map')
        self.global_map = global_map
        self.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.history = deque(maxlen=16)
        self.world = self.carla_map = None
        self.stop_geometry = {}
        self.last_ego_record = None
        self.last_metadata = None
        self._last_frame = self._last_timestamp = None
        self.map_metadata = dict(global_map.metadata)
        # Fail before driving for absent road/lane planes, including upper decks.
        for layer in global_map.metadata.get('height_layers', ['low']):
            selected = global_map.select_layer({'x': 0, 'y': 0, 'z': 0}, policy=layer,
                height_aware=True, channels=['drivable', 'lane'])
            if any(key not in selected['arrays'] for key in ('drivable', 'lane')):
                raise ObservationError('Policy road/lane mask missing from {} layer'.format(layer))

    @classmethod
    def from_carla(cls, world, cache_root=None, bev_settings=None, fixed_delta_seconds=FIXED_DELTA_SECONDS):
        from sdv.bev.global_map import load_global_map
        from sdv.collection.bev_adapter import resolved_map_root
        from sdv.bev.schema import normalize_bev_settings
        settings = normalize_bev_settings(bev_settings)
        carla_map = world.get_map()
        town = carla_map.name.rsplit('/', 1)[-1]
        global_map = load_global_map(resolved_map_root(settings), town, settings['global_ppm'])
        adapter = cls(global_map, fixed_delta_seconds)
        adapter.world, adapter.carla_map = world, carla_map
        return adapter

    def reset(self):
        self.history.clear()
        self._last_frame = self._last_timestamp = None

    def observe_carla(self, ego, snapshot, route_xyz, previous_control):
        from sdv.collection.bev_adapter import actor_record, CarlaBEVAdapter
        if self.world is None:
            raise ObservationError('Use from_carla() before observe_carla()')
        settings = self.world.get_settings()
        if not settings.synchronous_mode or not math.isclose(float(settings.fixed_delta_seconds or 0), self.fixed_delta_seconds, abs_tol=1e-8):
            raise ObservationError('CARLA world must be synchronous with fixed_delta_seconds=1/30')
        actors = list(self.world.get_actors())
        records, skipped = [], []
        ego_snapshot = snapshot.find(ego.id)
        if ego_snapshot is None:
            raise ObservationError('CARLA ego snapshot unavailable')
        ego_location = ego_snapshot.get_transform().location
        for actor in actors:
            if str(actor.type_id).startswith(('vehicle.', 'walker.pedestrian.')):
                record = actor_record(actor, snapshot, self.carla_map, actor.id == ego.id)
                if record is None:
                    state = snapshot.find(actor.id)
                    if state is not None:
                        at = state.get_transform().location
                        distance = math.hypot(at.x-ego_location.x, at.y-ego_location.y)
                        if math.isfinite(distance) and distance > 100.:
                            skipped.append({'actor_id': int(actor.id), 'distance_m': distance,
                                'reason': 'missing box data beyond 100m; outside policy physical view'})
                            continue
                    raise ObservationError('Actor {} lacks aligned snapshot/box state'.format(actor.id))
                records.append(record)
        own = next((record for record in records if record['is_ego']), None)
        if own is None:
            raise ObservationError('CARLA ego snapshot unavailable')
        self.last_ego_record = own
        traffic = CarlaBEVAdapter.traffic(self, actors)
        result = self.observe(own, records, traffic, route_xyz, previous_control,
            int(snapshot.frame), float(snapshot.timestamp.elapsed_seconds))
        result['metadata']['skipped_distant_actors'] = skipped
        # TrafficLight state is not part of ActorSnapshot; refuse a raced tick.
        if int(self.world.get_snapshot().frame) != int(snapshot.frame):
            raise ObservationError('CARLA frame advanced while collecting policy observation')
        return result

    def _pixel(self, xy):
        # sim.utility.world_to_pixel uses int(), i.e. truncation toward zero.
        value = np.asarray(xy, dtype=np.float64)
        if not np.isfinite(value).all() or value.shape[-1:] != (2,):
            raise ObservationError('Actor/route world XY must be finite')
        return np.trunc((value - self.global_map.world_offset) * POLICY_PPM).astype(np.int64)

    def _quantized_pose(self, ego):
        result = dict(ego)
        xy = self._pixel([ego['x'], ego['y']]) / POLICY_PPM + self.global_map.world_offset
        result.update(x=float(xy[0]), y=float(xy[1]))
        return result

    def _layer(self, item):
        if not item.get('ground_verified') or item.get('ground_z_m') is None:
            raise ObservationError('Actor ground height is not verified by native CARLA map')
        return self.global_map.select_layer(item, policy='hblr', height_aware=True, channels=[])['layer_id']

    def _roi(self, pose):
        corners = local_to_world(np.array([[0, 0], [191, 0], [191, 191], [0, 191]]), pose, GEOMETRY)
        pixels = (corners - self.global_map.world_offset) * POLICY_PPM
        origin = np.floor(pixels.min(axis=0)).astype(int) - 20
        finish = np.ceil(pixels.max(axis=0)).astype(int) + 20
        shape = tuple((finish - origin + 1)[::-1])
        return origin, shape

    def _warp_roi(self, mask, origin, pose):
        return warp_global(mask, pose, GEOMETRY, self.global_map.world_offset + origin / POLICY_PPM, .2)

    def _actor_masks(self, records, ego, pose, layer, origin, shape):
        import cv2
        vehicle = np.zeros(shape, dtype=np.uint8)
        pedestrian = np.zeros(shape, dtype=np.uint8)
        excluded, uncertain = [], []
        ego_id = ego.get('actor_id', ego.get('id'))
        for item in records:
            identity = item.get('actor_id', item.get('id'))
            if identity == ego_id:
                continue
            center = self._pixel([item['x'], item['y']]) - origin
            if np.any(center < -30) or center[0] > shape[1] + 30 or center[1] > shape[0] + 30:
                continue
            known_ground = item.get('ground_verified') is True and item.get('ground_z_m') is not None
            if known_ground:
                item_layer = self._layer(item)
                item_height = float(item['ground_z_m'])
            else:
                # Pedestrians may occupy native sidewalks/lawns absent from
                # OpenDRIVE. Their real snapshot-transformed box bottom is an
                # estimate, not a verified road surface. Keep uncertain
                # same-height occupancy rather than erase the obstacle or
                # fabricate a native road affiliation.
                contact = _vector(item.get('ground_contact_world_xyz'), 'actor bbox bottom-face contact')
                item_height = float(contact[2])
                item_layer = layer
                uncertain.append(identity)
            # HBLR separates named decks; local height additionally protects
            # single-layer maps and pedestrians. Unknown nearby heights fail.
            if item_layer != layer or abs(item_height - float(ego['ground_z_m'])) > 4.:
                excluded.append(identity)
                continue
            if item.get('object_class') == 'pedestrian':
                x, y = map(int, center)
                cv2.rectangle(pedestrian, (x-4, y-4), (x+4, y+4), 255, -1)
            elif item.get('object_class') in ('vehicle', 'cyclist'):
                yaw = math.radians(float(item['yaw']))
                c, s = math.cos(yaw), math.sin(yaw)
                points = np.array([[12, 5], [12, -5], [-12, -5], [-12, 5]], dtype=np.float32)
                points = points.dot(np.array([[c, -s], [s, c]], dtype=np.float32).T) + center
                cv2.fillConvexPoly(vehicle, np.round(points).astype(np.int32), 255)
        return self._warp_roi(vehicle, origin, pose), self._warp_roi(pedestrian, origin, pose), excluded, uncertain

    def _traffic_mask(self, signals, pose, layer, origin, shape):
        import cv2
        output = np.zeros(shape, dtype=np.uint8)
        for signal in signals:
            state = str(signal.get('state', 'unknown')).rsplit('.', 1)[-1].lower()
            if state not in TL_VALUES:
                # Off/unknown signals carry no invented traffic-light color.
                continue
            for segment in signal.get('segments', []):
                points = np.asarray(segment, dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ObservationError('Invalid traffic stop-line XYZ')
                for start, end in zip(points[:-1], points[1:]):
                    mid = (start+end)/2
                    info = dict(signal, x=mid[0], y=mid[1], z=mid[2], ground_z_m=mid[2])
                    if self._layer(info) != layer or abs(mid[2]-float(pose['ground_z_m'])) > 4.:
                        continue
                    pixels = np.rint((np.array([start[:2], end[:2]]) - self.global_map.world_offset)*POLICY_PPM).astype(np.int64) - origin
                    # LRS paints world-grid signal lines with LINE_AA then
                    # nearest-warps. Retain its edge intensities exactly.
                    cv2.line(output, tuple(pixels[0]), tuple(pixels[1]), TL_VALUES[state], 6, cv2.LINE_AA)
        return self._warp_roi(output, origin, pose)

    def _route(self, route_xyz, pose, layer):
        import cv2
        points = np.asarray(route_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ObservationError('A nonempty finite CARLA route Nx3 is required')
        output = np.zeros((192, 192), dtype=np.uint8)
        xy = self._pixel(points[:, :2]) / POLICY_PPM + self.global_map.world_offset
        local = np.rint(world_to_local(xy, pose, GEOMETRY)).astype(np.int32)
        # Route provided by current CARLA local planner; keep LRS width/point
        # rounding while cutting any segment that changes height layer.
        selected = self.global_map.layer_for_points(points, ground_verified=True, lane_type='Driving')['high']
        valid = ((selected if layer == 'high' else ~selected) &
            (np.abs(points[:, 2]-float(pose['ground_z_m'])) <= 4.))
        # LRS keeps in-canvas points, rounds their pixels, and draws a 12px
        # polyline (or a 6px-radius single point). Split first at opposite
        # decks so filtering cannot connect across an excluded overpass.
        runs, run = [], []
        for pixel, keep in zip(local, valid):
            if not keep:
                if run:
                    runs.append(run)
                    run = []
            elif np.all(pixel >= 0) and np.all(pixel < 192):
                run.append(pixel)
        if run:
            runs.append(run)
        for run in runs:
            if len(run) >= 2:
                cv2.polylines(output, [np.asarray(run,dtype=np.int32).reshape(-1,1,2)], False, 255, 12)
            elif len(run) == 1:
                cv2.circle(output, tuple(run[0]), 6, 255, -1)
        return output

    def observe(self, ego_record, actor_records, traffic_records, route_xyz, previous_control, frame_id, timestamp):
        frame_id, timestamp = int(frame_id), _finite(timestamp, 'timestamp')
        if self._last_frame is not None and (frame_id != self._last_frame+1 or
            not math.isclose(timestamp-self._last_timestamp, self.fixed_delta_seconds, abs_tol=1e-4)):
            raise ObservationError('Nonconsecutive CARLA frame/timestamp: policy history cannot be aligned')
        ego = copy.deepcopy(ego_record)
        for key in ('x', 'y', 'z', 'yaw'):
            _finite(ego.get(key), key)
        if ego.get('frame_id', frame_id) != frame_id or any(record.get('frame_id', frame_id) != frame_id for record in actor_records):
            raise ObservationError('Actor record frame differs from observation frame')
        layer = self._layer(ego)
        pose = self._quantized_pose(ego)
        selected = self.global_map.select_layer(ego, policy='hblr', height_aware=True, channels=['drivable', 'lane'])
        masks = [warp_global(selected['arrays'][key], pose, GEOMETRY, self.global_map.world_offset,
            self.global_map.meters_per_pixel).astype(np.uint8)*255 for key in ('drivable', 'lane')]
        state = vehicle_state(ego, previous_control)
        # Keep records in world coordinates, not historical ego-local images.
        current = {'frame_id': frame_id, 'actors': copy.deepcopy(list(actor_records)), 'traffic': copy.deepcopy(list(traffic_records))}
        frames = list(self.history) + [current]
        frames = frames[-16:]
        origin, shape = self._roi(pose)
        vehicles, pedestrians, signals, ids, excluded, uncertain = [], [], [], [], [], []
        zero = np.zeros((192, 192), dtype=np.uint8)
        for index in HISTORY_INDICES:
            if len(frames) >= -index:
                past = frames[index]
                vehicle, pedestrian, removed, provisional = self._actor_masks(past['actors'], ego, pose, layer, origin, shape)
                ids.append(past['frame_id'])
                excluded.extend(removed)
                uncertain.extend(provisional)
            else:
                vehicle, pedestrian = zero.copy(), zero.copy()
                ids.append(None)
            signal_frame = frames[max(0, len(frames)+index)]
            vehicles.append(vehicle)
            pedestrians.append(pedestrian)
            signals.append(self._traffic_mask(signal_frame['traffic'], pose, layer, origin, shape))
        birdview = np.stack([masks[0], self._route(route_xyz, pose, layer), masks[1]] + vehicles + pedestrians + signals)
        if birdview.shape != (15, 192, 192) or birdview.dtype != np.uint8 or not np.isfinite(state).all():
            raise ObservationError('Invalid Roach observation shape or values')
        self.history.append(current)
        self._last_frame, self._last_timestamp = frame_id, timestamp
        self.last_ego_record = ego
        metadata = dict(contract(), frame_id=frame_id, timestamp=timestamp, history_frames=ids,
            history_ready=len(frames) >= 15, history_samples=len(frames), selected_height_layer=layer,
            source_map_ppm=float(self.global_map.ppm), map_source_signature=self.global_map.metadata.get('source_signature'),
            excluded_height_actor_ids=sorted(set(excluded)), ground_height_source=ego.get('ground_source'),
            uncertain_ground_actor_ids=sorted(set(uncertain)),
            uncertain_ground_policy='Retain nearby same-height bbox-bottom occupancy without claiming native ground/layer verification',
            map_metric_elevation_available=False, interpolation='nearest_neighbor',
            height_limitations='Supplied high/low PNG selection plus native ground checks; no per-pixel metric height map supplied',
            traffic_source='CARLA lane-specific TrafficLight.get_stop_waypoints and live state inside synchronous tick')
        self.last_metadata = metadata
        return {'birdview': birdview, 'state': state, 'metadata': metadata}
