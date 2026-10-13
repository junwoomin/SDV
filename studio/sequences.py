"""Frame-aligned, scene-scoped inspection of actual SDV collection files.

No CARLA, OpenCV or torch import is required. File names are sample identifiers,
not simulator frame numbers. Missing modalities remain missing; no nearest-frame
substitution is performed. New metadata is preferred, legacy fields keep their
recorded meaning and provenance.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import io
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlencode

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .catalog import CAMERAS, IMAGE_SUFFIXES, read_text, safe_child
from .bev_preview import read_bev, describe as describe_bev, render_bev, CHANNELS as BEV_CHANNELS
from sdv.data_contract import (BEV_LAYERS, OD_CLASSES, OD_COLORS,
    decode_segmentation, colorize_segmentation, decode_depth,
    depth_encoding_for_scene, semantic_records)

MAX_PIXELS = 16 * 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024
LAYER_COLORS = {'das': '#334155', 'lane': '#f8fafc', 'vehicle': '#3b82f6',
                'walker': '#fb923c', 'traffic_light': '#f43f5e',
                'stopline': '#facc15', 'route': '#22d3ee'}


def natural_key(value):
    return tuple((0, int(s)) if s.isdigit() else (1, s.lower())
                 for s in re.split(r'(\d+)', str(value)))


def _finite(value):
    if isinstance(value, bool): return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError): return None


def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def read_json(path, limit=4 * 1024 * 1024):
    value = json.loads(read_text(path, limit))
    if not isinstance(value, dict): raise ValueError('JSON metadata must be an object')
    return value


def read_array(path):
    """Reject object pickle, malformed shapes and oversized image arrays."""
    path = Path(path)
    if path.stat().st_size > MAX_ARRAY_BYTES:
        raise ValueError('Array exceeds the preview size limit')
    if path.suffix.lower() == '.npy':
        array = np.load(str(path), allow_pickle=False, mmap_mode='r')
        if array.dtype.hasobject or array.size > MAX_PIXELS * 4 or array.nbytes > MAX_ARRAY_BYTES:
            raise ValueError('Unsupported array dtype or dimensions')
        return np.asarray(array)
    with Image.open(str(path)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError('Image exceeds the preview size limit')
        image.load()
        # P-mode semantic labels contain palette indexes, not raw class IDs.
        if image.mode == 'P': image = image.convert('RGB')
        return np.asarray(image).copy()


def bbox_annotations(path, width=None, height=None):
    array = read_array(path)
    if array.size == 0:
        if array.shape not in ((0,), (0, 5)): raise ValueError('Empty boxes must have shape (0,) or (0,5)')
        return []
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError('Bounding boxes must be N × 5: x1, y1, x2, y2, class_id')
    result = []
    for row in array:
        if not np.isfinite(row).all(): raise ValueError('Bounding box contains a non-finite value')
        x1, y1, x2, y2, category = [float(x) for x in row]
        if category != int(category) or int(category) not in OD_CLASSES:
            raise ValueError('Unknown object class ID: {}'.format(category))
        if x2 <= x1 or y2 <= y1: raise ValueError('Bounding box has non-positive area')
        outside = (x1 < 0 or y1 < 0 or
                   (width is not None and x2 > width) or
                   (height is not None and y2 > height))
        result.append({'class_id': int(category), 'class_name': OD_CLASSES[int(category)],
                       'xyxy': [x1, y1, x2, y2], 'coordinate_system': 'image_pixels',
                       'type': '2d', 'object_id': None, 'outside_image': outside})
    return result


def normalize_scene(raw, manifest, frame_key):
    raw = raw or {}
    extension = raw.get('sdv_metadata') or {}
    if not isinstance(extension, dict): extension = {}
    def part(name):
        value = extension.get(name)
        return value if isinstance(value, dict) else {}
    control = raw.get('current_control') or {}
    if not isinstance(control, dict): control = {}
    rotation = raw.get('ego_vehicle_rot') or {}
    if not isinstance(rotation, dict): rotation = {}
    ego_extra, nav, road, traffic, env, sync = [part(k) for k in
        ('ego', 'navigation', 'road', 'traffic', 'environment', 'synchronization')]
    ego = {'speed_kmh': raw.get('current_speed'), 'position_m': raw.get('ego_vehicle_loc'),
           'rotation_deg': rotation or None, 'yaw_deg': rotation.get('yaw'),
           'steering': control.get('steer'), 'throttle': control.get('throttle'),
           'brake': control.get('brake'), 'gear': ego_extra.get('gear'),
           'velocity_m_s': ego_extra.get('velocity_m_s'),
           'acceleration_m_s2': ego_extra.get('acceleration_m_s2'),
           'angular_velocity_deg_s': ego_extra.get('angular_velocity_deg_s'),
           'reverse': ego_extra.get('reverse'), 'hand_brake': ego_extra.get('hand_brake'),
           'imu': raw.get('imu'), 'source': 'info JSON; sdv_metadata for extended fields'}
    road_fields = ('road_id', 'section_id', 'lane_id', 'lane_type', 'lane_width_m',
                   'lane_change', 'left_lane_marking', 'right_lane_marking',
                   'speed_limit_kmh', 'is_junction', 'junction_id')
    road = dict({key: road.get(key) for key in road_fields}, source='sdv_metadata.road' if road else None)
    navigation = {key: nav.get(key) for key in ('road_option', 'incoming_road_option',
        'target_speed_kmh', 'target_waypoint_world', 'local_waypoints_world',
        'global_route_file', 'route_version', 'distance_to_route_m',
        'route_completion_pct', 'distance_to_goal_m', 'destination_reached', 'route_id', 'target_point', 'traffic_light_state',
        'remaining_distance_m', 'route_length_m', 'active_driving_agent')}
    navigation.update({'current_waypoint_world': raw.get('current_waypoint'),
        'route_command': nav.get('road_option'), 'upcoming_maneuver': nav.get('incoming_road_option'),
        'planned_route_bev_pixels': raw.get('global_waypoints'),
        'planned_local_waypoints_bev_pixels': raw.get('bev_future_waypoints'),
        'next_control': raw.get('next_control'),
        'future_ground_truth': None,
        'source': nav.get('source') or 'Legacy planner outputs are online plans, not future trajectory ground truth'})
    signals = traffic.get('signals')
    if not isinstance(signals, list): signals = None
    affecting = next((s for s in signals or [] if isinstance(s, dict) and s.get('affects_ego') is True), None)
    traffic_view = dict(traffic)
    traffic_view.update({'signals': signals, 'signal_state': affecting.get('state') if affecting else None,
        'affecting_signal_id': traffic.get('affecting_signal_id'),
        'signal_distance_m': affecting.get('distance_m') if affecting else None,
        'stop_line_distance_m': traffic.get('stop_line_distance_m'),
        'legacy_red_light_hazard': raw.get('traffic_light'),
        'legacy_traffic_light_20': raw.get('traffic_light_20'),
        'legacy_traffic_light_distance_m': raw.get('traffic_light_distance'),
        'vehicle_in_front': raw.get('vehicle_in_front'), 'collision': raw.get('collision'),
        'source': 'sdv_metadata.traffic' if traffic else 'Legacy hazard flags do not identify a signal state or lane applicability'})
    nearby = env.get('nearby_actors')
    traffic_view['nearby_vehicles'] = sum(str(a.get('type_id', '')).startswith('vehicle.') for a in nearby if isinstance(a, dict)) if isinstance(nearby, list) else None
    traffic_view['nearby_pedestrians'] = sum(str(a.get('type_id', '')).startswith('walker.pedestrian.') for a in nearby if isinstance(a, dict)) if isinstance(nearby, list) else None
    environment = {'map_name': env.get('map_name', manifest.get('town')),
        'weather': env.get('weather'), 'weather_setting': manifest.get('weather'),
        'nearby_actors': env.get('nearby_actors'),
        'source': 'sdv_metadata.environment' if env else 'collection_manifest (map/setting only)'}
    synchronization = {'sequence_id': sync.get('sequence_id', manifest.get('scene')),
        'sample_key': frame_key, 'frame_id': sync.get('carla_frame'),
        'timestamp_s': sync.get('timestamp_s'), 'sensors': sync.get('sensors'),
        'aligned': sync.get('aligned'), 'source': 'sdv_metadata.synchronization' if sync else None}
    bev = raw.get('bev_gt')
    bev = bev if isinstance(bev, dict) else {}
    ego_pose = bev.get('ego_pose')
    ego_pose = ego_pose if isinstance(ego_pose, dict) else {}
    position = ego.get('position_m')
    position = position if isinstance(position, dict) else {}
    global_map = bev.get('global_map')
    global_map = global_map if isinstance(global_map, dict) else {}
    geometry = bev.get('geometry')
    geometry = geometry if isinstance(geometry, dict) else {}
    bev_view = {'ego_z_m': ego_pose.get('z', position.get('z')),
        'ground_z_m': bev.get('ground_z_m'), 'map_layer_id': bev.get('map_layer_id'),
        'road_id': bev.get('road_id'), 'lane_id': bev.get('lane_id'),
        'geometry': bev.get('geometry'), 'schema_version': bev.get('schema_version'),
        'actors': bev.get('actors') if isinstance(bev.get('actors'), list) else None,
        'generation_ms': bev.get('generation_ms'), 'valid_surface_fraction': bev.get('valid_surface_fraction'),
        'layer_policy': bev.get('layer_policy'),
        'global_map': global_map or None,
        'map_source': global_map.get('source_path'), 'native_ppm': global_map.get('native_ppm'),
        'selected_height_layer': global_map.get('selected_height_layer'),
        'orientation': geometry.get('orientation'),
        'provenance': bev.get('provenance', bev.get('label_source')),
        'source': 'info.bev_gt' if bev else None}
    policy = dict(part('policy'))
    if policy:
        policy.setdefault('policy_ppm', policy.get('policy_input_ppm'))
        policy.setdefault('requested_control', policy.get('raw_vehicle_control'))
        observation_metadata = policy.get('observation_metadata') or {}
        if isinstance(observation_metadata, dict):
            policy.setdefault('observation_profile', observation_metadata.get('profile') or observation_metadata.get('observation_profile'))
    safety = dict(part('safety'))
    if not safety and policy:
        safety = {'intervened': policy.get('safety_intervened'), 'reason': policy.get('safety_reason'),
                  'requested_control': policy.get('raw_vehicle_control'), 'applied_control': policy.get('applied_control')}
    return clean_json({'ego': ego, 'road': road, 'navigation': navigation, 'bev': bev_view,
                      'traffic': traffic_view, 'environment': environment,
                      'synchronization': synchronization, 'policy': policy,
                      'safety': safety, 'evaluation': part('driving_metrics') or part('evaluation')})


class SequenceCatalog:
    def __init__(self, catalog):
        self.catalog = catalog
        self._cache = OrderedDict()
        self._images = OrderedDict()
        self._image_bytes = 0
        self._info_cache = OrderedDict()
        self._info_bytes = 0
        self._lock = threading.RLock()

    def invalidate(self):
        with self._lock:
            self._cache.clear()
            self._images.clear()
            self._info_cache.clear()
            self._image_bytes=0
            self._info_bytes=0

    def _read_info(self, path):
        """Bounded per-file cache survives live inventory rebuilds.

        Paths have already passed scene-scoped safe_child. Cached dictionaries
        are read-only internal values; public frame responses copy raw_info.
        The source byte budget is 8 MiB with at most 8,000 records, so playback
        of long scenes cannot retain unbounded metadata.
        """
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        key = str(path)
        with self._lock:
            cached = self._info_cache.get(key)
            if cached and cached[0] == signature:
                self._info_cache.move_to_end(key)
                return cached[1]
        value = read_json(path)
        with self._lock:
            previous = self._info_cache.pop(key, None)
            if previous is not None: self._info_bytes -= previous[0][0]
            self._info_cache[key] = (signature, value)
            self._info_bytes += stat.st_size
            while self._info_bytes > 8 * 1024 * 1024 or len(self._info_cache) > 8000:
                _, discarded = self._info_cache.popitem(last=False)
                self._info_bytes -= discarded[0][0]
        return value

    def _inventory(self, identifier):
        scene = self.catalog.scene_path(identifier)
        with self._lock:
            cached = self._cache.get(identifier)
            if cached and time.monotonic() - cached[0] < 1:
                self._cache.move_to_end(identifier)
                return cached[1]
        previous_frames = {f['frame_key']: f for f in cached[1]['frames']} if cached else {}
        channels, rows, warnings = {}, {}, []
        for folder in sorted(scene.iterdir()):
            if not folder.is_dir() or folder.is_symlink(): continue
            if not (folder.name in {'info', 'bev', 'bev_gt', 'lidar'} or
                    folder.name in {v['folder'] for v in BEV_LAYERS.values()} or
                    any(folder.name == camera + suffix for camera in CAMERAS
                        for suffix in ('_rgb', '_seg', '_seg_ids', '_depth', '_depth_m', '_bbs'))): continue
            files = {}
            for file in folder.iterdir():
                if (file.is_file() and not file.is_symlink() and
                        file.suffix.lower() in IMAGE_SUFFIXES | {'.npy', '.npz', '.json'}):
                    if file.stem in files:
                        warnings.append('Duplicate sample key: {}/{}'.format(folder.name, file.stem))
                        continue
                    if file.stat().st_size:
                        files[file.stem] = str(file.relative_to(scene))
                        rows.setdefault(file.stem, {})[folder.name] = files[file.stem]
            if files: channels[folder.name] = files
        manifest, calibration = {}, None
        for filename in ('collection_manifest.json', 'sensor_config.yaml'):
            path = safe_child(scene, filename)
            if not path.is_file() or path.is_symlink(): continue
            try:
                if filename.endswith('.json'): manifest = read_json(path)
                else:
                    calibration = yaml.safe_load(read_text(path, 2 * 1024 * 1024))
                    if not isinstance(calibration, dict): calibration = None
            except (OSError, ValueError, yaml.YAMLError) as error: warnings.append(str(error))
        frames = []
        for key in sorted(rows, key=natural_key):
            files = rows[key]
            raw, error, info_signature = {}, None, None
            if 'info' in files:
                try:
                    info_path = safe_child(scene, files['info'])
                    info_stat = info_path.stat()
                    info_signature = (info_stat.st_size, info_stat.st_mtime_ns)
                    previous = previous_frames.get(key)
                    if previous and previous.get('_info_signature') == info_signature:
                        # Keep only these small temporal fields in the index.
                        # Even scenes larger than the JSON LRU need no full
                        # reread on every live polling interval.
                        frames.append(dict(previous, files=files, missing=sorted(set(channels)-set(files))))
                        continue
                    raw = self._read_info(info_path)
                except (OSError, ValueError) as caught: error = str(caught)
            extension = raw.get('sdv_metadata') or {}
            if not isinstance(extension, dict):
                error = 'sdv_metadata must be an object'
                extension = {}
            sync = extension.get('synchronization') or {}
            if not isinstance(sync, dict):
                error = 'synchronization must be an object'
                sync = {}
            frames.append({'frame_key': key, 'files': files,
                'frame_id': sync.get('carla_frame'), 'timestamp': _finite(sync.get('timestamp_s')),
                'missing': sorted(set(channels) - set(files)), 'metadata_error': error,
                '_info_signature': info_signature})
        # Only actual timestamps can override natural sample-ID order.
        if frames and all(f['timestamp'] is not None for f in frames):
            frames.sort(key=lambda f: (f['timestamp'], natural_key(f['frame_key'])))
        for i, frame in enumerate(frames): frame['index'] = i
        result = {'path': scene, 'frames': frames, 'channels': channels,
                  'manifest': manifest, 'calibration': calibration, 'warnings': warnings}
        with self._lock:
            self._cache[identifier] = time.monotonic(), result
            while len(self._cache) > 8: self._cache.popitem(last=False)
        return result

    def sequence(self, identifier):
        data = self._inventory(identifier)
        layers = [dict(v, id=k, label=v['name'], color=LAYER_COLORS[k], available=v['folder'] in data['channels'])
                  for k, v in BEV_LAYERS.items()]
        frames = [{k: value for k, value in f.items() if k not in ('files', 'raw_info', '_info_signature')}
                  for f in data['frames']]
        for frame, original in zip(frames, data['frames']): frame['channels'] = sorted(original['files'])
        manifest = data['manifest']
        bev_description = None
        # Numeric channels are explicitly declared per frame. This is a view
        # catalogue only; rendering still checks the selected frame itself.
        first_numeric = next((f for f in data['frames'] if 'bev_gt' in f['files']), None)
        if first_numeric:
            try:
                bev_description = describe_bev(read_bev(self._path(data, first_numeric, 'bev_gt'), self._info(data, first_numeric).get('bev_gt')),
                                               self._info(data, first_numeric).get('bev_gt'))
                layers = bev_description['layers']
            except (ValueError, OSError) as error:
                data = dict(data, warnings=data['warnings'] + ['BEV: ' + str(error)])
        return clean_json({'id': identifier, 'name': data['path'].name,
            'sequence_id': manifest.get('scene', data['path'].name), 'total': len(frames),
            'frames': frames, 'channels': sorted(data['channels']),
            'cameras': [c for c in CAMERAS if c + '_rgb' in data['channels']],
            'layers': layers, 'bev_views': bev_description['views'] if bev_description else [],
            'bev_gt': bev_description, 'calibration': data['calibration'],
            'manifest': manifest, 'coordinate_systems': manifest.get('coordinates') or {
                'world': 'CARLA left-handed: x forward, y right, z up; metres',
                'ego': 'x forward, y right, z up; metres',
                'camera': 'image u right, v down; pixel coordinates',
                'bev': ('SDV native renderer: 500 x 500, ego centre (250,250), 5 pixels/m; source: collector code'
                        if manifest.get('format') == 'sdv-native-masks-v1' else None)},
            'warnings': data['warnings'] + ([] if all(f['timestamp'] is not None for f in frames) else
                ['Simulator timestamps are unavailable for one or more samples; ordering uses recorded sample IDs.']),
            'ordering': 'timestamp' if frames and all(f['timestamp'] is not None for f in frames) else 'natural_sample_id'})

    def _frame(self, identifier, index):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError('Frame index must be a non-negative integer')
        data = self._inventory(identifier)
        if index >= len(data['frames']): raise FileNotFoundError('Frame index is outside this sequence')
        frame = dict(data['frames'][index])
        frame['raw_info'] = self._info(data, frame)
        return data, frame

    def _info(self, data, frame):
        relative = frame['files'].get('info')
        if relative is None: return {}
        try: return self._read_info(safe_child(data['path'], relative))
        except (OSError, ValueError): return {}

    def _path(self, data, frame, channel):
        relative = frame['files'].get(channel)
        if relative is None: raise FileNotFoundError('This sample has no {} data'.format(channel))
        path = safe_child(data['path'], relative)
        if path.is_symlink() or not path.is_file(): raise FileNotFoundError('Frame file is unavailable')
        return path

    def frame(self, identifier, index, camera='front_cam'):
        if camera not in CAMERAS: raise ValueError('Unknown camera')
        data, frame = self._frame(identifier, index)
        metadata, warnings = {}, []
        for channel in frame['files']:
            if channel == 'info': continue
            try:
                path = self._path(data, frame, channel)
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    with Image.open(str(path)) as image:
                        metadata[channel] = {'width': image.width, 'height': image.height, 'mode': image.mode}
            except (ValueError, OSError) as error: warnings.append('{}: {}'.format(channel, error))
        boxes = None
        if camera + '_bbs' in frame['files']:
            dimensions = metadata.get(camera + '_rgb', {})
            try: boxes = bbox_annotations(self._path(data, frame, camera + '_bbs'), dimensions.get('width'), dimensions.get('height'))
            except (ValueError, OSError) as error: warnings.append('Bounding boxes: ' + str(error))
        aliases = {'rgb': camera + '_rgb', 'seg': camera + '_seg_ids' if camera + '_seg_ids' in frame['files'] else camera + '_seg',
                   'depth': camera + '_depth_m' if camera + '_depth_m' in frame['files'] else camera + '_depth', 'od': camera + '_bbs'}
        image_urls = {key: '/api/datasets/{}/frame?{}'.format(identifier, urlencode({'index': index, 'channel': key, 'camera': camera}))
                      for key, folder in aliases.items() if folder in frame['files'] and (key != 'od' or camera + '_rgb' in frame['files'])}
        bev_description = None
        if 'bev_gt' in frame['files']:
            try:
                bev_description = describe_bev(read_bev(self._path(data, frame, 'bev_gt'), frame['raw_info'].get('bev_gt')), frame['raw_info'].get('bev_gt'))
                for view in bev_description['views']:
                    if view['available']:
                        image_urls[view['id']] = '/api/datasets/{}/frame?{}'.format(identifier, urlencode({'index': index, 'channel': view['id']}))
                metadata['bev_gt'] = bev_description
                for warning in (frame['raw_info'].get('bev_gt') or {}).get('warnings', []):
                    if isinstance(warning, str): warnings.append('BEV: ' + warning)
            except (ValueError, OSError) as error: warnings.append('BEV: ' + str(error))
        elif 'bev' in frame['files'] or any(v['folder'] in frame['files'] for v in BEV_LAYERS.values()):
            image_urls['bev'] = '/api/datasets/{}/frame?{}'.format(identifier, urlencode({'index': index, 'channel': 'bev'}))
        if aliases['depth'] in frame['files']:
            encoding = depth_encoding_for_scene(data['path'], path=self._path(data, frame, aliases['depth']))
            metadata.setdefault(aliases['depth'], {}).update({'encoding': encoding, 'unit': 'm'})
            if encoding == 'meters_u8': warnings.append('Legacy depth PNG has metre rounding and saturation at 255 m; saturated pixels are invalid. Exact depth cannot be recovered.')
            elif encoding == 'legacy_100m_u8': warnings.append('Depth encoding has no collection provenance; legacy 100/255 scaling is explicit compatibility mode.')
        return clean_json({'id': identifier, 'index': index, 'total': len(data['frames']),
            'frame_key': frame['frame_key'], 'frame_id': frame['frame_id'], 'timestamp': frame['timestamp'],
            'channels': sorted(frame['files']), 'missing': frame['missing'],
            'scene': normalize_scene(frame['raw_info'], data['manifest'], frame['frame_key']),
            'raw_info': copy.deepcopy(frame['raw_info']), 'channel_metadata': metadata, 'boxes': boxes,
            'box_format': {'type': '2d', 'columns': ['x1','y1','x2','y2','class_id'], 'units': 'pixels', 'object_ids_recorded': False},
            'bev_gt': bev_description, 'image_urls': image_urls,
            'warnings': warnings + ([frame['metadata_error']] if frame['metadata_error'] else [])})

    def depth(self, identifier, index, camera='front_cam'):
        data, frame = self._frame(identifier, index)
        channel = camera + ('_depth_m' if camera + '_depth_m' in frame['files'] else '_depth')
        path = self._path(data, frame, channel)
        encoding = depth_encoding_for_scene(data['path'], path=path)
        values, valid = decode_depth(read_array(path), encoding)
        return values, valid, encoding

    def render(self, identifier, index, channel='rgb', camera='front_cam', layers=None,
               classes=None, show_boxes=True, colors=None, depth_min=0., depth_max=100., colormap='turbo'):
        if camera not in CAMERAS: raise ValueError('Unknown camera')
        data, frame = self._frame(identifier, index)
        source_signature = []
        for key in sorted(frame['files']):
            path = self._path(data, frame, key)
            stat = path.stat()
            source_signature.append((key, stat.st_size, stat.st_mtime_ns))
        cache_key = json.dumps([identifier, frame['frame_key'], source_signature, channel, camera,
            layers, classes, show_boxes, colors, depth_min, depth_max, colormap], sort_keys=True)
        with self._lock:
            if cache_key in self._images:
                self._images.move_to_end(cache_key)
                return self._images[cache_key]
        result = self._render(identifier, index, channel, camera, layers, classes, show_boxes,
                              colors, depth_min, depth_max, colormap)
        with self._lock:
            old = self._images.pop(cache_key, None)
            if old is not None: self._image_bytes -= len(old)
            self._images[cache_key] = result
            self._image_bytes += len(result)
            while self._image_bytes > 32 * 1024 * 1024 or len(self._images) > 64:
                _, removed = self._images.popitem(last=False)
                self._image_bytes -= len(removed)
        return result

    def _render(self, identifier, index, channel, camera, layers, classes,
                show_boxes, colors, depth_min, depth_max, colormap):
        data, frame = self._frame(identifier, index)
        if channel == 'od':
            image = Image.fromarray(read_array(self._path(data, frame, camera + '_rgb'))).convert('RGB')
            boxes = bbox_annotations(self._path(data, frame, camera + '_bbs'), image.width, image.height)
            if classes is not None and (not isinstance(classes, (list, tuple)) or any(c not in OD_CLASSES for c in classes)):
                raise ValueError('Unknown object classes')
            if colors is not None and (not isinstance(colors, dict) or any(not re.fullmatch(r'#[0-9A-Fa-f]{6}', str(v)) for v in colors.values())):
                raise ValueError('Class colors require #RRGGBB values')
            draw = ImageDraw.Draw(image)
            if show_boxes:
                for box in boxes:
                    cid = box['class_id']
                    if classes is not None and cid not in classes: continue
                    color = (colors or {}).get(str(cid), (colors or {}).get(cid, OD_COLORS[cid]))
                    draw.rectangle(box['xyxy'], outline=color, width=max(1, image.width // 300))
                    draw.text((max(0, box['xyxy'][0]), max(0, box['xyxy'][1] - 12)), box['class_name'], fill=color)
        elif channel == 'seg':
            key = camera + ('_seg_ids' if camera + '_seg_ids' in frame['files'] else '_seg')
            ids = decode_segmentation(read_array(self._path(data, frame, key)))
            image = Image.fromarray(colorize_segmentation(ids))
        elif channel == 'depth':
            low, high = _finite(depth_min), _finite(depth_max)
            if low is None or high is None or not 0 <= low < high <= 10000:
                raise ValueError('Depth visualization requires 0 <= minimum < maximum <= 10000 metres')
            values, valid, encoding = self.depth(identifier, index, camera)
            image = Image.fromarray(depth_colors(values, valid, low, high, colormap))
        elif channel in BEV_CHANNELS and 'bev_gt' in frame['files']:
            arrays = read_bev(self._path(data, frame, 'bev_gt'), frame['raw_info'].get('bev_gt'))
            from sdv.bev.schema import SEMANTIC_CLASSES
            palette = {record['id']: record['color'] for record in SEMANTIC_CLASSES}
            image = render_bev(arrays, channel, layers, palette)
        elif channel == 'bev':
            image = self._bev(data, frame, layers)
        else:
            key = camera + '_rgb' if channel == 'rgb' else channel
            if key not in frame['files'] or key in ('info', 'bev_gt', 'lidar') or key.endswith('_bbs'):
                raise FileNotFoundError('Unsupported or missing image channel')
            image = Image.fromarray(read_array(self._path(data, frame, key)))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()

    def _bev(self, data, frame, layers):
        default_layers = layers is None
        if default_layers: layers = [k for k, v in BEV_LAYERS.items() if v['folder'] in frame['files']]
        aliases = {v['folder']: k for k, v in BEV_LAYERS.items()}
        layers = [aliases.get(k, k) for k in layers]
        if any(k not in BEV_LAYERS for k in layers): raise ValueError('Unknown BEV layer')
        if not layers and default_layers and 'bev' in frame['files']:
            return Image.fromarray(read_array(self._path(data, frame, 'bev'))).convert('RGB')
        size = None
        for layer in BEV_LAYERS.values():
            if layer['folder'] in frame['files']:
                with Image.open(str(self._path(data, frame, layer['folder']))) as source: size = source.size
                break
        if size is None and 'bev' in frame['files']:
            with Image.open(str(self._path(data, frame, 'bev'))) as source: size = source.size
        if size is None: raise FileNotFoundError('No BEV labels in this sample')
        if size[0] * size[1] > MAX_PIXELS: raise ValueError('BEV exceeds preview limit')
        output = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        output[:] = (10, 18, 31)
        for key in layers:
            folder = BEV_LAYERS[key]['folder']
            array = read_array(self._path(data, frame, folder))
            if array.shape[:2] != output.shape[:2]: raise ValueError('BEV layer dimensions are not aligned')
            occupied = np.any(array > 0, axis=-1) if array.ndim == 3 else array > 0
            if key == 'traffic_light' and array.ndim == 3: output[occupied] = array[..., :3][occupied]
            else:
                color = LAYER_COLORS[key]
                output[occupied] = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        return Image.fromarray(output)


def depth_colors(values, valid, low, high, colormap):
    """Small explicit RGB LUTs keep the web service free of matplotlib/OpenCV."""
    palettes = {
        'gray': [(0,0,0),(255,255,255)],
        'viridis': [(68,1,84),(59,82,139),(33,145,140),(94,201,98),(253,231,37)],
        'plasma': [(13,8,135),(126,3,168),(204,71,120),(248,149,64),(240,249,33)],
        'turbo': [(48,18,59),(50,104,226),(36,221,175),(164,252,60),(251,126,33),(122,4,3)],
    }
    if colormap not in palettes: raise ValueError('Unsupported depth color map')
    normalized = np.clip((np.nan_to_num(values, nan=low, posinf=high, neginf=low) - low) / (high - low), 0, 1)
    points = np.asarray(palettes[colormap], dtype=np.float64)
    positions = np.linspace(0, 1, len(points))
    output = np.stack([np.interp(normalized, positions, points[:, i]) for i in range(3)], axis=-1).astype(np.uint8)
    output[~valid] = (0, 0, 0)
    return output
