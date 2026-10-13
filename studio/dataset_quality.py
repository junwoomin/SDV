"""Read-only quality checks with sample-level evidence and explicit unknowns."""
from __future__ import annotations
from collections import Counter
import math
import re

import numpy as np
from PIL import Image

from sdv.data_contract import SEMANTIC_NAMES, decode_segmentation, semantic_records
from .sequences import _finite, bbox_annotations, clean_json, natural_key, read_array


def validate_sequence(service, identifier):
    data = service._inventory(identifier)
    frames, manifest = data['frames'], data['manifest']
    cameras = [c for c in ('front_cam', 'left_cam', 'right_cam', 'rear_cam', 'rear_left_cam', 'rear_right_cam')
               if c + '_rgb' in data['channels']]
    issues, counts, semantic_pixels, object_counts = [], Counter(), Counter(), Counter()
    depth_summary = {'valid_pixels': 0, 'invalid_pixels': 0, 'saturated_pixels': 0,
                     'minimum_m': None, 'maximum_m': None, 'encodings': []}
    def issue(severity, code, frame, message, channel=None):
        counts[severity] += 1
        issues.append({'severity': severity, 'code': code,
                      'frame_index': frame['index'] if frame else None,
                      'frame_key': frame['frame_key'] if frame else None,
                      'channel': channel, 'message': message})
    if not frames: issue('error', 'empty_sequence', None, 'No recorded samples')
    if not data['calibration']: issue('warning', 'calibration_missing', None, 'Collection-time sensor calibration is unavailable')
    sensor_definitions = (data['calibration'] or {}).get('sensors') or {}
    expected_channels = set()
    if isinstance(sensor_definitions, dict):
        for name, fields in sensor_definitions.items():
            if not isinstance(fields, dict) or fields.get('enabled') is False: continue
            if name.endswith('_cam'):
                expected_channels.add(name + '_rgb')
                for flag, suffix in (('seg','_seg'),('depth','_depth'),('od','_bbs')):
                    if fields.get(flag): expected_channels.add(name + suffix)
    original_order = sorted(frames, key=lambda f: natural_key(f['frame_key']))
    previous, previous_sample = None, None
    timestamps = set()
    for frame in original_order:
        raw = service._info(data, frame)
        if 'bev_gt' in frame['files']:
            from sdv.bev.io import load_frame
            from sdv.bev.quality import validate_frame
            bev_meta = raw.get('bev_gt') or {}
            try:
                arrays = load_frame(service._path(data,frame,'bev_gt'),bev_meta.get('geometry'),metadata=bev_meta)
                for found in validate_frame(arrays,bev_meta,frame['frame_id'],frame['timestamp']):
                    issue(found.get('severity','error'),found['code'],frame,found['message'],'bev_gt')
                for message in bev_meta.get('warnings',[]):
                    issue('warning','bev_partial_coverage',frame,message,'bev_gt')
            except (OSError,ValueError,KeyError) as error:
                issue('error','invalid_bev_gt',frame,str(error),'bev_gt')
        if frame['metadata_error']: issue('error', 'invalid_metadata', frame, frame['metadata_error'], 'info')
        for missing in frame['missing']:
            issue('error', 'missing_modality', frame, 'No file with this sample key', missing)
        for missing in sorted(expected_channels - set(frame['files']) - set(frame['missing'])):
            # Authoritative arrays supersede their optional legacy previews.
            if missing.endswith('_depth') and missing + '_m' in frame['files']: continue
            if missing.endswith('_seg') and missing + '_ids' in frame['files']: continue
            issue('error', 'configured_sensor_missing', frame, 'Collection sensor configuration requested this modality but its sample file is absent', missing)
        if 'front_cam_rgb' not in frame['files']:
            issue('error', 'missing_reference_rgb', frame, 'Reference front RGB is missing', 'front_cam_rgb')
        sample_match = re.fullmatch(r'(\d+)_a', frame['frame_key'])
        sample_number = int(sample_match.group(1)) if sample_match else None
        # These names are the SDV collector's committed sample counter, not
        # native CARLA frame IDs. Native frame steps may legitimately be > 1.
        if sample_number is not None and previous_sample is not None and sample_number > previous_sample + 1:
            issue('warning', 'sample_id_gap', frame, 'Missing sample counters {}..{}; simulator elapsed time is not inferred'.format(previous_sample + 1, sample_number - 1))
        previous_sample = sample_number
        timestamp = frame['timestamp']
        if timestamp is None:
            issue('warning', 'timestamp_missing', frame, 'No recorded simulator timestamp; temporal alignment cannot be proven', 'info')
        else:
            if timestamp in timestamps: issue('error', 'timestamp_duplicate', frame, 'Repeated simulator timestamp', 'info')
            timestamps.add(timestamp)
            if previous is not None and previous['timestamp'] is not None and timestamp <= previous['timestamp']:
                issue('error', 'timestamp_order', frame, 'Timestamp is not increasing in sample-ID order', 'info')
        extension = raw.get('sdv_metadata') or {}
        if not isinstance(extension, dict): extension = {}
        synchronization = extension.get('synchronization') or {}
        if not isinstance(synchronization, dict): synchronization = {}
        sensors = synchronization.get('sensors')
        native_frame = synchronization.get('carla_frame')
        if native_frame is not None and (_finite(native_frame) is None or isinstance(native_frame, bool)):
            issue('error', 'invalid_simulator_frame', frame, 'CARLA frame ID is not a finite numeric value', 'info')
        if isinstance(sensors, dict):
            for sensor, record in sensors.items():
                if not isinstance(record, dict):
                    issue('error', 'sensor_metadata_invalid', frame, 'Sensor metadata must be an object', sensor)
                    continue
                # New v2 collectors require IMU alignment too. A legacy
                # asynchronous IMU is still identified explicitly.
                severity = 'warning' if sensor == 'imu' and extension.get('schema_version') not in (2,3,4) else 'error'
                if record.get('frame') is not None and native_frame is not None and record['frame'] != native_frame:
                    issue(severity, 'sensor_frame_mismatch', frame,
                          'Sensor frame {} differs from world frame {}'.format(record['frame'], native_frame), sensor)
                sensor_time = _finite(record.get('timestamp_s'))
                if sensor_time is not None and timestamp is not None and abs(sensor_time - timestamp) > 1e-5:
                    issue(severity, 'sensor_timestamp_mismatch', frame,
                          'Sensor timestamp differs from world timestamp', sensor)
        else: issue('warning', 'sensor_timestamps_missing', frame, 'File keys agree where present, but sensor timestamps were not saved')
        if synchronization.get('aligned') is False:
            issue('error', 'synchronization_failed', frame, 'Collector explicitly reported unaligned sensors')
        if previous is not None and native_frame is not None and previous['frame_id'] is not None:
            delta = _finite(native_frame)
            before = _finite(previous['frame_id'])
            expected = _finite(manifest.get('save_every_ticks'))
            if delta is not None and before is not None and delta <= before:
                issue('error', 'native_frame_order', frame, 'CARLA frame IDs are not increasing')
            elif delta is not None and before is not None and expected and delta - before != expected:
                issue('warning', 'native_frame_interval', frame,
                      'Observed CARLA interval {} differs from configured saved interval {}; missed acquisition or variable cadence'.format(delta-before, expected))
        dimensions = {}
        for channel in frame['files']:
            if channel in ('info','lidar','bev_gt') or channel.endswith('_bbs'): continue
            try:
                path = service._path(data, frame, channel)
                if path.suffix.lower() != '.npy':
                    with Image.open(str(path)) as image:
                        dimensions[channel] = image.size
                        image.verify()
            except (OSError, ValueError, SyntaxError) as error:
                issue('error', 'unreadable_image', frame, str(error), channel)
        for camera in cameras:
            rgb_size = dimensions.get(camera + '_rgb')
            calibration = (data['calibration'] or {}).get('sensors') or {}
            sensor = calibration.get(camera) if isinstance(calibration, dict) else None
            if isinstance(sensor, dict) and rgb_size and (sensor.get('width'), sensor.get('height')) != rgb_size:
                issue('error', 'calibration_resolution', frame, 'RGB dimensions do not match collection calibration', camera + '_rgb')
            for suffix in ('_seg', '_seg_ids', '_depth'):
                size = dimensions.get(camera + suffix)
                if size and rgb_size and size != rgb_size:
                    issue('error', 'image_alignment', frame, 'Label resolution differs from paired RGB', camera + suffix)
            segment_channel = camera + ('_seg_ids' if camera + '_seg_ids' in frame['files'] else '_seg')
            if segment_channel in frame['files']:
                try:
                    ids = decode_segmentation(read_array(service._path(data, frame, segment_channel)))
                    unique, pixel_counts = np.unique(ids, return_counts=True)
                    semantic_pixels.update({int(k): int(v) for k, v in zip(unique, pixel_counts)})
                    invalid = int(np.count_nonzero(ids == 255))
                    if invalid: issue('error', 'unknown_segmentation_label', frame, '{} pixels are not valid CARLA class IDs/palette colors'.format(invalid), segment_channel)
                except (OSError, ValueError) as error: issue('error', 'invalid_segmentation', frame, str(error), segment_channel)
            if camera + '_depth' in frame['files'] or camera + '_depth_m' in frame['files']:
                try:
                    values, valid, encoding = service.depth(identifier, frame['index'], camera)
                    if rgb_size and (values.shape[1], values.shape[0]) != rgb_size:
                        issue('error', 'image_alignment', frame, 'Depth dimensions differ from paired RGB', camera + '_depth')
                    if encoding not in depth_summary['encodings']: depth_summary['encodings'].append(encoding)
                    depth_summary['valid_pixels'] += int(valid.sum())
                    depth_summary['invalid_pixels'] += int((~valid).sum())
                    if valid.any():
                        low, high = float(values[valid].min()), float(values[valid].max())
                        depth_summary['minimum_m'] = min(low, depth_summary['minimum_m']) if depth_summary['minimum_m'] is not None else low
                        depth_summary['maximum_m'] = max(high, depth_summary['maximum_m']) if depth_summary['maximum_m'] is not None else high
                    saturated = int((values >= 255).sum()) if encoding == 'meters_u8' else 0
                    depth_summary['saturated_pixels'] += saturated
                    if saturated: issue('warning', 'depth_saturated', frame, '{} depth pixels are irreversibly saturated at 255 m'.format(saturated), camera + '_depth')
                    if (~valid).any(): issue('warning', 'depth_invalid', frame, '{} invalid depth pixels are masked'.format(int((~valid).sum())), camera + '_depth')
                    if np.any(np.isfinite(values) & ((values < 0) | (values > 1000))):
                        issue('error', 'depth_outside_sensor_range', frame, 'CARLA depth values outside 0..1000 metres', camera + '_depth')
                    if encoding == 'legacy_100m_u8': issue('warning', 'depth_units_unverified', frame, 'No collector manifest; legacy 100/255 unit assumption is explicitly used', camera + '_depth')
                except (OSError, ValueError, IndexError) as error: issue('error', 'invalid_depth', frame, str(error), camera + '_depth')
            if camera + '_bbs' in frame['files']:
                try:
                    boxes = bbox_annotations(service._path(data, frame, camera + '_bbs'), *(rgb_size or (None, None)))
                    object_counts.update(box['class_id'] for box in boxes)
                    if any(box['outside_image'] for box in boxes): issue('error', 'bbox_outside_image', frame, 'Bounding box exceeds paired RGB bounds', camera + '_bbs')
                    if not rgb_size: issue('error', 'bbox_without_rgb', frame, 'Bounding boxes have no same-frame RGB', camera + '_bbs')
                except (OSError, ValueError) as error: issue('error', 'invalid_bounding_boxes', frame, str(error), camera + '_bbs')
        speed = _finite(raw.get('current_speed'))
        if raw.get('current_speed') is not None and (speed is None or speed < 0 or speed > 400):
            issue('warning', 'ego_speed_outlier', frame, 'Speed is non-finite or outside plausibility range 0..400 km/h', 'info')
        control = raw.get('current_control') or {}
        if isinstance(control, dict):
            for key, low, high in [('steer',-1,1),('throttle',0,1),('brake',0,1)]:
                value = _finite(control.get(key))
                if control.get(key) is not None and (value is None or not low <= value <= high):
                    issue('error', 'control_range', frame, '{} outside [{},{}]'.format(key, low, high), 'info')
        imu = raw.get('imu') or {}
        acceleration = (extension.get('ego') or {}).get('acceleration_m_s2') if isinstance(extension.get('ego') or {}, dict) else None
        if acceleration is None and isinstance(imu, dict): acceleration = imu.get('accelerometer')
        if isinstance(acceleration, dict):
            xyz = [_finite(acceleration.get(k)) for k in ('x','y','z')]
            if any(v is None for v in xyz) or math.sqrt(sum(v*v for v in xyz)) > 100:
                issue('warning', 'acceleration_outlier', frame, 'Acceleration is non-finite or magnitude exceeds plausibility threshold 100 m/s²; inspect startup IMU', 'info')
        if not extension.get('road'): issue('warning', 'road_metadata_missing', frame, 'Road/lane fields were not recorded', 'info')
        if not extension.get('navigation'): issue('warning', 'navigation_metadata_missing', frame, 'RoadOption and world-space route targets were not recorded; legacy BEV plans remain available', 'info')
        if not extension.get('traffic'): issue('warning', 'traffic_metadata_missing', frame, 'Signal ID/state/lane applicability were not recorded; legacy hazard flags remain available', 'info')
        previous = frame
    if any(k.endswith('_bbs') for k in data['channels']) and not object_counts:
        issue('warning', 'no_object_annotations', None, 'All stored object arrays are empty. This is not evidence that no visible objects existed; old collectors used incompatible semantic visibility IDs.')
    return clean_json({'sequence_id': identifier, 'total_frames': len(frames), 'checked_frames': len(frames),
        'status': 'failed' if counts['error'] else 'warning' if counts['warning'] else 'passed',
        'counts': {key: counts[key] for key in ('error','warning','info')}, 'issues': issues,
        'summary': {'semantic_classes': [dict(item, pixels=semantic_pixels[item['id']]) for item in semantic_records(sorted(semantic_pixels))],
                    'unknown_semantic_pixels': semantic_pixels[255], 'object_instances': dict(object_counts),
                    'depth': depth_summary, 'timestamps_recorded': len(timestamps),
                    'coordinate_units': {'position': 'm', 'speed': 'km/h', 'rotation': 'degrees', 'depth': 'm', 'boxes': 'image pixels'},
                    'timestamps_inferred': False},
        'checks': ['file-key alignment','RGB/box pairing','image dimensions/calibration','semantic IDs/palette',
                   'depth units/range/invalids','sensor frame/timestamps','sample gaps','ego plausibility','navigation/traffic metadata']})
