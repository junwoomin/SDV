"""Offline realized future targets from recorded sequences, not planner plans.

The export intentionally is not a TCP-specific training schema. It references
original images/calibration instead of copying them, and labels targets derived
from future observations separately from information available at the input.
"""
from __future__ import annotations
import hashlib
import json
import math
import re

from .sequences import _finite, clean_json, normalize_scene


def _xyz(value):
    if not isinstance(value, dict): return None
    result = [_finite(value.get(k)) for k in ('x','y','z')]
    return result if all(v is not None for v in result) else None


def world_to_ego(point, origin, rotation):
    """CARLA's full pitch/yaw/roll inverse rotation (same as collector matrix)."""
    if not isinstance(rotation, dict): raise ValueError('Recorded ego rotation is required')
    angles = [_finite(rotation.get(k)) for k in ('pitch','yaw','roll')]
    if any(v is None for v in angles): raise ValueError('Finite pitch/yaw/roll are required')
    pitch, yaw, roll = [math.radians(a) for a in angles]
    cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
    matrix = ((cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr),
              (sy*cp, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr),
              (sp, -cp*sr, cp*cr))
    delta = [p-o for p, o in zip(point, origin)]
    return [sum(matrix[j][i] * delta[j] for j in range(3)) for i in range(3)]


def export_sequence(service, identifier, horizon=5, stride=1, max_gap_s=None):
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 100:
        raise ValueError('Future horizon must contain 1..100 recorded samples')
    if isinstance(stride, bool) or not isinstance(stride, int) or not 1 <= stride <= 100:
        raise ValueError('Target stride must be an integer in 1..100')
    if max_gap_s is not None:
        max_gap_s = _finite(max_gap_s)
        if max_gap_s is None or max_gap_s <= 0 or max_gap_s > 3600:
            raise ValueError('Maximum timestamp gap must be positive and at most 3600 seconds')
    data = service._inventory(identifier)
    frames, records, omitted, warnings = data['frames'], [], [], []
    has_time = bool(frames) and all(f['timestamp'] is not None for f in frames)
    fps = _finite(data['manifest'].get('fps'))
    frame_stride = _finite(data['manifest'].get('save_every_ticks'))
    expected_dt = frame_stride / fps if fps and fps > 0 and frame_stride and frame_stride > 0 else None
    effective_gap = max_gap_s if max_gap_s is not None else expected_dt * 1.5 if expected_dt else None
    if not has_time:
        warnings.append('Original simulator timestamps are missing: targets use recorded-sample offsets only; time continuity and target duration are unverified.')
        if max_gap_s is not None: raise ValueError('A seconds-based continuity limit requires recorded timestamps for every sample')
    digest = hashlib.sha256()
    def numeric_key(frame):
        match = re.fullmatch(r'(\d+)_a', frame['frame_key'])
        return int(match.group(1)) if match else None
    for index, frame in enumerate(frames):
        raw = service._info(data, frame)
        digest.update(json.dumps(raw, sort_keys=True, allow_nan=True).encode())
        origin = _xyz(raw.get('ego_vehicle_loc'))
        rotation = raw.get('ego_vehicle_rot')
        if origin is None:
            omitted.append({'frame_key': frame['frame_key'], 'reason': 'Recorded ego world position is missing'})
            continue
        try: world_to_ego(origin, origin, rotation)
        except ValueError as error:
            omitted.append({'frame_key': frame['frame_key'], 'reason': str(error)})
            continue
        targets, reason = [], None
        for offset in range(1, horizon*stride + 1):
            future_index = index + offset
            if future_index >= len(frames): reason = 'End of this sequence'; break
            before, future = frames[future_index - 1], frames[future_index]
            a, b = numeric_key(before), numeric_key(future)
            if a is not None and b is not None and b != a + 1:
                reason = 'Missing or reordered recorded sample counter'; break
            if has_time:
                dt = future['timestamp'] - before['timestamp']
                if dt <= 0 or (effective_gap is not None and dt > effective_gap):
                    reason = 'Timestamp discontinuity'; break
            previous_native, future_native = _finite(before['frame_id']), _finite(future['frame_id'])
            if frame_stride and previous_native is not None and future_native is not None and future_native - previous_native != frame_stride:
                reason = 'Simulator frame discontinuity'; break
            future_info = service._info(data, future)
            previous_info = service._info(data, before)
            position = _xyz(future_info.get('ego_vehicle_loc'))
            previous_position = _xyz(previous_info.get('ego_vehicle_loc'))
            if position is None or previous_position is None:
                reason = 'Future ego world position is missing'; break
            # Do not cross a teleport even if a producer reused the same scene.
            distance = math.sqrt(sum((x-y)**2 for x, y in zip(position, previous_position)))
            if distance > 50:
                reason = 'Position discontinuity > 50 metres between saved samples'; break
            if offset % stride: continue
            targets.append({'offset_samples': offset, 'frame_key': future['frame_key'],
                'frame_id': future['frame_id'], 'timestamp_s': future['timestamp'],
                'offset_s': future['timestamp'] - frame['timestamp'] if has_time else None,
                'position_world_m': position, 'position_ego_m': world_to_ego(position, origin, rotation),
                'speed_kmh': _finite(future_info.get('current_speed')),
                'applied_control': future_info.get('current_control'),
                'source': 'observed future info JSON, offline ground truth'})
        if not targets:
            omitted.append({'frame_key': frame['frame_key'], 'reason': reason or 'No future observation'})
            continue
        normalized = normalize_scene(raw, data['manifest'], frame['frame_key'])
        records.append({'sequence_id': identifier, 'frame_index': index, 'frame_key': frame['frame_key'],
            'frame_id': frame['frame_id'], 'timestamp_s': frame['timestamp'],
            'inputs': {'files': {k:v for k,v in frame['files'].items() if k.endswith('_rgb') or k=='lidar'},
                       'ego': normalized['ego'], 'navigation': normalized['navigation']},
            'simulator_ground_truth': {'files': {k:v for k,v in frame['files'].items() if not (k.endswith('_rgb') or k=='lidar')},
                       'road': normalized['road'], 'traffic': normalized['traffic'],
                       'environment': normalized['environment'], 'bev_metadata': raw.get('bev_gt'),
                       'online_input': False},
            'targets': {'future_observations': targets, 'complete_horizon': len(targets) == horizon,
                        'truncated_reason': reason, 'temporal_continuity_verified': has_time and effective_gap is not None}})
    return clean_json({'schema': 'sdv-planning-observations-v2', 'sequence_id': identifier,
        'source_sequence': str(data['path']), 'source_modified': False,
        'source_metadata_sha256': digest.hexdigest(), 'calibration': data['calibration'],
        'coordinates': {'world': 'CARLA left-handed XYZ, metres',
                        'target_ego': 'Full inverse source-frame CARLA rotation: x forward, y right, z up; metres',
                        'online_bev_plans': 'Recorded BEV pixels, not future ground truth'},
        'horizon_samples': horizon, 'stride_samples': stride, 'max_gap_s': max_gap_s,
        'effective_max_gap_s': effective_gap,
        'source_frames': len(frames), 'record_count': len(records), 'records': records,
        'omitted': omitted, 'warnings': warnings,
        'tcp_compatibility': 'Not asserted. Adapt this observation/target schema after inspecting the intended TCP training code.'})
