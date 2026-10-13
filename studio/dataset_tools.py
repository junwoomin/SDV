"""Explicit, reversible scene-level dataset preparation.

Source collections are never modified. Complete selected scenes are copied to
one new dataset, retaining binary DAS/lane labels and sensor calibration. The
native-mask loader fallback preserves overlapping road/lane pixels; converting
these into the old mutually-exclusive colour palette would lose information.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time

import yaml
from sdv.data_contract import normalize_task_settings

from .catalog import CAMERAS, Catalog, safe_child, read_text


def _frames(scene, modality):
    folder = safe_child(scene, modality)
    extensions = (('.npz',) if modality == 'bev_gt' else ('.json',) if modality == 'info' else
                  ('.npy',) if modality == 'lidar' or modality.endswith(('_bbs','_depth_m')) else
                  ('.npy','.png') if modality.endswith('_depth') else ('.png',))
    if not folder.is_dir():
        return set()
    result = set()
    for item in folder.iterdir():
        if item.is_symlink():
            raise ValueError(f'{scene.name}/{modality}: 심볼릭 링크 대신 실제 수집 파일이 필요합니다.')
        if item.is_file() and item.suffix.lower() in extensions:
            if item.stat().st_size == 0:
                raise ValueError(f'{scene.name}/{modality}/{item.name}: 저장이 완료되지 않은 파일입니다.')
            result.add(item.stem)
    return result


def _png_size(path):
    with path.open('rb') as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'{path.parent.name}/{path.name}: PNG 파일 헤더를 읽을 수 없습니다.')
    return struct.unpack('>II', header[16:24])


def _calibration_fingerprint(config, training):
    cameras = [c for c in CAMERAS if training.get('sensor_use', {}).get(c)]
    keys = {'width', 'height', 'fov', 'x', 'y', 'z', 'pitch', 'yaw', 'roll'}
    if not isinstance(config, dict) or not isinstance(config.get('sensors'), dict):
        raise ValueError('센서 보정 YAML에 sensors 객체가 필요합니다.')
    sensors = config['sensors']
    selected = {camera: {key: float(sensors[camera][key]) for key in sorted(keys)} for camera in cameras}
    if any(not math.isfinite(value) for sensor in selected.values() for value in sensor.values()):
        raise ValueError('센서 보정값은 유한한 숫자여야 합니다.')
    if training.get('lidar'):
        if not isinstance(sensors.get('lidar'), dict):
            raise ValueError('선택한 LiDAR의 센서 설정이 없습니다.')
        selected['lidar'] = sensors['lidar']
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def _origin(scene, default):
    origin_file = safe_child(scene, 'dataset_origin.json')
    collection_file = safe_child(scene, 'collection_manifest.json')
    if origin_file.is_file():
        origin = json.loads(read_text(origin_file, 65536))
        if not isinstance(origin, dict):
            raise ValueError('dataset_origin.json 형식이 잘못되었습니다.')
        return origin.get('source_scene_id', default)
    if collection_file.is_file():
        collection = json.loads(read_text(collection_file, 65536))
        if not isinstance(collection, dict):
            raise ValueError('collection_manifest.json 형식이 잘못되었습니다.')
        if all(collection.get(key) for key in ('run', 'town', 'scene')):
            return json.dumps([collection[key] for key in ('run', 'town', 'scene')])
    return default


def validate_scene(scene, train_config=None):
    """Validate selected modalities against sequence/frame keys and calibration."""
    train_config = train_config or {'sensor_use': {'front_cam': True}, 'model_conf': {}}
    front = _frames(scene, 'front_cam_rgb')
    if not front:
        raise ValueError(f'{scene.name}: front_cam_rgb에 저장된 PNG 표본이 없습니다.')
    cameras = [c for c in CAMERAS if train_config.get('sensor_use', {}).get(c)]
    if 'front_cam' not in cameras:
        raise ValueError('front_cam을 학습 센서로 선택하세요. 기존 로더의 표본 기준입니다.')
    tasks = train_config.get('model_conf', {})
    advanced = normalize_task_settings(train_config.get('task_settings'))
    groups = [(f'{c}_rgb',) for c in cameras]
    for camera in cameras:
        if tasks.get('Seg'):
            groups.append((camera+'_seg_ids', camera+'_seg') if advanced['segmentation']['mode']=='carla' else (camera+'_seg',))
        if tasks.get('Depth'): groups.append((camera+'_depth_m', camera+'_depth'))
        if tasks.get('od'): groups.append((camera+'_bbs',))
    if train_config.get('lidar') or train_config.get('sensor_use', {}).get('lidar'):
        groups.append(('lidar',))
    legacy_bev = _frames(scene, 'bev')
    native = _frames(scene, 'das') and _frames(scene, 'lane')
    bev_groups = []
    bev_schema = None
    valid_bev_pixels = {}
    if tasks.get('BEV'):
        from sdv.bev.training import frame_bev_schema
        schemas = {frame_bev_schema(scene, frame) or 1 for frame in front}
        if len(schemas) != 1:
            raise ValueError(f'{scene.name}: legacy/schema-3/schema-4 BEV를 한 장면에 섞어 학습할 수 없습니다.')
        bev_schema = next(iter(schemas))
        if bev_schema in (3,4):
            bev_groups = [('bev_gt',), ('info',)]
        else:
            for layer in advanced['bev']['layers']:
                name = {'das':'das','lane':'lane','vehicle':'vehicle_mask','walker':'walker_mask'}[layer]
                options = (name,'bev') if layer in ('das','lane') and _frames(scene,name)!=front else (name,)
                bev_groups.append(options)
        groups.extend(bev_groups)
    selected = set()
    for group in groups:
        actual = set()
        for modality in group:
            frames = _frames(scene, modality)
            actual.update(frames)
            if frames: selected.add(modality)
        if actual != front:
            missing, extra = sorted(front - actual), sorted(actual - front)
            details = []
            if missing: details.append(f'누락 {len(missing)}개 (예: {", ".join(missing[:3])})')
            if extra: details.append(f'불일치 {len(extra)}개 (예: {", ".join(extra[:3])})')
            raise ValueError(f'{scene.name}/{"|".join(group)}: 기준 카메라와 frame ID가 다릅니다. {"; ".join(details)}')
    # New collection stores a snapshot per scene. Do not quietly substitute
    # today's sensor settings for unknown historical camera extrinsics.
    calibration = safe_child(scene, 'sensor_config.yaml')
    if not calibration.is_file():
        raise ValueError(f'{scene.name}: 수집 당시 sensor_config.yaml 스냅샷이 없습니다. 당시 사용한 원본 설정을 이 장면 폴더에 복사한 후 다시 준비하세요. 현재 설정으로 자동 대체하지 않습니다.')
    try:
        config = yaml.safe_load(read_text(calibration, 2 * 1024 * 1024))
    except yaml.YAMLError as error:
        raise ValueError(f'{scene.name}: sensor_config.yaml을 읽을 수 없습니다: {error}') from error
    if not isinstance(config, dict) or not isinstance(config.get('sensors'), dict):
        raise ValueError(f'{scene.name}: 센서 스냅샷 형식이 잘못되었습니다.')
    camera_dimensions = set()
    for camera in cameras:
        if camera not in config['sensors']:
            raise ValueError(f'{scene.name}: 학습 센서 {camera}의 보정 정보가 없습니다.')
        fields = config['sensors'][camera]
        if not isinstance(fields, dict) or any(key not in fields for key in ('width', 'height', 'fov', 'x', 'y', 'z', 'pitch', 'yaw', 'roll')):
            raise ValueError(f'{scene.name}/{camera}: 카메라 크기/FOV/외부 보정값이 누락되었습니다.')
        try:
            expected = (int(fields['width']), int(fields['height']))
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f'{scene.name}/{camera}: width/height는 유한한 정수여야 합니다.') from error
        camera_dimensions.add(expected)
        for frame in front:
            if _png_size(scene / f'{camera}_rgb' / f'{frame}.png') != expected:
                raise ValueError(f'{scene.name}/{camera}: 실제 PNG 크기와 수집 보정 설정의 width/height가 다릅니다.')
    if len(camera_dimensions) > 1:
        raise ValueError(f'{scene.name}: 기존 로더가 첫 카메라의 크기를 모든 카메라에 사용하므로 선택한 카메라들의 width/height가 같아야 합니다.')
    if tasks.get('BEV'):
        if bev_schema in (3,4):
            import numpy as np
            from sdv.bev.training import model_ground_truth, validate_training_layers, camera_optical_to_bev_frame
            valid_bev_pixels = {layer:0 for layer in advanced['bev']['layers']}
            for frame in front:
                numeric = model_ground_truth(scene, frame)
                validate_training_layers(numeric['metadata'], advanced['bev']['layers'])
                for camera in cameras:
                    camera_optical_to_bev_frame(scene, frame, camera)
                for layer in valid_bev_pixels:
                    valid_bev_pixels[layer] += int(np.isfinite(numeric[layer]).sum())
            empty = [layer for layer, count in valid_bev_pixels.items() if not count]
            if empty:
                raise ValueError(f'{scene.name}: 선택한 BEV 태스크의 유효 학습 픽셀이 없습니다: {", ".join(empty)}')
        else:
            for frame in front:
                for group in bev_groups:
                    path = next((scene / modality / (frame+'.png') for modality in group
                                 if (scene / modality / (frame+'.png')).is_file()), None)
                    if path is None or _png_size(path) != (500,500):
                        raise ValueError(f'{scene.name}/{frame}: BEV 라벨은 native 500x500, 0.2 m/px 격자여야 합니다.')
    digest = _calibration_fingerprint(config, train_config)
    return {'samples': len(front), 'frames': sorted(front), 'modalities': sorted(selected),
        'calibration_sha256': digest, 'sensor_config': config, 'bev_schema':bev_schema,
        'valid_bev_pixels':valid_bev_pixels, 'label_format': 'global_raster_numeric_v4' if bev_schema == 4 else 'height_aware_numeric_v3' if bev_schema == 3 else
        'legacy_palette' if legacy_bev else 'native_binary' if native else 'none'}


def prepare_dataset(catalog, name, train_scenes, val_scenes, train_config=None):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('데이터셋 이름은 영문·숫자로 시작하는 1~64자의 영문, 숫자, _, -만 사용하세요.')
    if not isinstance(train_scenes, list) or not isinstance(val_scenes, list) or not train_scenes or not val_scenes:
        raise ValueError('학습과 검증에 각각 한 개 이상의 독립된 장면을 선택하세요.')
    if any(not isinstance(value, str) for value in train_scenes + val_scenes):
        raise ValueError('장면 ID 형식이 잘못되었습니다.')
    if len(set(train_scenes + val_scenes)) != len(train_scenes + val_scenes):
        raise ValueError('같은 장면을 중복 선택하거나 학습·검증에 동시에 넣을 수 없습니다.')
    selected = [(split, identifier, catalog.scene_path(identifier)) for split, ids in [('train', train_scenes), ('val', val_scenes)] for identifier in ids]
    validated, calibration_hashes, origins = [], set(), set()
    for split, identifier, scene in selected:
        checked = validate_scene(scene, train_config)
        origin = _origin(scene, identifier)
        if not isinstance(origin, str) or origin in origins:
            raise ValueError('같은 원본 장면 또는 그 복사본을 중복 선택했습니다. 학습·검증 누출을 방지하려면 독립된 장면을 선택하세요.')
        origins.add(origin)
        checked['source_scene_id'] = origin
        calibration_hashes.add(checked['calibration_sha256'])
        validated.append((split, identifier, scene, checked))
    if len(calibration_hashes) != 1:
        raise ValueError('선택한 장면의 학습 센서 보정값이 다릅니다. 기존 로더는 데이터셋 전체에 하나의 보정 설정을 사용하므로 같은 보정으로 수집한 장면만 함께 선택하세요.')
    if len({entry[3]['bev_schema'] for entry in validated if entry[3]['bev_schema'] is not None}) > 1:
        raise ValueError('legacy/schema-3/schema-4 BEV의 보정·라벨 좌표 계약이 다릅니다. 각각 별도 데이터셋과 실험으로 준비하세요.')
    # Prepared datasets are fixed under the project to avoid arbitrary writes.
    prepared = safe_child(catalog.root, 'data/prepared')
    prepared.mkdir(parents=True, exist_ok=True)
    destination = safe_child(prepared, name)
    if destination.exists():
        raise ValueError('같은 이름의 준비된 데이터셋이 이미 있습니다. 새 이름을 사용하세요.')
    required_bytes = 0
    for _, _, scene, _ in validated:
        for folder, directories, files in os.walk(scene, followlinks=False):
            if any((Path(folder) / d).is_symlink() for d in directories):
                raise ValueError(f'{scene.name}: 심볼릭 링크가 있는 장면은 복사하지 않습니다.')
            for filename in files:
                path = Path(folder) / filename
                if path.is_symlink():
                    raise ValueError(f'{scene.name}: 심볼릭 링크가 있는 장면은 복사하지 않습니다.')
                required_bytes += path.stat().st_size
    if shutil.disk_usage(prepared).free < required_bytes + 64 * 1024 * 1024:
        raise ValueError(f'원본을 보존하며 복사할 저장 공간이 부족합니다. 약 {required_bytes / 1024 ** 3:.2f} GiB가 필요합니다.')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=prepared))
    manifest = {'version': 1, 'name': name, 'created_at': time.time(), 'split_method': 'explicit_scene_selection', 'frame_randomization': False, 'source_modified': False, 'copy_method': 'independent_copy', 'label_conversion': 'none', 'label_note': '원본 PNG/숫자 BEV, 높이·유효 마스크, info 및 센서 보정을 변환 없이 함께 보존.', 'calibration_sha256': next(iter(calibration_hashes)), 'scenes': []}
    counts = {'train': 0, 'val': 0}
    try:
        for split, identifier, scene, checked in validated:
            # Opaque suffix avoids scene-name collisions across source towns.
            target_name = f'{scene.name}_{identifier[:8]}'
            target = temporary / split / target_name
            shutil.copytree(scene, target, copy_function=shutil.copy2)
            (target / 'dataset_origin.json').write_text(json.dumps({'source_scene_id': checked['source_scene_id'], 'source_path': str(scene)}, ensure_ascii=False, indent=2), encoding='utf-8')
            # Recheck after copying to catch a writer changing an active scene.
            copied = validate_scene(target, train_config)
            if copied['frames'] != checked['frames'] or copied['calibration_sha256'] != checked['calibration_sha256']:
                raise ValueError(f'{scene.name}: 준비 중 데이터가 변경되었습니다. 수집을 중지한 후 다시 시도하세요.')
            counts[split] += copied['samples']
            manifest['scenes'].append({'id': identifier, 'source': str(scene), 'split': split, 'destination': f'{split}/{target_name}', 'samples': copied['samples'], 'label_format': copied['label_format'], 'bev_schema': copied['bev_schema'], 'valid_bev_pixels': copied['valid_bev_pixels'], 'modalities': copied['modalities']})
        (temporary / 'sensor_config.yaml').write_text(yaml.safe_dump(validated[0][3]['sensor_config'], allow_unicode=True, sort_keys=False), encoding='utf-8')
        (temporary / 'dataset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        # No overwrite; in-process lock serializes preparations, and destination
        # existence is checked again immediately before rename.
        if destination.exists():
            raise ValueError('데이터셋 이름이 이미 사용 중입니다.')
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {'name': name, 'path': str(destination), 'samples': counts, 'scenes': len(selected), 'manifest': str(destination / 'dataset_manifest.json'), 'settings': {'train_data': str(destination / 'train'), 'val_data': str(destination / 'val'), 'sensor_config': str(destination / 'sensor_config.yaml')}, 'message': '장면 단위 분할과 센서 보정을 검증했습니다. 원본을 보존하여 복사했고, 학습 전 모델·가중치·GPU 조건은 별도로 확인합니다.'}


def validate_training_inputs(root, settings, training):
    """Read-only launch gate for prepared datasets and manually selected splits.

    Run this on explicit preflight/start, not on frequent status polling. It
    checks every frame's modality IDs and PNG dimensions from headers.
    Schema-3 BEV additionally decodes strict numeric arrays and verifies actual
    same-frame calibration/validity; no CUDA, torch or CARLA is imported.
    """
    catalog = Catalog(root, settings)
    blockers, origins, calibration_hashes, bev_schemas = [], {}, set(), set()
    for key, label in [('train_data', '학습'), ('val_data', '검증')]:
        split = catalog.path(settings.get(key, f'data/{"train" if key == "train_data" else "val"}'))
        scenes = []
        if split.is_dir():
            for directory, dirs, files in os.walk(str(split), followlinks=False):
                parent = Path(directory)
                if any((parent / name).is_symlink() for name in dirs):
                    blockers.append(f'{label}: 장면 심볼릭 링크는 사용할 수 없습니다: {parent.name}')
                dirs[:] = sorted(name for name in dirs if not name.startswith('.') and not (parent / name).is_symlink())
                if 'front_cam_rgb' in dirs:
                    scenes.append(parent)
                    dirs[:] = []
        scenes.sort()
        if not scenes:
            blockers.append(f'{label}: <분할>/<장면>/front_cam_rgb/*.png 구조의 데이터가 없습니다.')
            continue
        for scene in scenes:
            try:
                checked = validate_scene(scene, training)
                calibration_hashes.add(checked['calibration_sha256'])
                if checked['bev_schema'] is not None: bev_schemas.add(checked['bev_schema'])
                origin = _origin(scene, str(scene.resolve()))
                if not isinstance(origin, str):
                    raise ValueError('원본 장면 식별자가 잘못되었습니다.')
                if origin in origins:
                    blockers.append(f'{label}: {scene.name}는 {origins[origin]}와 같은 원본 장면입니다. 학습·검증에서 중복 장면을 제거하세요.')
                origins[origin] = f'{label}/{scene.name}'
            except (ValueError, OSError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
                blockers.append(f'{label}: {error}')
    if len(calibration_hashes) > 1:
        blockers.append('학습·검증 장면의 센서 보정값이 다릅니다. 동일한 보정으로 수집한 장면을 선택하세요.')
    if len(bev_schemas) > 1:
        blockers.append('학습·검증에 legacy/schema-3/schema-4 BEV가 섞여 있습니다. 보정·라벨 좌표 계약별로 별도 실험을 사용하세요.')
    calibration = catalog.path(settings.get('sensor_config') or 'data/sensor_config.yaml')
    try:
        if not calibration.is_file():
            raise ValueError(f'학습용 센서 보정 파일이 없습니다: {calibration}')
        current = yaml.safe_load(read_text(calibration, 2 * 1024 * 1024))
        digest = _calibration_fingerprint(current, training)
        if calibration_hashes and calibration_hashes != {digest}:
            blockers.append('설정에서 선택한 학습 보정 파일이 장면의 보정 스냅샷과 다릅니다. 준비된 데이터셋의 sensor_config.yaml을 선택하세요.')
    except (ValueError, OSError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        blockers.append(f'학습 보정: {error}')
    return list(dict.fromkeys(blockers))
