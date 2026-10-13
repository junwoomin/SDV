"""Read-only training choices from complete train/validation file inventories.

This is a lightweight UI gate, not a substitute for validate_training_inputs.
It does not decode files, inspect calibration, import ML packages or certify a
model. Inventories are scanned once per request; no timestamp cache can hide an
existing file being truncated by a writer. Call on dataset/selection changes,
not on the frequent job-status event stream.
"""
import os
from pathlib import Path
import stat
from sdv.data_contract import normalize_task_settings


# Same names as the research loader's sensor_use contract.
CAMERAS = ('front_cam', 'left_cam', 'right_cam', 'rear_cam',
           'rear_left_cam', 'rear_right_cam')
SENSORS = CAMERAS + ('lidar',)
TASKS = ('BEV', 'Seg', 'Depth', 'od')
SPLITS = (('train', 'train_data'), ('val', 'val_data'))


def _issue(code, ko, en, **details):
    return dict(details, reason_code=code, reason=ko, reason_en=en)


def _located(issue, scene, modality=None):
    detail = dict(issue, split=scene['split'], scene=scene['path'].name)
    if modality:
        detail['modality'] = modality
    location = '/'.join(str(detail[name]) for name in ('split', 'scene', 'modality') if name in detail)
    detail['reason'] = '{}: {}'.format(location, issue['reason'])
    detail['reason_en'] = '{}: {}'.format(location, issue['reason_en'])
    return detail


def _option(issues, requires=()):
    first = issues[0] if issues else {
        'reason_code': 'available',
        'reason': '학습·검증의 모든 장면에 필요한 파일과 frame ID가 있습니다. 실행 전 준비 검사는 별도로 필요합니다.',
        'reason_en': 'Required files and frame IDs are present in every training and validation scene. Run the full preflight before training.',
    }
    return {'available': not issues, 'reason': first['reason'],
            'reason_en': first['reason_en'], 'reason_code': first['reason_code'],
            'requires': list(requires), 'details': issues[:8], 'issue_count': len(issues)}


def _split_path(root, value):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError('Invalid dataset path')
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    # Resolving first rejects traversal and symlinks escaping the project.
    resolved = path.resolve()
    resolved.relative_to(root)
    # Prepared data uses independent real files; do not silently follow links.
    cursor = path
    while cursor != root and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError('Symbolic links are not training data')
        cursor = cursor.parent
    return resolved


def _inventory(scene, modality):
    if modality in scene['inventory']:
        return scene['inventory'][modality]
    folder = scene['path'] / modality
    frames, issues = set(), []
    result = {'frames': frames, 'issues': issues}
    scene['inventory'][modality] = result
    extensions = ('.npy',) if modality == 'lidar' or modality.endswith(('_bbs', '_depth_m')) else (('.npy', '.png') if modality.endswith('_depth') else ('.png',))
    extension = '/'.join(extensions)
    try:
        if folder.is_symlink():
            issues.append(_issue('unsafe_path', '심볼릭 링크 대신 실제 데이터 폴더가 필요합니다.',
                                 'A real data folder is required instead of a symbolic link.'))
            return result
        if not folder.exists():
            return result
        if not folder.is_dir():
            issues.append(_issue('unsupported_file', '모달리티 경로가 폴더가 아닙니다.',
                                 'The modality path is not a directory.'))
            return result
        with os.scandir(str(folder)) as entries:
            for entry in entries:
                # The original glob('*') also omits dotfiles.
                if entry.name.startswith('.'):
                    continue
                if entry.is_symlink():
                    issues.append(_issue('unsafe_path', '심볼릭 링크 대신 실제 데이터 파일이 필요합니다.',
                                         'A real data file is required instead of a symbolic link.', file=entry.name))
                    continue
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or Path(entry.name).suffix.lower() not in extensions:
                    # The research loader globs every visible entry, so ignoring
                    # an extra text file/directory could allow index corruption.
                    issues.append(_issue('unsupported_file', '{} 파일만 있는 폴더가 필요합니다.'.format(extension),
                                         'The folder must contain only {} files.'.format(extension), file=entry.name))
                    continue
                frames.add(Path(entry.name).stem)
                if info.st_size <= 0:
                    issues.append(_issue('empty_file', '저장이 완료되지 않은 빈 파일이 있습니다.',
                                         'An empty file has not finished saving.', file=entry.name))
    except OSError:
        issues.append(_issue('unreadable_data', '파일 목록이나 크기를 읽을 수 없습니다. 경로와 접근 권한을 확인하세요.',
                             'Cannot read the file inventory or size. Check the path and access permissions.'))
    # Keep error responses bounded even for a directory full of invalid entries.
    if len(issues) > 8:
        del issues[8:]
    return result


def _aligned(scene, modality):
    inventory = _inventory(scene, modality)
    if inventory['issues']:
        return [_located(issue, scene, modality) for issue in inventory['issues']]
    actual, expected = inventory['frames'], scene['front']
    if not actual:
        return [_located(_issue('missing_modality', '필요한 {} 파일이 없습니다.'.format(modality),
                                 'Required {} files are missing.'.format(modality)), scene, modality)]
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        return [_located(_issue('frame_mismatch',
                                 'front_cam_rgb와 frame ID가 다릅니다 (누락 {}개, 추가 {}개).'.format(len(missing), len(extra)),
                                 'Frame IDs differ from front_cam_rgb ({} missing, {} extra).'.format(len(missing), len(extra)),
                                 missing_count=len(missing), extra_count=len(extra),
                                 missing_examples=missing[:3], extra_examples=extra[:3]), scene, modality)]
    return []


def _aligned_choices(scene, modalities):
    """Raw formats may be partially backfilled; loader falls back per frame."""
    frames, issues = set(), []
    for modality in modalities:
        inventory = _inventory(scene, modality)
        frames.update(inventory['frames'])
        issues.extend(_located(issue, scene, modality) for issue in inventory['issues'])
    if issues: return issues
    expected = scene['front']
    if not frames:
        return [_located(_issue('missing_modality', '필요한 {} 파일이 없습니다.'.format('/'.join(modalities)),
                                 'Required {} files are missing.'.format('/'.join(modalities))), scene)]
    if frames != expected:
        missing, extra = sorted(expected - frames), sorted(frames - expected)
        return [_located(_issue('frame_mismatch', 'Frame ID 누락 {}개, 추가 {}개'.format(len(missing),len(extra)),
                                 'Frame IDs: {} missing, {} extra'.format(len(missing),len(extra)),
                                 missing_count=len(missing),extra_count=len(extra),missing_examples=missing[:3],extra_examples=extra[:3]),scene)]
    return []


def _scan(root, settings):
    scenes, common, splits = [], [], {}
    for split, key in SPLITS:
        summary = {'path': str(settings.get(key, 'data/' + split)), 'scenes': 0,
                   'samples': 0, 'available': False}
        splits[split] = summary
        try:
            path = _split_path(root, settings.get(key, 'data/' + split))
        except (ValueError, TypeError, OSError, RuntimeError):
            common.append(_issue('invalid_dataset_path', '{}: 프로젝트 내부의 실제 데이터 경로를 지정하세요.'.format(split),
                                 '{}: Select a real dataset path inside the project.'.format(split), split=split))
            continue
        split_issues, found = [], []
        try:
            if path.is_dir():
                for directory, dirs, filenames in os.walk(str(path), followlinks=False):
                    base = Path(directory)
                    dirs[:] = sorted(name for name in dirs if not name.startswith('.'))
                    for name in list(dirs):
                        if (base / name).is_symlink():
                            split_issues.append(_issue('unsafe_path', '{}: 장면 심볼릭 링크는 사용할 수 없습니다.'.format(split),
                                                       '{}: Scene symbolic links are not supported.'.format(split), split=split, scene=name))
                            dirs.remove(name)
                    # A scene is identified by actual modality folders, never a
                    # town/collection directory name. Root itself may be a scene.
                    if 'front_cam_rgb' in dirs or any(name.endswith('_rgb') and name[:-4] in CAMERAS for name in dirs):
                        found.append(base)
                        dirs[:] = []
        except OSError:
            split_issues.append(_issue('unreadable_data', '{}: 장면 목록을 읽을 수 없습니다.'.format(split),
                                       '{}: Cannot read the scene list.'.format(split), split=split))
        if not found and not split_issues:
            split_issues.append(_issue('no_dataset_scenes', '{}: <분할>/<장면>/front_cam_rgb/*.png 데이터가 없습니다.'.format(split),
                                       '{}: No data under <split>/<scene>/front_cam_rgb/*.png.'.format(split), split=split))
        for scene_path in sorted(found):
            scene = {'path': scene_path, 'split': split, 'inventory': {}}
            front = _inventory(scene, 'front_cam_rgb')
            scene['front'] = front['frames']
            summary['scenes'] += 1
            summary['samples'] += len(scene['front'])
            if front['issues']:
                split_issues.extend(_located(issue, scene, 'front_cam_rgb') for issue in front['issues'])
            elif not scene['front']:
                split_issues.append(_located(_issue('front_frames_required', '표본 기준인 front_cam_rgb PNG 프레임이 없습니다.',
                                                    'The required sample-reference front_cam_rgb PNG frames are missing.'), scene, 'front_cam_rgb'))
            scenes.append(scene)
        summary['available'] = not split_issues
        common.extend(split_issues)
    return scenes, common, splits


def training_options(root, settings, sensor_use=None, task_settings=None):
    """Return bilingual availability for actual train/val files and draft inputs.

    `settings` may be a copied settings object with validated draft dataset paths.
    Neither it nor any file is modified. The default input is front_cam only.
    Tasks depend on the selected input cameras; an unrelated camera's missing
    labels do not disable a task, while the complete launch validator still
    checks selected loader inputs, calibration and image dimensions.
    """
    if sensor_use is None:
        sensor_use = {'front_cam': True}
    if not isinstance(sensor_use, dict) or any(key not in SENSORS or type(value) is not bool
                                              for key, value in sensor_use.items()):
        raise ValueError('sensor_use는 지원 센서 이름과 true/false 값의 객체여야 합니다.')
    advanced = normalize_task_settings(task_settings)
    selected = {key: sensor_use.get(key, False) for key in SENSORS}
    cameras = [key for key in CAMERAS if selected[key]]
    scenes, common, splits = _scan(Path(root).resolve(), settings)
    sensors = {}
    for sensor in SENSORS:
        modality = sensor + '_rgb' if sensor in CAMERAS else 'lidar'
        issues = list(common)
        if not common:
            for scene in scenes:
                issues.extend(_aligned(scene, modality))
        sensors[sensor] = _option(issues, ('front_cam_rgb', modality))
    dependencies = list(common)
    if not dependencies and not selected['front_cam']:
        dependencies.append(_issue('front_camera_required', 'front_cam을 선택하세요. 기존 로더의 표본 기준입니다.',
                                   'Select front_cam; the existing loader uses it as the sample reference.'))
    unavailable = [key for key in SENSORS if selected[key] and not sensors[key]['available']]
    if not dependencies and unavailable:
        dependencies.append(_issue('selected_sensor_unavailable', '선택한 입력 센서의 데이터가 부족합니다: {}.'.format(', '.join(unavailable)),
                                   'Selected input sensor data is incomplete: {}.'.format(', '.join(unavailable)), sensors=unavailable))
        for key in unavailable:
            dependencies.extend(sensors[key]['details'])
    tasks = {}
    for task in TASKS:
        issues = list(dependencies)
        requirements = []
        if not issues:
            for scene in scenes:
                if task == 'BEV':
                    # Only selected heads are read. Native masks are preferred;
                    # palette BEV is a fallback for DAS/Lane only.
                    layers = advanced['bev']['layers']
                    for layer in layers:
                        native = {'das':'das','lane':'lane','vehicle':'vehicle_mask','walker':'walker_mask'}[layer]
                        modalities = (native, 'bev') if layer in ('das','lane') else (native,)
                        if _inventory(scene, native)['frames'] == scene['front']:
                            modalities = (native,)
                        requirements.extend(modalities)
                        issues.extend(_aligned_choices(scene, modalities))
                else:
                    suffix = {'Seg': 'seg', 'Depth': 'depth', 'od': 'bbs'}[task]
                    for camera in cameras:
                        modality = camera + '_' + suffix
                        raw = camera + ('_seg_ids' if task == 'Seg' else '_depth_m')
                        modalities = (raw, modality) if (task == 'Depth' or (task == 'Seg' and advanced['segmentation']['mode']=='carla')) else (modality,)
                        requirements.extend(modalities)
                        issues.extend(_aligned_choices(scene, modalities) if len(modalities)>1 else _aligned(scene, modality))
        if not issues and task_settings is not None:
            selection = {'Seg':advanced['segmentation']['class_ids'], 'od':advanced['detection']['class_ids'], 'BEV':advanced['bev']['layers']}.get(task, [True])
            if task=='Seg' and advanced['segmentation']['mode']=='legacy': selection=[True]
            if not selection:
                issues.append(_issue('no_selected_classes', '실제 데이터가 있는 학습 클래스/레이어를 선택하세요.',
                                     'Select training classes/layers that exist in the dataset.'))
        tasks[task] = _option(issues, sorted(set(requirements)))
    return {'sensors': sensors, 'tasks': tasks, 'selected_sensors': selected,
            'selected_cameras': cameras, 'splits': splits,
            'scope': 'file_presence_and_frame_alignment',
            'requires_full_preflight': True}
