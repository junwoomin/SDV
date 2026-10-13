"""Read actual SDV experiment and collection artifacts without importing CUDA.

The original SDV_web metrics_history.json contract is retained alongside SDV's
train_metrics.csv and results.log. A file being present is never treated as
proof that a process completed. All user-facing IDs are opaque, and downloads
are constrained to files beneath the selected experiment (including symlinks).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any
import zipfile

import yaml

from sdv.training.metric_registry import metric_definition, metric_registry

MAX_TEXT = 16 * 1024 * 1024
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
ARTIFACT_SUFFIXES = IMAGE_SUFFIXES | {'.csv', '.json', '.jsonl', '.log', '.txt', '.yaml', '.yml', '.pth', '.pt', '.ckpt', '.npz', '.npy', '.pdf'}
RUN_MARKERS = {'train_metrics.csv', 'metrics_history.json', 'results.log', 'events.jsonl', 'train_config.yaml'}
CAMERAS = ('front_cam', 'left_cam', 'right_cam', 'rear_cam', 'rear_left_cam', 'rear_right_cam')
TRAIN_CONFIG_PATHS = ('config/train_config.yaml', 'output/train_config.yaml', 'train_config.yaml')


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def safe_child(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, and symlinks escaping a selected root."""
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts or '\\' in relative:
        raise ValueError('상위 경로나 절대 경로는 사용할 수 없습니다.')
    result = (root / candidate).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError:
        raise ValueError('허용된 폴더 밖의 파일은 읽을 수 없습니다.') from None
    return result


def read_text(path: Path, limit=MAX_TEXT):
    if path.stat().st_size > limit:
        raise ValueError(f'{path.name}: 파일이 너무 큽니다 (최대 {limit // 1024 // 1024} MB).')
    return path.read_text(encoding='utf-8', errors='replace')


def _identifier(path: Path):
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]


def _relative(path: Path, root: Path):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _walk(root: Path, depth=4):
    """Bounded traversal. Never follow directory symlinks or read hidden trees."""
    if not root.is_dir():
        return
    for folder, dirs, files in os.walk(root, followlinks=False):
        path = Path(folder)
        level = len(path.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (path / d).is_symlink()) if level < depth else []
        yield path, dirs, files


def _series_from_csv(path: Path):
    rows = list(csv.DictReader(io.StringIO(read_text(path))))
    series = {}
    for index, row in enumerate(rows):
        x = finite(row.get('epoch'))
        if x is None:
            continue  # Do not invent an epoch when a malformed row lacks one.
        for field, value in row.items():
            if not field or field in {'epoch', 'step', 'timestamp'}:
                continue
            y = finite(value)
            if y is None:
                continue
            split, _, name = field.partition('_')
            if split not in {'train', 'val', 'test'}:
                split, name = 'observed', field
            key = (name, split)
            item = series.setdefault(key, {'name': name, 'split': split, 'x_label': 'epoch', 'points': [], 'source': path.name, 'granularity': 'epoch'})
            item['points'].append({'x': x, 'y': y})
    return list(series.values())


def _series_from_history(path: Path):
    """Compatibility with sdv_web/core/metric_utils.py's original JSON format."""
    blob = json.loads(read_text(path))
    result = []
    for task, metrics in blob.get('tasks', {}).items():
        if not isinstance(metrics, dict):
            continue
        for name, record in metrics.items():
            if not isinstance(record, dict) or not isinstance(record.get('x'), list):
                continue
            for split in ('train', 'val'):
                values = record.get(split)
                if not isinstance(values, list):
                    continue
                points = [{'x': finite(x), 'y': finite(y)} for x, y in zip(record['x'], values) if finite(x) is not None and finite(y) is not None]
                if points:
                    result.append({'name': f'{task}/{name}', 'split': split, 'x_label': 'recorded x', 'points': points, 'source': path.name, 'granularity': 'recorded'})
    return result


def _series_from_log(path: Path):
    """Legacy records contain batch losses, not epoch means. Keep that explicit."""
    content = read_text(path)
    series = {}
    number = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'
    for line in content.splitlines():
        marker = re.search(r'\b(TRAIN|VAL)\[(\d+)\]', line)
        if not marker:
            continue
        split, epoch = marker.group(1).lower(), int(marker.group(2))
        # TRAIN includes a recorded global counter. VAL includes only an epoch.
        step = re.search(r'TRAIN\[\d+\]\s*\[(\d+)\]', line)
        x = int(step.group(1)) if step else epoch
        x_label = 'step' if step else 'epoch (logged batch value)'
        for metric, raw in re.findall(r'\b([A-Za-z][A-Za-z_]+):?\s+(' + number + r')\b', line[marker.end():]):
            y = finite(raw)
            if y is None:
                continue
            item = series.setdefault((metric.lower(), split, x_label), {'name': metric.lower(), 'split': split, 'x_label': x_label, 'points': [], 'source': path.name, 'granularity': 'logged_batch'})
            item['points'].append({'x': x, 'y': y, 'epoch': epoch})
    return list(series.values())


def _series_from_events(path: Path):
    series = {}
    for line in read_text(path).splitlines():
        try:
            item = json.loads(line[len('SDV_EVENT '):] if line.startswith('SDV_EVENT ') else line)
        except (ValueError, TypeError):
            continue  # A running process may leave an incomplete trailing line.
        if not isinstance(item, dict):
            continue
        metrics = item.get('metrics')
        if not isinstance(metrics, dict):
            continue
        event = item.get('event', item.get('type'))
        is_step = event == 'training_step' or item.get('granularity') in {'batch', 'step'}
        if is_step:
            if finite(item.get('global_step')) is not None:
                x, x_label = finite(item['global_step']), 'global step'
            else:
                x, x_label = finite(item.get('step')), f"step (epoch {item.get('epoch', '?')})"
            granularity = 'batch'
        else:
            x = finite(item.get('epoch', item.get('step')))
            x_label = 'epoch' if 'epoch' in item else 'step'
            granularity = 'epoch' if event == 'training_epoch' else item.get('granularity', 'recorded')
        if x is None:
            continue
        for metric, value in metrics.items():
            y = finite(value)
            if y is None:
                continue
            split, _, name = str(metric).partition('_')
            if split not in {'train', 'val', 'test'}:
                split, name = item.get('split', 'observed'), str(metric)
            record = series.setdefault((name, split, x_label, granularity), {'name': name, 'split': split, 'x_label': x_label, 'points': [], 'source': path.name, 'granularity': granularity})
            record['points'].append({'x': x, 'y': y})
    return list(series.values())


def _series_best(series):
    """Choose the best *epoch* observation before graph downsampling.

    Batch losses, arbitrary recorded x values and unknown metrics cannot be
    interpreted as epoch-level model quality. Equal best values retain the
    earliest epoch, including logs whose rows were recorded out of order.
    """
    if series.get('granularity') != 'epoch' or series.get('x_label') != 'epoch':
        return None
    key = '{}_{}'.format(series['split'], series['name'])
    definition = metric_definition(key)
    if definition is None:
        return None
    direction = definition['direction']
    if direction not in ('min', 'max'):
        return None
    best = None
    for point in series['points']:
        value, epoch = finite(point.get('y')), finite(point.get('x'))
        if value is None or epoch is None:
            continue
        improved = (best is None or
                    (value < best['value'] if direction == 'min' else value > best['value']) or
                    (value == best['value'] and epoch < best['epoch']))
        if improved:
            best = {'value': value, 'epoch': int(epoch) if epoch.is_integer() else epoch,
                    'direction': direction}
    return best


class Catalog:
    def __init__(self, project_root, settings):
        self.root = Path(project_root).resolve()
        self._settings = settings
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._prepare_lock = threading.Lock()

    @property
    def settings(self):
        return self._settings() if callable(self._settings) else self._settings

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def _cached(self, key, fn, seconds=3):
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < seconds:
                return cached[1]
        value = fn()
        with self._cache_lock:
            self._cache[key] = time.monotonic(), value
        return value

    def invalidate(self):
        with self._cache_lock:
            self._cache.clear()

    def _runs(self):
        roots = {self.path(self.settings.get('results_root', 'results')), self.root / 'Training_result', self.root / 'sdv_web/runs'}
        found = {}
        for root in roots:
            for folder, dirs, files in _walk(root, depth=3):
                job_root = folder.name.startswith('job_') and ('exit.json' in files or ('config' in dirs and (folder / 'config/settings.json').is_file()))
                if job_root and (folder/'config/experiment.json').is_file():
                    try:
                        if json.loads(read_text(folder/'config/experiment.json')).get('kind')=='collection':
                            dirs[:]=[]
                            continue
                    except (OSError,ValueError,TypeError):pass
                if job_root or RUN_MARKERS.intersection(files) or any(Path(f).suffix in {'.pt', '.pth', '.ckpt'} for f in files):
                    found[_identifier(folder)] = folder
                    dirs[:] = []
        return found

    def run_path(self, experiment_id):
        result = self._runs().get(experiment_id)
        if result is None:
            raise FileNotFoundError('실험을 찾을 수 없습니다.')
        return result

    def _training_config(self, run, warnings=None, require_model_conf=False):
        """Read the experiment snapshot; never substitute current UI settings."""
        for relative in TRAIN_CONFIG_PATHS:
            try:
                path = safe_child(run, relative)
                if path.is_file():
                    config = yaml.safe_load(read_text(path, 2 * 1024 * 1024))
                    if isinstance(config, dict):
                        if require_model_conf and not isinstance(config.get('model_conf'), dict):
                            continue
                        return config, relative
                    if warnings is not None:
                        warnings.append('{}: 학습 설정은 객체여야 합니다.'.format(relative))
            except (OSError, ValueError, yaml.YAMLError) as error:
                if warnings is not None:
                    warnings.append('{}: {}'.format(relative, error))
        return {}, None

    def _configuration_summary(self, run, warnings):
        config, source = self._training_config(run, warnings)
        model = config.get('model') if isinstance(config.get('model'), dict) else {}
        optimizers = config.get('opt_checkboxes')
        if isinstance(optimizers, dict):
            selected = [str(name) for name, value in optimizers.items() if value is True]
            optimizer = selected[0] if len(selected) == 1 else None
            if len(selected) > 1:
                warnings.append('학습 설정에 여러 옵티마이저가 선택되어 있습니다. 실행 옵티마이저를 추정하지 않습니다.')
        else:
            optimizer = config.get('optimizer')
            optimizer = optimizer if isinstance(optimizer, str) else None
        sensors = config.get('sensor_use') if isinstance(config.get('sensor_use'), dict) else {}
        tasks = config.get('model_conf') if isinstance(config.get('model_conf'), dict) else {}

        def selected_keys(mapping, ordered):
            return [key for key in ordered if mapping.get(key) is True] + sorted(
                str(key) for key, value in mapping.items() if key not in ordered and value is True)

        batch = finite(config.get('batch'))
        epochs = finite(config.get('Epochs'))
        summary = {
            'optimizer': optimizer,
            'encoder': model.get('encoder') if isinstance(model.get('encoder'), str) else None,
            'decoder': model.get('decoder') if isinstance(model.get('decoder'), str) else None,
            'bev_decoder': model.get('bev_decoder') if isinstance(model.get('bev_decoder'), str) else None,
            'learning_rate': finite(config.get('LR')),
            'batch': int(batch) if batch is not None and batch.is_integer() else batch,
            'configured_epochs': int(epochs) if epochs is not None and epochs.is_integer() else epochs,
            'selected_sensors': selected_keys(sensors, (*CAMERAS, 'lidar')),
            'tasks': selected_keys(tasks, ('BEV', 'Seg', 'Depth', 'od')),
            'config_source': source}
        name = None
        for relative in ('config/experiment.json', 'experiment.json'):
            try:
                path = safe_child(run, relative)
                if path.is_file():
                    record = json.loads(read_text(path, 2 * 1024 * 1024))
                    if isinstance(record, dict) and isinstance(record.get('name'), str) and record['name'].strip():
                        name = record['name'].strip()[:200]
                        break
            except (OSError, ValueError) as error:
                warnings.append('{}: {}'.format(relative, error))
        return summary, name

    def metrics(self, run):
        warnings = []
        config, _ = self._training_config(run, require_model_conf=True)
        enabled_tasks = config.get('model_conf') if isinstance(config.get('model_conf'), dict) else None
        # Epoch CSV is authoritative; logs are only a fallback for old/running runs.
        for name, parser in [('output/train_metrics.csv', _series_from_csv), ('train_metrics.csv', _series_from_csv), ('output/metrics_history.json', _series_from_history), ('metrics_history.json', _series_from_history), ('output/events.jsonl', _series_from_events), ('events.jsonl', _series_from_events), ('output/results.log', _series_from_log), ('results.log', _series_from_log)]:
            path = run / name
            if not path.is_file():
                continue
            try:
                path = safe_child(run, name)
                result = parser(path)
                if result:
                    if enabled_tasks is not None:
                        visible = []
                        for series in result:
                            metric = series['name'].lower()
                            task = 'BEV' if any(word in metric for word in ('bev', 'vehicle', 'walker', 'lane', 'das')) else 'Seg' if 'seg' in metric else 'Depth' if any(word in metric for word in ('depth', 'abs_rel')) else 'od' if 'od_' in metric or metric.startswith('od/') else None
                            if task is None or enabled_tasks.get(task, True):
                                visible.append(series)
                        result = visible
                    for series in result:
                        series['source'] = name
                        best = _series_best(series)
                        if best is not None:
                            series['best'] = best
                            definition = metric_definition('{}_{}'.format(series['split'], series['name']),config)
                            series['label'] = definition['label']
                            series['direction'] = definition['direction']
                            series['task'] = definition['task']
                            series['unit'] = definition['unit']
                            series['maximum'] = definition.get('maximum')
                            series['detail'] = definition.get('detail',False)
                        if series['name'].lower() == 'od_iou':
                            series['description'] = 'Anchor–GT IoU (legacy): 양성 anchor와 정답 상자의 겹침. 예측 상자의 IoU 또는 detection mAP가 아닙니다.'
                            if series['description'] not in warnings:
                                warnings.append(series['description'])
                        series['total_points'] = len(series['points'])
                        if len(series['points']) > 2000:
                            stride = math.ceil(len(series['points']) / 1999)
                            selected = series['points'][::stride]
                            if selected[-1] != series['points'][-1]:
                                selected.append(series['points'][-1])
                            series['points'] = selected
                            series['sample_stride'] = stride
                    if name.endswith('results.log'):
                        warnings.append('이 실험은 레거시 로그의 기록된 batch 값을 표시합니다. epoch 평균 Loss로 해석하지 마세요.')
                    return result, warnings
            except (OSError, ValueError, TypeError, AttributeError, csv.Error) as error:
                warnings.append(f'{name} 읽기 오류: {error}')
        return [], warnings

    def _artifacts(self, run):
        result = []
        for folder, dirs, files in _walk(run, depth=5):
            for name in sorted(files):
                if Path(name).suffix.lower() not in ARTIFACT_SUFFIXES:
                    continue
                relative = str((folder / name).relative_to(run))
                try:
                    path = safe_child(run, relative)
                    stat = path.stat()
                except (OSError, ValueError):
                    continue
                if not path.is_file():
                    continue
                result.append({'name': name, 'path': relative, 'size': stat.st_size, 'modified': stat.st_mtime, 'kind': 'checkpoint' if path.suffix in {'.pth', '.pt', '.ckpt'} else 'image' if path.suffix in IMAGE_SUFFIXES else 'file'})
                if len(result) >= 3000:
                    return result
        return result

    def _summary(self, identifier, path):
        metrics, warnings = self.metrics(path)
        artifacts = self._artifacts(path)
        latest = {f"{m['split']}_{m['name']}": m['points'][-1]['y'] for m in metrics if m['points']}
        epochs = max((p['x'] for m in metrics if m['x_label'] == 'epoch' for p in m['points']), default=None)
        modified = max((a['modified'] for a in artifacts), default=path.stat().st_mtime)
        configuration, saved_name = self._configuration_summary(path, warnings)
        best = {'{}_{}'.format(m['split'], m['name']): m['best'] for m in metrics if 'best' in m}
        return {'id': identifier, 'name': saved_name or path.name, 'job_id': path.name[len('job_'):] if path.name.startswith('job_') else None, 'path': _relative(path, self.root), 'modified': modified, 'epochs': epochs, 'latest_metrics': latest, 'best_metrics': best, 'checkpoints_count': sum(a['kind'] == 'checkpoint' for a in artifacts), 'source': metrics[0]['source'] if metrics else None, 'status': 'Recorded', 'warnings': warnings, **configuration}, metrics, artifacts

    def experiments(self):
        def scan():
            result = []
            for identifier, path in self._runs().items():
                try:
                    result.append(self._summary(identifier, path)[0])
                except OSError:
                    continue
            return sorted(result, key=lambda row: row['modified'], reverse=True)
        return self._cached(('experiments', str(self.settings.get('results_root'))), scan)

    def experiment(self, identifier):
        path = self.run_path(identifier)
        row, metrics, artifacts = self._summary(identifier, path)
        configs = {}
        for name in ('train_config.yaml', 'sensor_config.yaml', 'config.yaml', 'config.json', 'experiment.json', 'config/train_config.yaml', 'config/sensor_config.yaml', 'config/settings.json', 'config/experiment.json', 'output/train_config.yaml', 'output/sensor_config.yaml'):
            try:
                candidate = safe_child(path, name)
                if candidate.is_file():
                    configs[name] = yaml.safe_load(read_text(candidate, 2 * 1024 * 1024))
            except (OSError, ValueError, yaml.YAMLError) as error:
                row['warnings'].append(f'{name}: {error}')
        training_config,_=self._training_config(path)
        return {**row, 'metrics': metrics, 'metric_registry':metric_registry(training_config), 'config': configs, 'artifacts': artifacts, 'checkpoints': [a for a in artifacts if a['kind'] == 'checkpoint'], 'notice': '저장된 결과입니다. 파일 존재만으로 프로세스 완료를 추정하지 않습니다.'}

    def artifact(self, identifier, relative):
        path = safe_child(self.run_path(identifier), relative)
        if path.suffix.lower() not in ARTIFACT_SUFFIXES or not path.is_file():
            raise FileNotFoundError('다운로드할 결과 파일을 찾을 수 없습니다.')
        return path

    def export(self, identifier):
        """Export numeric results/config/logs. Checkpoints download individually."""
        path = self.run_path(identifier)
        directory = self.root / '.runtime/studio/exports'
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'{identifier}-results.zip'
        fd, temporary = tempfile.mkstemp(prefix='export-', suffix='.zip', dir=directory)
        os.close(fd)
        try:
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                total = 0
                for artifact in self._artifacts(path):
                    if Path(artifact['name']).suffix not in {'.csv', '.json', '.jsonl', '.yaml', '.yml', '.log', '.txt'}:
                        continue
                    source = self.artifact(identifier, artifact['path'])
                    total += source.stat().st_size
                    if total > 256 * 1024 * 1024:
                        raise ValueError('내보내기 256 MB 제한을 초과했습니다. 개별 파일을 다운로드하세요.')
                    archive.write(source, artifact['path'])
                archive.writestr('export_manifest.json', json.dumps({'experiment': identifier, 'source': str(path), 'exported_at': time.time(), 'note': '실제 결과/설정/로그만 포함. 모델 및 이미지는 개별 다운로드.'}, ensure_ascii=False, indent=2))
            os.replace(temporary, destination)
        finally:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        return destination

    def _scene_paths(self):
        found = {}
        roots = [(self.path(self.settings.get('data_root', 'data/collections')), 'collection'), (self.path(self.settings.get('train_data', 'data/train')), 'train'), (self.path(self.settings.get('val_data', 'data/val')), 'val'), (self.root / 'data/prepared', 'prepared')]
        # Also discover pre-existing SDV data/TownXX/scene_* collections.
        if (self.root / 'data').is_dir():
            roots += [(p, 'collection') for p in (self.root / 'data').glob('Town*') if p.is_dir() and not p.is_symlink()]
        for root, kind in roots:
            for folder, dirs, files in _walk(root, depth=4):
                if 'front_cam_rgb' in dirs or ('info' in dirs and ('das' in dirs or 'bev' in dirs)):
                    identifier = _identifier(folder)
                    found.setdefault(identifier, (folder, kind))
                    dirs[:] = []
        return found

    def scene_path(self, identifier):
        result = self._scene_paths().get(identifier)
        if result is None:
            raise FileNotFoundError('수집 장면을 찾을 수 없습니다.')
        return result[0]

    def _scene_summary(self, identifier, path, kind):
        channels, counts, previews = [], {}, {}
        for child in sorted(path.iterdir()):
            if child.is_dir() and not child.is_symlink():
                files = [p for p in child.iterdir() if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_SUFFIXES | {'.npy', '.json'}]
                if files:
                    counts[child.name] = len(files)
                    if any(p.suffix.lower() in IMAGE_SUFFIXES for p in files):
                        channels.append(child.name)
                        latest = max((p for p in files if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name)
                        previews[child.name] = {'filename': latest.name, 'modified': latest.stat().st_mtime}
        samples = counts.get('front_cam_rgb', counts.get('info', 0))
        sensor = self.sensor_config_path(path)
        blockers = []
        if not counts.get('front_cam_rgb'):
            blockers.append('기존 학습 로더의 기준 카메라 front_cam_rgb가 없습니다.')
        snapshot = path / 'sensor_config.yaml'
        if not snapshot.is_file() or snapshot.is_symlink():
            blockers.append('수집 당시의 sensor_config.yaml이 필요합니다.')
        return {'id': identifier, 'name': path.name, 'path': _relative(path, self.root), 'kind': kind, 'town': next((p for p in path.parts if p.startswith('Town')), None), 'samples': samples, 'counts': counts, 'channels': channels, 'preview_frames': previews, 'modified': max([path.stat().st_mtime] + [p['modified'] for p in previews.values()]), 'sensor_config': str(sensor) if sensor else None, 'calibration_verified': snapshot.is_file() and not snapshot.is_symlink(), 'blockers': blockers, 'label_format': 'legacy_palette' if counts.get('bev') else 'native_binary' if counts.get('das') and counts.get('lane') else 'missing'}

    def datasets(self):
        def scan():
            result = []
            for identifier, (path, kind) in self._scene_paths().items():
                try:
                    result.append(self._scene_summary(identifier, path, kind))
                except OSError:
                    continue
            return sorted(result, key=lambda item: item['modified'], reverse=True)
        return self._cached(('datasets', str(self.settings.get('data_root')), str(self.settings.get('train_data')), str(self.settings.get('val_data'))), scan, 5)

    def sensor_config_path(self, scene):
        # New collector snapshots its sensor config per scene. For legacy data,
        # retain the original data/sensor_config.yaml convention explicitly.
        candidates = [scene / 'sensor_config.yaml', scene.parent / 'sensor_config.yaml', scene.parent.parent / 'sensor_config.yaml']
        candidates += [self.path(self.settings.get('sensor_config') or 'data/sensor_config.yaml')]
        return next((p for p in candidates if p.is_file() and not p.is_symlink()), None)

    def preview(self, identifier, channel='front_cam_rgb'):
        if not re.fullmatch(r'[A-Za-z0-9_]+', channel):
            raise ValueError('지원하는 센서 채널을 선택하세요.')
        scene = self.scene_path(identifier)
        folder = safe_child(scene, channel)
        def latest():
            if not folder.is_dir():
                raise FileNotFoundError('저장된 센서 프레임이 없습니다.')
            candidates = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file() and not p.is_symlink()), key=lambda p: p.name, reverse=True)
            # Ignore a frame currently being written; no simulation-side encode.
            now = time.time()
            for path in candidates:
                stat = path.stat()
                if stat.st_size and now - stat.st_mtime >= .25:
                    return safe_child(scene, str(path.relative_to(scene)))
            raise FileNotFoundError('완전히 저장된 센서 프레임을 기다리는 중입니다.')
        return self._cached(('preview', identifier, channel), latest, 2)

    def trajectory(self, identifier):
        scene = self.scene_path(identifier)
        info = safe_child(scene, 'info')
        files = sorted(info.glob('*.json')) if info.is_dir() else []
        stride = max(1, math.ceil(len(files) / 2000))
        points = []
        for file in files[::stride]:
            try:
                value = json.loads(read_text(safe_child(scene, str(file.relative_to(scene))), 256 * 1024))
                loc = value.get('ego_vehicle_loc', {})
                if finite(loc.get('x')) is None or finite(loc.get('y')) is None:
                    continue
                points.append({'frame': file.stem, 'x': finite(loc['x']), 'y': finite(loc['y']), 'z': finite(loc.get('z')), 'speed': finite(value.get('current_speed')), 'collision': value.get('collision')})
            except (OSError, ValueError, AttributeError, TypeError):
                continue
        return {'scene_id': identifier, 'points': points, 'count': len(points), 'source_frames': len(files), 'sample_stride': stride, 'coordinate_system': 'CARLA world (metres)', 'speed_unit': 'km/h', 'source': 'info/*.json'}

    def prepare_dataset(self, name, train_scenes, val_scenes, train_config=None):
        from .dataset_tools import prepare_dataset
        if not self._prepare_lock.acquire(blocking=False):
            raise ValueError('다른 데이터셋을 준비 중입니다. 완료 후 다시 시도하세요.')
        try:
            result = prepare_dataset(self, name, train_scenes, val_scenes, train_config)
            self.invalidate()
            return result
        except (OSError, yaml.YAMLError, KeyError, TypeError) as error:
            raise ValueError(f'데이터셋 준비 실패: {error}') from error
        finally:
            self._prepare_lock.release()
