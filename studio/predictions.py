"""Bounded, CPU-only metadata access to real SDV prediction image histories."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import threading

from .catalog import safe_child


MAX_MANIFEST_BYTES = 32 * 1024 * 1024
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
_CACHE_MAX_BYTES = 4 * 1024 * 1024
_CACHE_MAX_ENTRIES = 20000
_CACHE_ITEMS = 2
_MANIFEST_CACHE = OrderedDict()
_ENTRY_CACHE = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _identity(path):
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)


def _cached(cache, key):
    with _CACHE_LOCK:
        result = cache.get(key)
        if result is not None: cache.move_to_end(key)
        return result


def _store(cache, key, value):
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_ITEMS: cache.popitem(last=False)


def _relative_image(relative):
    if not isinstance(relative, str) or len(relative) > 1024:
        return None
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('상위 경로나 절대 경로는 사용할 수 없습니다.')
    return relative if path.suffix.lower() in IMAGE_SUFFIXES else None


def _source(run_or_output):
    root = Path(run_or_output).resolve()
    if (root / 'predictions' / 'index.json').is_file():
        return root, root, ''
    if (root / 'output' / 'predictions' / 'index.json').is_file():
        return root, root / 'output', 'output/'
    # Old runs may likewise be supplied by their job root or output folder.
    if (root / 'output').is_dir():
        return root, root / 'output', 'output/'
    return root, root, ''


def _listed_image(root, relative):
    if not isinstance(relative, str) or len(relative) > 1024:
        return None
    path = safe_child(root, relative)
    return relative if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES else None


def _legacy(root, output, prefix):
    entries = []
    for folder_name, task in (('save_samples_seg', 'Seg'), ('save_samples_depth', 'Depth'), ('save_samples_od', 'od')):
        pred = output / folder_name / 'pred'
        if not pred.is_dir() or pred.is_symlink():
            continue
        for path in sorted(pred.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = prefix + path.relative_to(output).as_posix()
            if not _listed_image(root, relative):
                continue
            gt = prefix + (Path(folder_name) / 'gt' / path.name).as_posix()
            images = {'pred': relative}
            if _listed_image(root, gt): images['gt'] = gt
            identifier = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]
            entries.append({'id': 'legacy:' + identifier, 'legacy': True, 'task': task,
                            'split': 'unknown', 'epoch': None, 'global_step': None,
                            'sample_id': path.stem, 'relative_frame': None,
                            'dataset_fingerprint': None, 'camera': None,
                            'images': images, 'source': 'existing_saved_prediction'})
    return entries


def _manifest(root, output, prefix):
    path = output / 'predictions' / 'index.json'
    # Check the metadata path itself, including an escaping symlink.
    safe_child(root, prefix + 'predictions/index.json')
    identity = _identity(path)
    if identity[2] > MAX_MANIFEST_BYTES:
        raise ValueError('예측 이미지 목록이 너무 큽니다 (최대 32 MB).')
    key = (str(path.resolve()), identity)
    cached = _cached(_MANIFEST_CACHE, key)
    if cached is not None: return cached
    try:
        blob = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError):
        raise ValueError('예측 이미지 목록 JSON을 읽지 못했습니다.') from None
    if not isinstance(blob, dict) or blob.get('version') != 1 or not isinstance(blob.get('entries'), list):
        raise ValueError('예측 이미지 목록 형식이 잘못되었습니다.')
    if identity[2] <= _CACHE_MAX_BYTES and len(blob['entries']) <= _CACHE_MAX_ENTRIES:
        _store(_MANIFEST_CACHE, key, blob)
    return blob


def _load(run_or_output):
    root, output, prefix = _source(run_or_output)
    path = output / 'predictions' / 'index.json'
    if not path.is_file():
        return root, _legacy(root, output, prefix), {}, True
    identity = _identity(path)
    key = (str(root), str(path.resolve()), identity)
    # Even a cache hit validates the index path, so replacing it by a symlink
    # cannot turn historical metadata into an arbitrary-file read.
    safe_child(root, prefix + 'predictions/index.json')
    if identity[2] > MAX_MANIFEST_BYTES:
        raise ValueError('예측 이미지 목록이 너무 큽니다 (최대 32 MB).')
    cached = _cached(_ENTRY_CACHE, key)
    if cached is not None: return root, cached[0], cached[1], False
    blob = _manifest(root, output, prefix)
    entries = []
    for record in blob['entries']:
        if not isinstance(record, dict) or record.get('superseded'): continue
        images = record.get('images') or {name: record.get(name) for name in ('input', 'gt', 'pred')}
        if not isinstance(images, dict): continue
        converted = {}
        for name in ('input', 'gt', 'pred'):
            relative = images.get(name)
            if relative is None: continue
            relative = _relative_image(relative)
            if relative is not None: converted[name] = prefix + relative
        # Resolve/stat only a requested page below. Large histories should not
        # require tens of thousands of filesystem operations on each refresh.
        if 'pred' not in converted: continue
        entry = dict(record, images=converted, legacy=False)
        for name in ('input', 'gt', 'pred'):
            if name in converted: entry[name] = converted[name]
        entries.append(entry)
    datasets = blob.get('datasets', {})
    datasets = datasets if isinstance(datasets, dict) else {}
    if identity[2] <= _CACHE_MAX_BYTES and len(entries) <= _CACHE_MAX_ENTRIES:
        _store(_ENTRY_CACHE, key, (entries, datasets))
    return root, entries, datasets, False


def prediction_index(run_or_output, split=None, epoch=None, offset=0, limit=100,
                     task=None, sample_id=None, global_step=None):
    """Newest first, paginated. All image paths are relative to supplied root."""
    offset, limit = int(offset), int(limit)
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError('예측 이미지 조회는 offset ≥ 0, limit 1~200이어야 합니다.')
    if split is not None and split not in ('train', 'val', 'unknown'):
        raise ValueError('예측 split은 train 또는 val이어야 합니다.')
    epoch = int(epoch) if epoch is not None else None
    global_step = int(global_step) if global_step is not None else None
    if global_step is not None and global_step < 0:raise ValueError('global_step must be non-negative')
    root, entries, datasets, legacy = _load(run_or_output)
    selected = [entry for entry in entries
                if (split is None or entry.get('split') == split)
                and (epoch is None or entry.get('epoch') == epoch)
                and (global_step is None or entry.get('global_step') == global_step)
                and (task is None or entry.get('task') == task)
                and (sample_id is None or entry.get('sample_id') == sample_id)]
    selected.sort(key=lambda item: (item.get('capture_id') or 0, item.get('epoch') or 0,
                                    item.get('global_step') or 0, item.get('sample_id') or '',
                                    item.get('task') or '', item.get('camera') or ''), reverse=True)
    timeline = {}
    for item in entries:
        if (split is not None and item.get('split') != split) or (task is not None and item.get('task') != task) or (sample_id is not None and item.get('sample_id') != sample_id):continue
        if type(item.get('epoch')) is not int or type(item.get('global_step')) is not int:continue
        key = (item['epoch'],item['global_step'],item.get('capture_id'))
        timeline[key] = {key:item.get(key) for key in ('epoch','global_step','epoch_step','steps_per_epoch','capture_id')}
    timeline = sorted(timeline.values(),key=lambda item:(item['global_step'],item['epoch'],item.get('capture_id') or 0))
    fixed = datasets.get('val', {})
    page, unavailable = [], 0
    window = selected[offset:offset + limit]
    for entry in window:
        images = {name: value for name, value in entry['images'].items() if _listed_image(root, value)}
        if 'pred' not in images:
            unavailable += 1
            continue
        item = dict(entry, images=images)
        for name in ('input', 'gt', 'pred'):
            if name in images: item[name] = images[name]
            else: item.pop(name, None)
        page.append(item)
    next_offset = min(len(selected), offset + len(window))
    _,output,prefix=_source(run_or_output)
    storage={}
    budget_path=safe_child(root,prefix+'predictions/storage_status.json')
    if budget_path.is_file() and budget_path.stat().st_size<10000:
        try:storage=json.loads(budget_path.read_text())
        except (OSError,ValueError):pass
    return {'storage_budget':storage if isinstance(storage,dict) else {},'entries': page, 'timeline': timeline, 'total': len(selected), 'unavailable_entries': unavailable,
            'next_offset': next_offset, 'has_more': next_offset < len(selected),
            'offset': offset, 'limit': limit, 'datasets': datasets,
            'fixed_samples': fixed.get('samples', []) if isinstance(fixed, dict) else [],
            'dataset_fingerprint': fixed.get('dataset_fingerprint') if isinstance(fixed, dict) else None,
            'fixed_sample_count': fixed.get('count', 0) if isinstance(fixed, dict) else 0,
            'reason': fixed.get('reason', '') if isinstance(fixed, dict) else '',
            'reason_en': fixed.get('reason_en', '') if isinstance(fixed, dict) else '',
            'legacy': legacy,
            'global_steps': sorted({e['global_step'] for e in entries if type(e.get('global_step')) is int
                and (split is None or e.get('split')==split) and (epoch is None or e.get('epoch')==epoch)}),
            'epochs': sorted({entry['epoch'] for entry in entries if isinstance(entry.get('epoch'), int)}),
            'tasks': sorted({entry['task'] for entry in entries if isinstance(entry.get('task'), str)}),
            'splits': sorted({entry['split'] for entry in entries if isinstance(entry.get('split'), str)})}


def prediction_frame(run_or_output, relative):
    """Return a file only if it is an actual manifest-listed prediction image."""
    root, output, prefix = _source(run_or_output)
    # Validate even before membership testing so traversal always fails clearly.
    path = safe_child(root, relative)
    if (output / 'predictions/index.json').is_file():
        blob = _manifest(root, output, prefix)
        # Image requests do not stat every historical image. Only the requested
        # manifest-listed file needs resolving; gallery listings check existence.
        listed = set()
        for entry in blob['entries']:
            if not isinstance(entry, dict): continue
            images = entry.get('images') or {name: entry.get(name) for name in ('input', 'gt', 'pred')}
            if not isinstance(images, dict): continue
            for name in ('input', 'gt', 'pred'):
                image = images.get(name)
                if isinstance(image, str): listed.add(prefix + image)
    else:
        entries = _legacy(root, output, prefix)
        listed = {value for entry in entries for value in entry['images'].values()}
    if relative not in listed or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError('저장된 예측 이미지 목록에 없는 파일입니다.')
    return path
