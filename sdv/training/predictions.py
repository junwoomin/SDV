"""Real, repeatable training predictions without changing optimization state.

Dataset selection and manifests need only the standard library. PyTorch and
image dependencies are imported when a research worker actually renders.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
DATA_SUFFIXES = IMAGE_SUFFIXES | {'.npy', '.npz', '.json', '.yaml', '.yml'}
CLASS_COLORS = ((0, 255, 255), (255, 0, 0), (0, 255, 0),
                (0, 0, 255), (255, 255, 0), (0, 0, 0))
DETECTION_CLASSES = ('vehicles', 'pedestrians')


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def fixed_samples(dataset_root, project_root, image_front=None, maximum=4):
    """Content identity is independent of model, selected tasks and batch size.

    Every dataset modality contributes to the fingerprint. A stat-based digest
    cache avoids re-reading unchanged files on subsequent jobs, while copies of
    the same content produce the same fingerprint. Files escaping the dataset
    through symlinks are rejected.
    """
    root, project = Path(dataset_root).resolve(), Path(project_root).resolve()
    folder = project / '.runtime' / 'studio' / 'prediction_samples'
    cache_key = hashlib.sha256(str(root).encode('utf-8')).hexdigest()
    cache_path = folder / ('inventory_' + cache_key + '.json')
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cache = {}
    inventory, fingerprint = {}, hashlib.sha256(b'SDV_FIXED_PREDICTIONS_V1\0')
    if not root.is_dir():
        raise ValueError('예측 비교용 데이터셋 경로가 없습니다: ' + str(root))
    for directory, dirs, files in os.walk(str(root), followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith('.')
                         and not (Path(directory) / name).is_symlink())
        for filename in sorted(files):
            path = Path(directory) / filename
            if path.suffix.lower() not in DATA_SUFFIXES or filename.startswith('.'):
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                raise ValueError('데이터셋 밖의 파일 링크는 예측에 사용할 수 없습니다.')
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                         stat.st_ino, stat.st_dev]
            old = cache.get(relative, {})
            digest = old.get('sha256') if old.get('signature') == signature else None
            digest = digest or _digest(path)
            inventory[relative] = {'signature': signature, 'sha256': digest}
            fingerprint.update(relative.encode('utf-8') + b'\0' + digest.encode('ascii') + b'\0')
    _atomic_json(cache_path, inventory)
    fingerprint = fingerprint.hexdigest()
    paths = image_front if image_front is not None else sorted(root.glob('*/front_cam_rgb/*'))
    indexed = {}
    for index, filename in enumerate(paths):
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            raise ValueError('RGB 표본이 데이터셋 밖에 있습니다.')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            indexed[relative] = index
    relatives = sorted(indexed)
    count = min(int(maximum), len(relatives))
    # The two endpoints and evenly spaced interior frames remain predictable.
    positions = ([0] if count == 1 else
                 [int(round(i * (len(relatives) - 1) / float(count - 1))) for i in range(count)])
    selected = [{'sample_id': hashlib.sha256(relatives[pos].encode('utf-8')).hexdigest()[:16],
                 'relative_frame': relatives[pos], 'index': indexed[relatives[pos]]}
                for pos in positions]
    manifest = {'version': 1, 'dataset_fingerprint': fingerprint,
                'total_samples': len(relatives), 'requested_samples': int(maximum),
                'samples': selected, 'count': count,
                'reason': ('데이터셋 표본이 4개 미만이므로 실제 존재하는 표본만 표시합니다.'
                           if count < maximum else ''),
                'reason_en': ('Fewer than four dataset samples exist; only actual distinct samples are shown.'
                              if count < maximum else '')}
    # Dataset-level identity is shared by all jobs. The local loader indices are
    # reconstructed above rather than trusted from an older loader order.
    _atomic_json(folder / (fingerprint + '.json'), manifest)
    return manifest


def _cpu_array(value):
    return value.detach().cpu().numpy()


def _rgb(value):
    import numpy as np
    array = _cpu_array(value).transpose(1, 2, 0)
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def _seg(value, classes=None, ground_truth=False):
    import numpy as np
    array = _cpu_array(value)
    mask = array.argmax(axis=0)
    colors = [item['color'] for item in classes] if classes is not None else CLASS_COLORS
    image = np.asarray(colors, dtype=np.uint8)[mask]
    if ground_truth:
        image[array.sum(axis=0) == 0] = (255, 0, 255)
    return image


def _mask(value, predicted, forward_up=False, valid=None):
    import numpy as np
    array = _cpu_array(value).squeeze()
    if array.ndim != 2:
        raise ValueError('BEV 예측 마스크는 2차원이어야 합니다.')
    # SDV binary heads expose logits. Its existing loss uses BCEWithLogitsLoss.
    image = ((array > 0) if predicted else (array > .5)).astype(np.uint8) * 255
    image = np.stack((image, image, image), axis=-1)
    known = np.isfinite(array)
    if valid is not None:
        known &= _cpu_array(valid).squeeze().astype(bool)
    image[~known] = (255, 0, 255)
    # Model uses rows=ego Y, columns=ego X. New labels are shown with ego
    # forward up/right right, matching the stored collection raster.
    return np.rot90(image).copy() if forward_up else image


def _depth(value, settings=None, valid=None):
    import numpy as np
    import cv2
    array = _cpu_array(value).squeeze()
    if array.ndim != 2:
        raise ValueError('깊이 예측은 2차원이어야 합니다.')
    settings = settings or {'visual_min_m': 0., 'visual_max_m': 100., 'colormap': 'plasma'}
    low, high = settings['visual_min_m'], settings['visual_max_m']
    scaled = np.rint(np.clip((np.nan_to_num(array) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    maps = {'plasma': cv2.COLORMAP_PLASMA, 'viridis': cv2.COLORMAP_VIRIDIS}
    if hasattr(cv2, 'COLORMAP_TURBO'): maps['turbo'] = cv2.COLORMAP_TURBO
    if settings['colormap'] != 'gray' and settings['colormap'] not in maps:
        raise ValueError('This OpenCV version does not support the requested depth colormap: ' + settings['colormap'])
    if settings['colormap'] == 'gray':
        image = np.stack((scaled,) * 3, axis=-1)
    else:
        image = cv2.cvtColor(cv2.applyColorMap(scaled, maps[settings['colormap']]), cv2.COLOR_BGR2RGB)
    if valid is not None:
        image[~_cpu_array(valid).squeeze().astype(bool)] = (255, 0, 255)
    return image


def _save_png(path, pixels):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    Image.fromarray(pixels, 'RGB').save(str(temporary), format='PNG')
    os.replace(str(temporary), str(path))


def _draw_boxes(pixels, boxes, labels, scores=None, classes=None):
    from PIL import Image, ImageDraw
    image = Image.fromarray(pixels.copy(), 'RGB')
    draw = ImageDraw.Draw(image)
    for index, (box, label) in enumerate(zip(boxes, labels)):
        label = int(label)
        if label < 0:
            continue
        names = [item['name'] for item in classes] if classes is not None else DETECTION_CLASSES
        name = names[label] if label < len(names) else str(label)
        if scores is not None:
            name += ' {:.2f}'.format(float(scores[index]))
        coords = tuple(float(value) for value in box)
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        color = tuple(classes[label]['color']) if classes is not None and label < len(classes) else ((96, 220, 255) if scores is not None else (255, 170, 64))
        draw.rectangle(coords, outline=color, width=2)
        draw.text((max(0, coords[0]), max(0, coords[1] - 12)), name, fill=color)
    import numpy as np
    return np.asarray(image)


def _detections(model, images, classification, regression):
    """Decode SDV RetinaNet anchors, score filter, class-aware NMS."""
    import torch
    from torchvision.ops import batched_nms
    from sdv.training.modeling.utils import BBoxTransform, ClipBoxes
    shape = images.shape
    flattened = images.reshape(-1, *shape[-3:])
    scores = classification.reshape(flattened.shape[0], -1, classification.shape[-1])
    regression = regression.reshape(flattened.shape[0], -1, 4)
    anchors = model.anchors(flattened)
    boxes = ClipBoxes()(BBoxTransform()(anchors, regression), flattened)
    result = []
    for camera_boxes, camera_scores in zip(boxes, scores):
        confidence, labels = camera_scores.max(dim=-1)
        valid = (confidence > .25) & torch.isfinite(camera_boxes).all(dim=-1)
        selected_boxes, selected_scores, selected_labels = (camera_boxes[valid], confidence[valid], labels[valid])
        # Bound visualization/NMS cost without changing the training loss.
        if selected_scores.numel() > 1000:
            top = selected_scores.topk(1000).indices
            selected_boxes, selected_scores, selected_labels = selected_boxes[top], selected_scores[top], selected_labels[top]
        keep = batched_nms(selected_boxes, selected_scores, selected_labels, .5)[:100]
        result.append((_cpu_array(selected_boxes[keep]), _cpu_array(selected_labels[keep]),
                       _cpu_array(selected_scores[keep])))
    return result


class PreviewBudgetExceeded(Exception):
    pass


class PredictionRecorder:
    def __init__(self, output_dir, project_root, val_dataset, resume_epoch=None, resume_global_step=None):
        self.output = Path(output_dir).resolve()
        self.project = Path(project_root).resolve()
        self.folder = self.output / 'predictions'
        self.index_path = self.folder / 'index.json'
        self.selections = {}
        self.budget_exhausted = False
        monitor = getattr(val_dataset, 'train_config', {}).get('monitor', {})
        self.max_bytes = int(monitor.get('preview_max_mib', 1024)) * 1024 * 1024
        self.used_bytes = sum(p.stat().st_size for p in self.folder.rglob('*') if p.is_file()) if self.folder.exists() else 0
        try:
            storage = json.loads((self.folder / 'storage_status.json').read_text())
            self.budget_exhausted = bool(storage.get('budget_exhausted') and self.max_bytes <= storage.get('max_bytes',0))
        except (OSError,ValueError,TypeError,AttributeError):pass
        self.fixed_validation = self.selection(val_dataset)
        if self.index_path.is_file():
            self.index = json.loads(self.index_path.read_text(encoding='utf-8'))
            if self.index.get('version') != 1 or not isinstance(self.index.get('entries'), list):
                raise ValueError('기존 예측 이미지 목록 형식이 잘못되었습니다.')
        else:
            self.index = {'version': 1, 'entries': [], 'datasets': {}, 'captures': 0}
        if resume_epoch is not None:
            # Retain aborted segment files, but exclude predictions beyond the
            # restored checkpoint from the active optimization timeline.
            for entry in self.index['entries']:
                if (entry.get('epoch') or 0) > resume_epoch or (resume_global_step is not None and (entry.get('global_step') or 0) > resume_global_step):
                    entry['superseded'] = True
                    entry['superseded_reason'] = 'checkpoint_rewind'
            if self.index_path.exists():_atomic_json(self.index_path,self.index)


    def selection(self, dataset):
        key = str(Path(dataset.data_root).resolve())
        if key not in self.selections:
            self.selections[key] = fixed_samples(dataset.data_root, self.project, dataset.image_front)
        return self.selections[key]

    def capture(self, model, dataset, collater, split='val', epoch=1, global_step=0, indices=None, epoch_step=None, steps_per_epoch=None):
        import numpy as np
        import torch
        if split not in ('train', 'val'):
            raise ValueError('예측 split은 train 또는 val이어야 합니다.')
        if self.budget_exhausted:
            return {'count':0,'split':split,'epoch':int(epoch),'global_step':int(global_step),'budget_exhausted':True,
                    'used_bytes':self.used_bytes,'max_bytes':self.max_bytes,'reason':'Prediction PNG 용량 한도에 도달했습니다. 기존 기록은 보존하며 학습을 계속합니다.'}
        selection = self.selection(dataset)
        from sdv.data_contract import normalize_task_settings, segmentation_spec, detection_spec
        settings = normalize_task_settings(getattr(dataset, 'train_config', {}).get('task_settings'))
        seg_classes, od_classes = segmentation_spec(settings), detection_spec(settings)
        schema = {'segmentation': seg_classes, 'detection': od_classes, 'task_settings': settings, 'ignore_color': [255, 0, 255]}
        samples = selection['samples']
        if indices is not None:
            allowed = set(int(index) for index in indices)
            samples = [sample for sample in samples if sample['index'] in allowed]
        if not samples:
            return {'count': 0, 'reason': selection['reason'], 'split': split, 'samples': []}
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        modes = [(module, module.training) for module in model.modules()]
        py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        capture_id = int(self.index.get('captures', 0)) + 1
        relative_base = 'predictions/{}/epoch_{:04d}/step_{:08d}/capture_{:06d}'.format(
            split, int(epoch), int(global_step), capture_id)
        entries = []
        written_bytes = 0
        def save_png(path, array):
            # Encode before touching disk. A failed capture removes only its own
            # newly created directory; every older capture remains intact.
            import io
            from PIL import Image
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format='PNG')
            payload = stream.getvalue()
            nonlocal written_bytes
            if self.used_bytes + written_bytes + len(payload) > self.max_bytes:
                raise PreviewBudgetExceeded()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            written_bytes += len(payload)
        try:
            # Future random augmentations cannot make cross-model probes drift.
            seed = int(selection['dataset_fingerprint'][:8], 16)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model.eval()
            with torch.no_grad():
                # One frame at a time keeps 4 probes usable with BEV/LiDAR even
                # when the training batch size is already at the GPU limit.
                for sample in samples:
                    sample_seed = int(hashlib.sha256((selection['dataset_fingerprint'] + sample['sample_id']).encode()).hexdigest()[:8], 16)
                    random.seed(sample_seed); np.random.seed(sample_seed); torch.manual_seed(sample_seed)
                    batch = collater([dataset[sample['index']]])
                    batch = [value.to(device) if torch.is_tensor(value) else value for value in batch]
                    imgs, imgs_ori = batch[:2]
                    outputs = model(imgs, batch[13], batch[12], batch[11], batch[9], batch[8], batch[6], batch[7])
                    if len(outputs) != 8:
                        raise ValueError('인지 모델 예측은 SDV 8개 태스크 출력 계약을 따라야 합니다.')
                    camera_keys = getattr(dataset, 'sensor_keys', ['front_cam'])
                    front = camera_keys.index('front_cam') if 'front_cam' in camera_keys else 0
                    sample_base = relative_base + '/sample_' + sample['sample_id']
                    input_paths = {}
                    for camera, key in enumerate(camera_keys):
                        path = sample_base + '/' + key + '_input.png'
                        save_png(self.output / path, _rgb(imgs_ori[0, camera]))
                        input_paths[key] = path
                    def add(task, camera, prediction, truth, units=None):
                        suffix = (camera + '_') if camera else ''
                        pred_path, gt_path = (sample_base + '/' + suffix + task.lower() + '_' + kind + '.png'
                                              for kind in ('pred', 'gt'))
                        save_png(self.output / pred_path, prediction)
                        save_png(self.output / gt_path, truth)
                        entry = {'id': '{}:{}:{}:{}'.format(capture_id, sample['sample_id'], camera or 'bev', task),
                                 'capture_id': capture_id, 'split': split, 'epoch': int(epoch),
                                 'global_step': int(global_step), 'epoch_step': epoch_step, 'steps_per_epoch': steps_per_epoch, 'sample_id': sample['sample_id'],
                                 'relative_frame': sample['relative_frame'], 'dataset_fingerprint': selection['dataset_fingerprint'],
                                 'task': task, 'camera': camera, 'input': input_paths[camera or camera_keys[front]],
                                 'gt': gt_path, 'pred': pred_path, 'created_at': time.time(), 'source': 'actual_model_prediction', 'label_schema': schema}
                        entry['images'] = {'input': entry['input'], 'gt': gt_path, 'pred': pred_path}
                        if units: entry['display_scale'] = units
                        entries.append(entry)
                    for offset, name in enumerate(('DAS', 'Lane', 'Vehicle', 'Walker')):
                        if name.lower() in settings['bev']['layers'] and outputs[offset] is not None and batch[offset + 2] is not None:
                            forward_up = getattr(dataset, 'bev_schema', None) in (3,4)
                            truth = batch[offset + 2][0]
                            add(name, None, _mask(outputs[offset][0], True, forward_up, torch.isfinite(truth)),
                                _mask(truth, False, forward_up))
                            if forward_up:
                                entries[-1]['bev_ground_truth'] = {
                                    'schema_version': dataset.bev_schema, 'source': 'provided_global_raster_and_simulator_actors' if dataset.bev_schema == 4 else 'simulator_ground_truth',
                                    'model_grid': {'x': [-20., 20., .2], 'y': [-20., 20., .2]},
                                    'tensor_axes': 'row=ego Y+, column=ego X+',
                                    'image_axes': 'forward-up/right-right',
                                    'unknown_color': [255, 0, 255],
                                    'validity': ('occupancy_validity' if name in ('Vehicle', 'Walker') else
                                                 'lane_validity & validity' if name == 'Lane' else 'validity'),
                                    'height_auxiliary': 'GT only; not model input'}
                    detection = (_detections(model, imgs, outputs[6], outputs[7])
                                 if outputs[6] is not None and outputs[7] is not None else None)
                    for camera, key in enumerate(camera_keys):
                        if outputs[4] is not None and batch[14] is not None:
                            add('Seg', key, _seg(outputs[4][0, camera], seg_classes), _seg(batch[14][0, camera], seg_classes, ground_truth=True))
                        if outputs[5] is not None and batch[15] is not None:
                            add('Depth', key, _depth(outputs[5][0, camera], settings['depth']),
                                _depth(batch[15][0, camera], settings['depth'], batch[16][0, camera]),
                                {'min': settings['depth']['visual_min_m'], 'max': settings['depth']['visual_max_m'], 'unit': 'm', 'colormap': settings['depth']['colormap']})
                        if detection is not None:
                            boxes, labels, scores = detection[camera]
                            annotation = _cpu_array(batch[17][0, camera])
                            add('od', key, _draw_boxes(_rgb(imgs_ori[0, camera]), boxes, labels, scores, od_classes),
                                _draw_boxes(_rgb(imgs_ori[0, camera]), annotation[:, :4], annotation[:, 4], classes=od_classes))
        except PreviewBudgetExceeded:
            self.budget_exhausted = True
            import shutil
            shutil.rmtree(self.output / relative_base, ignore_errors=True)
            _atomic_json(self.folder / 'storage_status.json', {'used_bytes':self.used_bytes,'max_bytes':self.max_bytes,'budget_exhausted':True})
            return {'count': 0, 'split': split, 'epoch': int(epoch), 'global_step': int(global_step),
                    'reason': 'Prediction 저장 용량 한도에 도달했습니다. 기존 기록은 보존합니다.',
                    'budget_exhausted': True, 'used_bytes': self.used_bytes, 'max_bytes': self.max_bytes}
        finally:
            # Preserve frozen/eval submodules as well as the parent train mode.
            for module, mode in modes: module.training = mode
            random.setstate(py_rng); np.random.set_state(np_rng); torch.set_rng_state(cpu_rng)
            if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        # Publish metadata only after every image is complete. No partial capture
        # is advertised by the API; existing historical records stay untouched.
        self.used_bytes += written_bytes
        self.index['storage_budget'] = {'used_bytes': self.used_bytes, 'max_bytes': self.max_bytes, 'budget_exhausted':False}
        _atomic_json(self.folder / 'storage_status.json', self.index['storage_budget'])
        self.index['entries'].extend(entries)
        self.index['captures'] = capture_id
        self.index['datasets'][split] = selection
        self.index['label_schema'] = schema
        _atomic_json(self.index_path, self.index)
        return {'count': len(entries), 'capture_id': capture_id, 'split': split,
                'epoch': int(epoch), 'global_step': int(global_step), 'epoch_step': epoch_step, 'steps_per_epoch': steps_per_epoch,
                'dataset_fingerprint': selection['dataset_fingerprint'],
                'samples': samples, 'reason': selection['reason'], 'reason_en': selection['reason_en']}
