"""Native SDV label contracts shared by collection, web inspection and training.

The palette is the collector's CARLA 0.9.15 palette (RGB, not OpenCV BGR).
Optional image dependencies are imported only in decoding functions.
"""
from __future__ import annotations
import copy
import math

SEMANTIC_NAMES = ('Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
                  'Pole', 'Traffic Light', 'Traffic Sign', 'Vegetation', 'Terrain',
                  'Sky', 'Pedestrian', 'Rider', 'Car', 'Truck', 'Bus', 'Train',
                  'Motorcycle', 'Bicycle', 'Static', 'Dynamic', 'Other', 'Water',
                  'Road Line', 'Ground', 'Bridge', 'Rail Track', 'Guard Rail')
SEMANTIC_COLORS = ((0,0,0),(128,64,128),(244,35,232),(70,70,70),(102,102,156),
                   (190,153,153),(153,153,153),(250,170,30),(220,220,0),(107,142,35),
                   (152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),
                   (0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32),
                   (110,190,160),(170,120,50),(55,90,80),(45,60,150),(157,234,50),
                   (81,0,81),(150,100,100),(230,150,140),(180,165,180))
OD_CLASSES = {0: 'Vehicle', 1: 'Pedestrian'}
OD_COLORS = {0: (59,130,246), 1: (251,146,60)}
BEV_LAYERS = {
    'das': {'name': 'Drivable Area', 'folder': 'das', 'kind': 'semantic', 'trainable': True},
    'lane': {'name': 'Lane', 'folder': 'lane', 'kind': 'map', 'trainable': True},
    'vehicle': {'name': 'Vehicle', 'folder': 'vehicle_mask', 'kind': 'semantic', 'trainable': True},
    'walker': {'name': 'Pedestrian', 'folder': 'walker_mask', 'kind': 'semantic', 'trainable': True},
    'traffic_light': {'name': 'Traffic Light', 'folder': 'light_topview', 'kind': 'map', 'trainable': False},
    'stopline': {'name': 'Stop Line', 'folder': 'stopline', 'kind': 'map', 'trainable': False},
    'route': {'name': 'Route', 'folder': 'route', 'kind': 'navigation', 'trainable': False},
}
# Preserve old checkpoint channel order only when explicitly using legacy mode.
LEGACY_SEG_NAMES = ('Pedestrians', 'RoadLines', 'Roads', 'Sidewalks', 'Vehicles', 'Other')
LEGACY_SEG_SOURCE_COLORS = ((102,102,156),(153,153,153),(250,170,30),(220,220,0),(152,251,152),(110,190,160))
LEGACY_SEG_COLORS = ((0,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,0,0))
IGNORE_INDEX = 255


def semantic_records(ids=None):
    values = range(len(SEMANTIC_NAMES)) if ids is None else ids
    return [{'id': int(i), 'name': SEMANTIC_NAMES[int(i)], 'color': list(SEMANTIC_COLORS[int(i)])}
            for i in values if 0 <= int(i) < len(SEMANTIC_NAMES)]


def task_settings_defaults():
    """Compatibility defaults; the web selects discovered classes explicitly."""
    return {
        'segmentation': {'mode': 'legacy', 'class_ids': [], 'excluded': 'ignore'},
        'depth': {'encoding': 'auto', 'min_m': .001, 'max_m': 100., 'out_of_range': 'mask',
                  'invalid': 'mask', 'visual_min_m': 0., 'visual_max_m': 100., 'colormap': 'plasma'},
        'detection': {'class_ids': [0, 1]},
        'bev': {'layers': ['das', 'lane', 'vehicle', 'walker'], 'visual_layers': list(BEV_LAYERS)},
    }


def _number(value, key, low=0., high=1000.):
    if isinstance(value, bool): raise ValueError(key + ': a finite number is required')
    try: value = float(value)
    except (TypeError, ValueError): raise ValueError(key + ': a finite number is required')
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(key + ': expected range %s..%s' % (low, high))
    return value


def _selection(value, allowed, key, empty=False):
    if not isinstance(value, list) or any(type(x) not in (str, int) or x not in allowed for x in value):
        raise ValueError(key + ': unsupported selection')
    if not empty and not value: raise ValueError(key + ': select at least one class/layer')
    if len(set(value)) != len(value): raise ValueError(key + ': duplicate selection')
    return list(value)


def normalize_task_settings(value=None):
    defaults = task_settings_defaults()
    if value is None: return defaults
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError('task_settings: unsupported task')
    result = copy.deepcopy(defaults)
    for task, fields in value.items():
        if not isinstance(fields, dict) or set(fields) - set(defaults[task]):
            raise ValueError(task + ': unsupported advanced setting')
        result[task].update(fields)
    seg = result['segmentation']
    if seg['mode'] not in ('legacy', 'carla'): raise ValueError('segmentation.mode: legacy or carla required')
    if seg['excluded'] not in ('background', 'ignore'): raise ValueError('segmentation.excluded: background or ignore required')
    seg['class_ids'] = _selection(seg['class_ids'], range(29), 'segmentation.class_ids', empty=True)
    # Background is a real output channel and always index zero when selected.
    seg['class_ids'] = sorted(seg['class_ids'])
    depth = result['depth']
    if depth['encoding'] not in ('auto', 'legacy_100m_u8', 'meters_u8', 'meters_f32'):
        raise ValueError('depth.encoding: unknown source encoding')
    for field in ('min_m', 'max_m', 'visual_min_m', 'visual_max_m'):
        depth[field] = _number(depth[field], 'depth.' + field)
    if depth['min_m'] >= depth['max_m'] or depth['visual_min_m'] >= depth['visual_max_m']:
        raise ValueError('Depth minimum must be below maximum')
    if depth['out_of_range'] not in ('mask', 'clip'): raise ValueError('depth.out_of_range: mask or clip required')
    if depth['invalid'] != 'mask': raise ValueError('Invalid depth must remain masked')
    if depth['colormap'] not in ('plasma', 'viridis', 'turbo', 'gray'):
        raise ValueError('Unsupported depth colormap')
    result['detection']['class_ids'] = sorted(_selection(result['detection']['class_ids'], OD_CLASSES, 'detection.class_ids', empty=True))
    bev = result['bev']
    bev['layers'] = _selection(bev['layers'], ('das','lane','vehicle','walker'), 'bev.layers', empty=True)
    bev['visual_layers'] = _selection(bev['visual_layers'], BEV_LAYERS, 'bev.visual_layers', empty=True)
    return result


def segmentation_spec(settings=None):
    settings = normalize_task_settings(settings)
    seg = settings['segmentation']
    if seg['mode'] == 'legacy':
        return [{'output_index': i, 'id': None, 'name': name, 'color': list(LEGACY_SEG_COLORS[i])}
                for i, name in enumerate(LEGACY_SEG_NAMES)]
    ids = list(seg['class_ids'])
    if seg['excluded'] == 'background' and 0 not in ids: ids.insert(0, 0)
    values = semantic_records(ids)
    for index, item in enumerate(values):
        item['output_index'] = index
        if item['id'] == 0 and seg['excluded'] == 'background': item['name'] = 'Background / excluded'
    return values


def decode_segmentation(image):
    """Decode RGB palette or integer mask; unknown pixels remain ignore 255.

    No nearest-color heuristic: interpolated/compressed colors are invalid labels.
    Raw CARLA BGRA must first provide its red channel (array[..., 2]).
    """
    import numpy as np
    array = np.asarray(image)
    if array.ndim == 2:
        valid = np.isfinite(array) & (array >= 0) & (array < len(SEMANTIC_NAMES)) & (array == np.floor(array))
        out = np.full(array.shape, IGNORE_INDEX, dtype=np.uint8)
        out[valid] = array[valid].astype(np.uint8)
        return out
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError('Segmentation requires RGB palette or integer class IDs')
    rgb = array[..., :3].astype(np.uint32)
    packed = rgb[...,0] * 65536 + rgb[...,1] * 256 + rgb[...,2]
    out = np.full(array.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for class_id, color in enumerate(SEMANTIC_COLORS):
        out[packed == color[0]*65536+color[1]*256+color[2]] = class_id
    return out


def remap_segmentation(ids, settings=None):
    import numpy as np
    settings = normalize_task_settings(settings)
    seg = settings['segmentation']
    if seg['mode'] != 'carla':
        raise ValueError('Raw semantic IDs require segmentation.mode=carla')
    ids = np.asarray(ids)
    result = np.full(ids.shape, 0 if seg['excluded']=='background' else IGNORE_INDEX, dtype=np.int64)
    # Unknown/invalid labels must not silently become background.
    result[ids == IGNORE_INDEX] = IGNORE_INDEX
    for item in segmentation_spec(settings): result[ids == item['id']] = item['output_index']
    return result


def colorize_segmentation(ids):
    import numpy as np
    ids = np.asarray(ids)
    result = np.full(ids.shape + (3,), (255,0,255), dtype=np.uint8)
    for class_id, color in enumerate(SEMANTIC_COLORS): result[ids == class_id] = color
    return result


def infer_depth_encoding(scene, path=None):
    """Recognize collector provenance, never guess a unit from pixel magnitudes."""
    import json
    from pathlib import Path
    scene = Path(scene)
    if path is not None and Path(path).suffix.lower() == '.npy': return 'meters_f32'
    try:
        manifest = json.loads((scene/'collection_manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError): manifest = {}
    enc = manifest.get('depth_encoding')
    if enc in ('meters_f32', 'meters_u8', 'legacy_100m_u8'): return enc
    if manifest.get('format') == 'sdv-native-masks-v1': return 'meters_u8'
    return 'legacy_100m_u8'


def decode_depth(image, encoding='legacy_100m_u8'):
    """Return meters and mask; old collector PNG 255 is saturated and unrecoverable."""
    import numpy as np
    array = np.asarray(image)
    if array.ndim != 2: raise ValueError('Depth requires a single-channel array')
    value = array.astype(np.float32)
    if encoding == 'legacy_100m_u8': value = value * (100. / 255.)
    elif encoding not in ('meters_u8', 'meters_f32'): raise ValueError('Resolve depth encoding before decoding')
    valid = np.isfinite(value) & (value > 0)
    if encoding == 'meters_u8': valid &= array < 255
    return value, valid


def prepare_depth(image, settings=None, encoding=None):
    import numpy as np
    depth = normalize_task_settings(settings)['depth']
    encoding = encoding or depth['encoding']
    if encoding == 'auto': raise ValueError('Auto depth encoding requires dataset provenance')
    values, valid = decode_depth(image, encoding)
    inside = (values >= depth['min_m']) & (values <= depth['max_m'])
    if depth['out_of_range'] == 'mask': valid &= inside
    else: values = np.where(valid, np.clip(values, depth['min_m'], depth['max_m']), values)
    return values, valid


def depth_encoding_for_scene(scene, requested='auto', path=None):
    return infer_depth_encoding(scene, path) if requested == 'auto' else requested


def detection_spec(settings=None):
    selected = normalize_task_settings(settings)['detection']['class_ids']
    return [{'output_index': i, 'id': key, 'name': OD_CLASSES[key], 'color': list(OD_COLORS[key])}
            for i, key in enumerate(selected)]


def new_task_settings(observed_seg_ids=None, observed_od_ids=None):
    """New jobs opt into corrected semantics; selections come from real labels."""
    settings = task_settings_defaults()
    settings['segmentation'].update(mode='carla', class_ids=sorted(set(observed_seg_ids or [])))
    settings['detection']['class_ids'] = sorted(set(observed_od_ids or []))
    return normalize_task_settings(settings)
