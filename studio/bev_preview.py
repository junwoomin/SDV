"""Views of recorded numeric BEV ground truth; never infer missing heights.

This module has no CARLA/OpenCV/torch dependency. Palette colors are display
only; every mask and height value comes from the selected frame's NPZ file.
"""
from __future__ import annotations


import numpy as np
from PIL import Image

from sdv.bev.schema import SEMANTIC_CLASSES, SEMANTIC_ORDER

MASK_COLORS = {'road': '#263548'}
for key, cid in SEMANTIC_ORDER:
    if key == 'stop_green': MASK_COLORS['static_stopline'] = '#f0f4fa'
    MASK_COLORS[key] = '#' + ''.join('{:02x}'.format(v) for v in SEMANTIC_CLASSES[cid]['color'])
HEIGHT_FIELDS = ('elevation_m', 'object_ground_z_m', 'object_min_z_m',
                 'object_max_z_m', 'object_height_m', 'min_z_m', 'max_z_m', 'height_m')
CHANNELS = ('bev', 'bev_height', 'bev_object_height', 'bev_occupancy', 'bev_road', 'bev_route')
ALIASES = {'das': 'drivable', 'walker': 'pedestrian'}


def _metadata(value):
    if value is not None and not isinstance(value, dict):
        raise ValueError('bev_gt metadata must be an object')
    value = value or {}
    if 'settings' in value and not isinstance(value['settings'], dict):
        raise ValueError('BEV capture settings must be an object')
    if 'actors' in value and (not isinstance(value['actors'], list) or
            any(not isinstance(actor, dict) for actor in value['actors'])):
        raise ValueError('BEV actors must be a list of objects')
    if 'warnings' in value and (not isinstance(value['warnings'], list) or
            any(not isinstance(warning, str) for warning in value['warnings'])):
        raise ValueError('BEV warnings must be a list of strings')
    geometry = value.get('geometry')
    if geometry is not None and (not isinstance(geometry, dict) or
            any(type(geometry.get(key)) is not int for key in ('width', 'height'))):
        raise ValueError('BEV geometry requires integer width and height')
    return value


def read_bev(path, metadata=None):
    """Use exactly the same numeric validation as the training loader."""
    from sdv.bev.io import load_frame
    metadata = _metadata(metadata)
    return load_frame(path, metadata.get('geometry'), metadata=metadata)


def layer_catalog(arrays, metadata=None):
    settings = (metadata or {}).get('settings') or {}
    if not isinstance(settings, dict): settings = {}
    semantic = settings.get('semantic_layers')
    actors = settings.get('actor_classes')
    source = (metadata or {}).get('global_map') or {}
    files = source.get('files') if isinstance(source, dict) else None
    def enabled(key):
        if isinstance(files, dict) and key in ('road','drivable','lane','road_boundary','sidewalk','junction'):
            source_key = 'drivable' if key == 'road' else key
            if source_key not in files: return False
        if key in ('road', 'drivable', 'lane', 'road_boundary', 'sidewalk', 'junction') and isinstance(semantic, list):
            return ('drivable' if key == 'road' else key) in semantic
        if key in ('vehicle', 'pedestrian', 'cyclist') and isinstance(actors, list): return key in actors
        return True
    return [dict(id=key, label=('Source stop line · state unknown' if key == 'static_stopline' else key.replace('_', ' ').title()), name=key,
                 color=color, available=key in arrays and enabled(key),
                 kind=('provided_map_annotation' if isinstance(files, dict) and key in ('road','drivable','lane','road_boundary','sidewalk','junction','static_stopline') else 'simulator_ground_truth'))
            for key, color in MASK_COLORS.items()]


def describe(arrays, metadata=None):
    metadata = _metadata(metadata)
    settings = metadata.get('settings') or {}
    shape = next(iter(arrays.values())).shape
    heights = {}
    for key in HEIGHT_FIELDS:
        if key not in arrays: continue
        values = arrays[key]
        valid = np.isfinite(values)
        if key == 'elevation_m' and 'validity' in arrays: valid &= arrays['validity'] > 0
        finite = values[valid]
        heights[key] = {'minimum_m': float(finite.min()) if finite.size else None,
                        'maximum_m': float(finite.max()) if finite.size else None,
                        'valid_pixels': int(finite.size)}
    return {'available': True, 'schema_version': metadata.get('schema_version'),
            'width': shape[1], 'height': shape[0], 'channels': sorted(arrays),
            'height_ranges': heights, 'geometry': metadata.get('geometry'),
            'visual_overlays': settings.get('visual_overlays'),
            'global_map': metadata.get('global_map'),
            'views': available_views(arrays), 'layers': layer_catalog(arrays, metadata),
            'provenance': metadata.get('provenance', metadata.get('label_source'))}


def available_views(arrays):
    return [dict(id=channel, label=label, available=any(key in arrays and
                (bool(np.isfinite(arrays[key]).any()) if key in HEIGHT_FIELDS else True) for key in keys))
            for channel, label, keys in (
                ('bev', 'Semantic BEV', ('semantic', 'drivable')),
                ('bev_height', 'Road elevation (m)', ('elevation_m',)),
                ('bev_object_height', 'Object height (m)', ('object_height_m', 'height_m')),
                ('bev_occupancy', 'Object occupancy', ('vehicle', 'pedestrian', 'ego', 'cyclist')),
                ('bev_road', 'Road / lane', ('drivable', 'lane', 'road_boundary')),
                ('bev_route', 'Route', ('route',))) ]


def _rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def render_bev(arrays, channel='bev', layers=None, semantic_palette=None):
    if channel not in CHANNELS: raise ValueError('Unknown BEV view')
    if not next(v['available'] for v in available_views(arrays) if v['id'] == channel):
        raise FileNotFoundError('This frame has no numeric data for the selected BEV view')
    shape = next(iter(arrays.values())).shape
    output = np.empty(shape + (3,), dtype=np.uint8)
    output[:] = (10, 18, 31)
    if channel in ('bev_height', 'bev_object_height'):
        key = 'elevation_m' if channel == 'bev_height' else ('object_height_m' if 'object_height_m' in arrays else 'height_m')
        values = arrays[key]
        valid = np.isfinite(values)
        if key == 'elevation_m' and 'validity' in arrays: valid &= arrays['validity'] > 0
        if valid.any():
            low, high = float(values[valid].min()), float(values[valid].max())
            scaled = np.zeros(shape, dtype=np.float32)
            scaled[valid] = (values[valid] - low) / (high - low) if high > low else .5
            # A deterministic cyan-to-amber legend; it is never saved as a label.
            start, end = np.array([26, 88, 130]), np.array([251, 191, 89])
            output[valid] = (start + scaled[valid, None] * (end - start)).astype(np.uint8)
        return Image.fromarray(output)
    allowed = set(MASK_COLORS)
    if channel == 'bev_occupancy': allowed &= {'vehicle', 'pedestrian', 'cyclist', 'ego'}
    elif channel == 'bev_road': allowed &= {'drivable', 'lane', 'road_boundary', 'sidewalk', 'junction'}
    elif channel == 'bev_route': allowed &= {'route'}
    if layers is not None:
        selected = []
        for key in layers:
            selected.extend(('static_stopline','stop_red','stop_yellow','stop_green') if key in ('stopline','stoplines','traffic_light') else [ALIASES.get(key,key)])
        if any(k not in MASK_COLORS for k in selected): raise ValueError('Unknown numeric BEV layer')
        allowed &= set(selected)
    # Exact semantic class IDs are interpreted only with the generator's palette.
    # Selecting independent mask overlays intentionally shows overlaps instead.
    if channel == 'bev' and layers is None and 'semantic' in arrays and semantic_palette:
        for class_id, color in semantic_palette.items():
            output[arrays['semantic'] == int(class_id)] = color
    else:
        for key, color in MASK_COLORS.items():
            if key in allowed and key in arrays: output[arrays[key] > 0] = _rgb(color)
    return Image.fromarray(output)
