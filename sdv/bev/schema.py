"""Numeric BEV contract. Only visualization uses the RGB palette."""
import copy
import math

SCHEMA_VERSION = 3
GLOBAL_RASTER_SCHEMA_VERSION = 4
SEMANTIC_CLASSES = [
    {'id': 0, 'name': 'Unknown / background', 'color': [18, 25, 39]},
    {'id': 1, 'name': 'Drivable area', 'color': [70, 88, 112]},
    {'id': 2, 'name': 'Sidewalk', 'color': [105, 78, 111]},
    {'id': 3, 'name': 'Junction', 'color': [71, 106, 120]},
    {'id': 4, 'name': 'Road boundary', 'color': [160, 175, 198]},
    {'id': 5, 'name': 'Lane marking', 'color': [224, 235, 249]},
    {'id': 6, 'name': 'Route', 'color': [42, 200, 235]},
    {'id': 7, 'name': 'Green stop line', 'color': [52, 211, 153]},
    {'id': 8, 'name': 'Yellow stop line', 'color': [250, 204, 21]},
    {'id': 9, 'name': 'Red stop line', 'color': [248, 75, 97]},
    {'id': 10, 'name': 'Vehicle', 'color': [70, 137, 250]},
    {'id': 11, 'name': 'Pedestrian', 'color': [251, 146, 60]},
    {'id': 12, 'name': 'Cyclist', 'color': [195, 132, 252]},
    {'id': 13, 'name': 'Ego vehicle', 'color': [97, 231, 216]},
]
SEMANTIC_ORDER = [('drivable',1), ('sidewalk',2), ('junction',3),
    ('road_boundary',4), ('lane',5), ('route',6), ('stop_green',7),
    ('stop_yellow',8), ('stop_red',9), ('vehicle',10), ('pedestrian',11),
    ('cyclist',12), ('ego',13)]
STATIC_CHANNELS = ('road', 'drivable', 'lane', 'road_boundary', 'sidewalk', 'junction')
BINARY_CHANNELS = STATIC_CHANNELS + ('vehicle', 'pedestrian', 'cyclist', 'ego',
    'stop_red', 'stop_yellow', 'stop_green', 'route', 'validity', 'occupancy_validity', 'lane_validity')
HEIGHT_CHANNELS = ('elevation_m', 'object_ground_z_m', 'object_min_z_m',
                  'object_max_z_m', 'object_height_m')

def bev_settings_defaults():
    return {'mode':'global_raster', 'width':500, 'height':500, 'meters_per_pixel':0.2,
        'anchor_x':0.5, 'anchor_y':0.5, 'height_aware':True,
        'layer_policy':'hblr_ground_height', 'object_height':True,
        'map_root':'assets/bev_maps', 'global_ppm':5, 'rotate_with_ego':True,
        'actor_classes':['vehicle','pedestrian','cyclist'],
        'semantic_layers':['drivable','lane','road_boundary','sidewalk','junction'],
        'rasterization':'polygon',
        'visual_overlays':['drivable','lane','road_boundary','sidewalk','junction',
            'vehicle','pedestrian','cyclist','ego','route','stoplines']}

def normalize_bev_settings(value=None):
    result = bev_settings_defaults()
    if value is not None:
        if not isinstance(value, dict) or set(value)-set(result):
            raise ValueError('bev_generator: unsupported setting')
        result.update(copy.deepcopy(value))
        if 'layer_policy' not in value and result['mode']=='height_aware':
            result['layer_policy']='topology_surface'
    for key, lo, hi in [('width',64,1024),('height',64,1024),
                        ('meters_per_pixel',0.05,1.0),('anchor_x',0,1),('anchor_y',0,1)]:
        v = result[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError('bev_generator.{}: expected {}..{}'.format(key,lo,hi))
        if key in ('width','height'):
            if int(v) != v: raise ValueError('BEV dimensions must be integers')
            result[key] = int(v)
    if result['width']*result['meters_per_pixel'] > 200 or result['height']*result['meters_per_pixel'] > 200:
        raise ValueError('BEV width/height must not exceed 200 metres')
    if result['mode'] not in ('global_raster','height_aware','legacy'): raise ValueError('Unsupported BEV mode')
    if result['layer_policy'] not in ('topology_surface','hblr_ground_height') or result['rasterization'] != 'polygon':
        raise ValueError('Unsupported BEV layer policy or rasterization')
    if result['mode']=='global_raster' and result['layer_policy']!='hblr_ground_height':
        raise ValueError('Global map images require the HBLR ground-height layer policy')
    if type(result['global_ppm']) is not int or result['global_ppm'] not in (5,10):
        raise ValueError('Global map PPM must be an available native resolution: 5 or 10')
    if not isinstance(result['map_root'],str) or not result['map_root'].strip() or '\x00' in result['map_root'] or len(result['map_root'])>4096:
        raise ValueError('Global map root must be a non-empty filesystem path')
    for key in ('height_aware','object_height','rotate_with_ego'):
        if type(result[key]) is not bool: raise ValueError('bev_generator.'+key+': expected boolean')
    for key in ('actor_classes','semantic_layers','visual_overlays'):
        allowed = bev_settings_defaults()[key]
        values = result[key]
        if not isinstance(values,list) or any(not isinstance(x,str) or x not in allowed for x in values) or len(set(values)) != len(values):
            raise ValueError('bev_generator.'+key+': unsupported or duplicate selection')
    return result

def channel_schema():
    fields = {k:{'dtype':'uint8', 'range':[0,1]} for k in BINARY_CHANNELS}
    fields.update({k:{'dtype':'float32','unit':'m','invalid':'NaN'} for k in HEIGHT_CHANNELS})
    fields.update({'semantic':{'dtype':'uint8','range':[0,13]},
                   'actor_id':{'dtype':'int32','background':0,'overlap':'topmost bbox; all instances in JSON'},
                   'surface_id':{'dtype':'int32','unknown':-1,'lookup':'surface_table'}})
    return fields
