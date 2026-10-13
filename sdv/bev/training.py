"""Schema 3/4 ground-truth adapters for SDV's existing four binary BEV heads.

Schema 3 rasters are forward-up/right-right; schema 4 uses its recorded affine
and supports unrotated world rasters. The unmodified SDV voxel grid is indexed
[ego Y, ego X], both increasing. Resampling is in metres at voxel centres;
it is not an arbitrary image resize. Height is auxiliary ground truth, never
a camera/LiDAR input or an automatically added model channel.
"""
import json
import zipfile
from pathlib import Path

import numpy as np

from .io import load_frame
from .schema import HEIGHT_CHANNELS
from .geometry import BEVGeometry, world_to_local


MODEL_GRID = {'x': [-20.0, 20.0, 0.2], 'y': [-20.0, 20.0, 0.2]}
HEAD_CHANNELS = {'das': 'drivable', 'lane': 'lane', 'vehicle': 'vehicle',
                 'walker': 'pedestrian'}


def read_bev_metadata(scene, stem, required=False):
    """Read aligned JSON only; a declared numeric frame cannot fall back to PNG."""
    path = Path(scene) / 'info' / (str(stem) + '.json')
    if not path.is_file():
        if required:
            raise ValueError('Missing aligned BEV metadata: ' + str(path))
        return None
    try:
        with path.open(encoding='utf-8') as stream:
            value = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError('Unreadable BEV metadata {}: {}'.format(path, error))
    if not isinstance(value, dict):
        raise ValueError('BEV frame info must be an object: ' + str(path))
    metadata = value.get('bev_gt')
    if metadata is None and required:
        raise ValueError('Missing bev_gt schema declaration: ' + str(path))
    if metadata is not None:
        if not isinstance(metadata, dict) or metadata.get('schema_version') not in (3,4):
            raise ValueError('Unsupported BEV ground-truth schema: ' + str(path))
    return metadata


def frame_bev_schema(scene, stem):
    path = Path(scene) / 'bev_gt' / (str(stem) + '.npz')
    metadata = read_bev_metadata(scene, stem, required=path.is_file())
    if metadata is not None and not path.is_file():
        raise ValueError('Missing declared numeric BEV archive: ' + str(path))
    return metadata['schema_version'] if path.is_file() else None


def frame_has_numeric_bev(scene, stem):
    return frame_bev_schema(scene, stem) is not None


def _geometry(metadata):
    geometry = metadata.get('geometry')
    if not isinstance(geometry, dict):
        raise ValueError('Numeric BEV requires geometry metadata')
    for name in ('width', 'height'):
        if type(geometry.get(name)) is not int or not 1 <= geometry[name] <= 16384:
            raise ValueError('Invalid BEV geometry ' + name)
    for name in ('meters_per_pixel', 'anchor_x', 'anchor_y'):
        value = geometry.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Invalid BEV geometry ' + name)
    if geometry['meters_per_pixel'] <= 0:
        raise ValueError('BEV meters_per_pixel must be positive')
    if not (0 <= geometry['anchor_x'] <= 1 and 0 <= geometry['anchor_y'] <= 1):
        raise ValueError('BEV anchor coordinates must be normalized fractions')
    return geometry


def load_bev_ground_truth(scene, stem):
    """Strict pickle-free numeric loading; colors never become training labels."""
    metadata = read_bev_metadata(scene, stem, required=True)
    geometry = _geometry(metadata)
    if metadata['schema_version'] == 4:
        validate_raster_affine(metadata)
    path = Path(scene) / 'bev_gt' / (str(stem) + '.npz')
    try:
        arrays = load_frame(path, geometry, metadata=metadata)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError('Invalid BEV archive {}: {}'.format(path, error))
    return arrays, metadata


def validate_raster_affine(metadata):
    """Schema-4 crop coordinates are explicit, including rotation OFF frames."""
    geometry = _geometry(metadata)
    try:
        matrix = np.asarray(geometry['world_to_bev_pixel'], dtype=np.float64)
        pose = metadata['ego_pose']
        xy = np.asarray([pose['x'],pose['y']], dtype=np.float64)
        yaw = float(pose['yaw'])
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('Global raster BEV requires world_to_bev_pixel and actual ego pose: '+str(error))
    if matrix.shape != (2,3) or not np.isfinite(matrix).all() or not np.isfinite(xy).all() or not np.isfinite(yaw):
        raise ValueError('Invalid global raster affine or ego pose')
    expected_scale = 1. / geometry['meters_per_pixel']
    if not np.allclose(matrix[:,:2] @ matrix[:,:2].T, np.eye(2)*expected_scale**2,rtol=1e-5,atol=1e-6):
        raise ValueError('Raster affine scale disagrees with meters_per_pixel')
    anchor = (geometry['width']*geometry['anchor_x'],geometry['height']*geometry['anchor_y'])
    if not np.allclose(matrix[:,:2] @ xy + matrix[:,2],anchor,rtol=0,atol=.01):
        raise ValueError('Raster affine does not place ego at the recorded anchor')
    orientation = geometry.get('orientation')
    if orientation not in ('ego_forward_up_ego_right_right','world_x_up_world_y_right'):
        raise ValueError('Global raster BEV must declare its stored image orientation')
    raster_yaw = np.radians(yaw if orientation == 'ego_forward_up_ego_right_right' else 0.)
    c, s = np.cos(raster_yaw),np.sin(raster_yaw)
    expected = np.array([[-s,c],[-c,-s]]) * expected_scale
    if not np.allclose(matrix[:,:2],expected,rtol=1e-5,atol=1e-6):
        raise ValueError('Raster affine orientation disagrees with recorded rotation mode')
    return matrix


def model_sampling_indices(geometry, metadata=None):
    """Nearest source pixels for fixed model cell centres, plus coverage mask."""
    x0, x1, dx = MODEL_GRID['x']
    y0, y1, dy = MODEL_GRID['y']
    x = x0 + (np.arange(int(round((x1 - x0) / dx))) + .5) * dx
    y = y0 + (np.arange(int(round((y1 - y0) / dy))) + .5) * dy
    forward, right = np.meshgrid(x, y)
    grid = BEVGeometry(**{key:geometry[key] for key in
        ('width','height','meters_per_pixel','anchor_x','anchor_y')})
    # Shared LRS-derived transform is also used by collection, preventing a
    # second independent sign/anchor convention in the training adapter.
    points = np.stack((forward,right),axis=-1)
    if metadata is not None and metadata.get('schema_version') == 4:
        affine = validate_raster_affine(metadata)
        pose = metadata['ego_pose']
        yaw = np.radians(float(pose['yaw']))
        c, s = np.cos(yaw), np.sin(yaw)
        world = points @ np.array([[c,s],[-s,c]]) + (float(pose['x']),float(pose['y']))
        uv = world @ affine[:,:2].T + affine[:,2]
    else:
        uv = world_to_local(points,(0.,0.,0.,0.),grid)
    u = np.floor(uv[...,0] + .5).astype(np.int64)
    v = np.floor(uv[...,1] + .5).astype(np.int64)
    inside = ((u >= 0) & (u < geometry['width']) & (v >= 0) & (v < geometry['height']))
    return np.clip(v, 0, geometry['height'] - 1), np.clip(u, 0, geometry['width'] - 1), inside


def model_ground_truth(scene, stem):
    """Return labels and aux arrays in the model's Y,X layout, with validity."""
    arrays, metadata = load_bev_ground_truth(scene, stem)
    v, u, inside = model_sampling_indices(_geometry(metadata),metadata)
    valid = inside & arrays['validity'][v, u].astype(bool)
    occupancy_valid = inside & arrays['occupancy_validity'][v, u].astype(bool)
    lane_valid = valid & arrays['lane_validity'][v, u].astype(bool)
    elevation_valid = valid & np.isfinite(arrays['elevation_m'][v,u])
    if 'elevation_validity' in arrays:
        elevation_valid &= arrays['elevation_validity'][v,u].astype(bool)
    result = {'validity': valid[None], 'occupancy_validity': occupancy_valid[None],
              'lane_validity': lane_valid[None], 'elevation_validity':elevation_valid[None]}
    for task, channel in HEAD_CHANNELS.items():
        label = arrays[channel][v, u].astype(np.float32)
        task_valid = occupancy_valid if task in ('vehicle', 'walker') else valid
        if task == 'lane':
            task_valid = lane_valid
        label[~task_valid] = np.nan
        result[task] = label[None]
    for name in HEIGHT_CHANNELS:
        if name in arrays:
            value = arrays[name][v, u].copy()
            height_valid = elevation_valid if name == 'elevation_m' else occupancy_valid
            value[~height_valid] = np.nan
            result[name] = value[None]
    if 'actor_id' in arrays:
        actor = arrays['actor_id'][v, u].copy()
        actor[~occupancy_valid] = -1
        result['actor_id'] = actor[None]
    semantic = arrays['semantic'][v, u].copy()
    semantic[~inside] = 0
    surface = arrays['surface_id'][v, u].copy()
    surface[~valid] = -1
    result.update(semantic=semantic[None], surface_id=surface[None])
    result['metadata'] = metadata
    return result


def validate_training_layers(metadata, selected):
    """Disabled collection layers are absent labels, never negative examples."""
    settings = metadata.get('settings')
    if not isinstance(settings, dict):
        raise ValueError('Numeric BEV requires the recorded generator settings')
    semantics = settings.get('semantic_layers', [])
    actors = settings.get('actor_classes', [])
    supported = {'das': 'drivable' in semantics, 'lane': 'lane' in semantics,
                 'vehicle': 'vehicle' in actors, 'walker': 'pedestrian' in actors}
    if metadata.get('schema_version') == 4:
        available = metadata.get('available_static_channels')
        if not isinstance(available,list) or any(not isinstance(name,str) for name in available):
            raise ValueError('Global raster BEV requires available_static_channels metadata')
        supported['das'] &= 'drivable' in available
        supported['lane'] &= 'lane' in available
    unavailable = [name for name in selected if not supported.get(name, False)]
    if unavailable:
        raise ValueError('Selected BEV labels were disabled during collection: ' + ', '.join(sorted(unavailable)))


def model_to_display(array):
    """Y,X model raster -> forward-up/right-right, without value interpolation."""
    return np.rot90(array, axes=(-2, -1)).copy()


def camera_optical_to_ego(dof):
    """CARLA's existing get_matrix rotation, then optical (right,down,forward).

    Mirrors sdv/collection/carla_engine/main.py:get_matrix and _coords_to_2d_bb.
    Applied only to explicit numeric-BEV datasets, keeping old calibration intact.
    """
    x, y, z, pitch, roll, yaw = [float(value) for value in dof]
    pitch, roll, yaw = np.radians([pitch, roll, yaw])
    cp, sp, cr, sr, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll), np.cos(yaw), np.sin(yaw)
    sensor_rotation = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr]], dtype=np.float64)
    optical_to_sensor = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
    rotation = sensor_rotation @ optical_to_sensor
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (x, y, z)
    return transform, rotation, np.array([[x], [y], [z]])


def camera_optical_to_bev_frame(scene, stem, camera):
    """Use the saved sensor pose to match yaw-only raster axes on slopes.

    A mounted sensor's nominal transform cannot recover the ego vehicle's
    pitch/roll. Actual same-frame world transforms are therefore mandatory for
    schemas 3/4; no flat-road assumption silently fills absent calibration.
    """
    path = Path(scene) / 'info' / (str(stem) + '.json')
    with path.open(encoding='utf-8') as stream:
        info = json.load(stream)
    try:
        bev = info['bev_gt']
        sync = info['sdv_metadata']['synchronization']
        packet = sync['sensors'][camera + '_rgb']
        frame = bev['frame_id']
        if frame is None or packet['frame'] != frame or sync['carla_frame'] != frame:
            raise ValueError('RGB/BEV frame mismatch')
        if abs(float(packet['timestamp_s']) - float(bev['timestamp'])) > 1e-5:
            raise ValueError('RGB/BEV timestamp mismatch')
        pose = bev['ego_pose']
        sensor = packet['world_transform']
        position, rotation = sensor['position'], sensor['rotation_deg']
        dof = [position[k] for k in ('x','y','z')] + [rotation[k] for k in ('pitch','roll','yaw')]
        if not np.isfinite(np.asarray(dof, dtype=np.float64)).all():
            raise ValueError('Non-finite sensor pose')
        _transform, optical_to_world, camera_position = camera_optical_to_ego(dof)
        yaw = np.radians(float(pose['yaw']))
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_leveled = np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
        ego_position = np.array([[float(pose[k])] for k in ('x','y','z')])
        orientation = world_to_leveled @ optical_to_world
        translation = world_to_leveled @ (camera_position - ego_position)
        transform = np.eye(4)
        transform[:3,:3], transform[:3,3] = orientation, translation[:,0]
        if not np.isfinite(transform).all():
            raise ValueError('Non-finite BEV calibration')
        return transform, orientation, translation
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Numeric BEV requires aligned actual RGB world calibration {}: {}'.format(path,error))
