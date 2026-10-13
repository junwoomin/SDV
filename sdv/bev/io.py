"""Strict, pickle-free BEV labels and atomic frame writes."""
import os
import tempfile
from pathlib import Path
import numpy as np
from .schema import BINARY_CHANNELS, HEIGHT_CHANNELS, SEMANTIC_CLASSES, SEMANTIC_ORDER

def semantic_from_channels(arrays):
    result = np.zeros(arrays['validity'].shape, dtype=np.uint8)
    for channel, class_id in SEMANTIC_ORDER:
        if channel in arrays: result[arrays[channel] > 0] = class_id
    return result

def colorize(semantic):
    ids = np.asarray(semantic)
    palette = np.array([r['color'] for r in SEMANTIC_CLASSES], dtype=np.uint8)
    if ids.ndim != 2 or ids.dtype.kind not in 'iu' or np.any(ids >= len(palette)) or np.any(ids < 0):
        raise ValueError('Invalid BEV semantic IDs')
    return palette[ids]

def validate_arrays(arrays, geometry=None, metadata=None):
    required = set(BINARY_CHANNELS + HEIGHT_CHANNELS + ('semantic','actor_id','surface_id'))
    if required-set(arrays): raise ValueError('Missing BEV channels: '+', '.join(sorted(required-set(arrays))))
    shape = arrays['semantic'].shape
    if len(shape) != 2 or not 1 <= min(shape) <= max(shape) <= 1024:
        raise ValueError('BEV arrays must have a bounded two-dimensional shape')
    if geometry is not None:
        if not isinstance(geometry,dict) or any(type(geometry.get(k)) is not int for k in ('height','width')):
            raise ValueError('BEV geometry requires integer width and height')
        if shape != (geometry['height'], geometry['width']): raise ValueError('BEV geometry/array shape mismatch')
    for name in required:
        a = arrays[name]
        if a.shape != shape: raise ValueError('BEV channel shape mismatch: '+name)
        dtype = 'float32' if name in HEIGHT_CHANNELS else ('int32' if name in ('actor_id','surface_id') else 'uint8')
        if a.dtype != np.dtype(dtype): raise ValueError('BEV channel dtype mismatch: '+name)
        if name in BINARY_CHANNELS and not np.isin(a,(0,1)).all(): raise ValueError('BEV channel must be binary: '+name)
        if name in HEIGHT_CHANNELS and np.isinf(a).any(): raise ValueError('Infinite BEV heights: '+name)
    if np.any(arrays['semantic'] > 13): raise ValueError('Invalid BEV class ID')
    if 'static_stopline' in arrays:
        stopline=arrays['static_stopline']
        if stopline.shape!=shape or stopline.dtype!=np.uint8 or not np.isin(stopline,(0,1)).all():
            raise ValueError('static_stopline must be a same-size uint8 binary mask')
    elevation_validity=arrays.get('elevation_validity')
    if metadata is not None and metadata.get('schema_version')==4 and elevation_validity is None:
        raise ValueError('Global raster schema 4 requires explicit elevation_validity')
    if elevation_validity is not None:
        if elevation_validity.shape!=shape or elevation_validity.dtype!=np.uint8 or not np.isin(elevation_validity,(0,1)).all():
            raise ValueError('elevation_validity must be a same-size uint8 binary mask')
        if not np.array_equal(elevation_validity.astype(bool),np.isfinite(arrays['elevation_m'])):
            raise ValueError('Elevation validity must exactly identify finite metric heights')
        if np.any(elevation_validity>arrays['validity']):
            raise ValueError('Metric elevation cannot be valid outside map coverage')
    elif np.any(arrays['validity'].astype(bool) & ~np.isfinite(arrays['elevation_m'])):
        raise ValueError('Valid road surface requires finite elevation')
    return arrays

def load_frame(path, geometry=None, metadata=None):
    import zipfile
    try: return _load_frame(path,geometry,metadata)
    except (zipfile.BadZipFile,EOFError,KeyError,TypeError) as error:
        raise ValueError('Invalid BEV archive: '+str(error)) from error

def _load_frame(path, geometry=None, metadata=None):
    import zipfile
    path = Path(path)
    # Bound decompression before NumPy allocation, even for imported datasets.
    with zipfile.ZipFile(str(path)) as archive:
        members = archive.infolist()
        if len(members) > 64 or len({i.filename for i in members}) != len(members):
            raise ValueError('Invalid BEV archive members')
        if sum(i.file_size for i in members) > 192*1024*1024:
            raise ValueError('BEV frame exceeds decompressed size limit')
        for item in members:
            if not item.filename.endswith('.npy') or '/' in item.filename:
                raise ValueError('Only flat numeric NPY channels are allowed')
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1,0): shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2,0): shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else: raise ValueError('Unsupported BEV NPY version')
                if len(shape) != 2 or not 1 <= min(shape) <= max(shape) <= 1024 or dtype not in (np.dtype('uint8'),np.dtype('int32'),np.dtype('float32')):
                    raise ValueError('Invalid BEV NPY shape or dtype')
                if item.file_size != stream.tell() + int(np.prod(shape))*dtype.itemsize:
                    raise ValueError('BEV NPY payload length mismatch')
    with np.load(str(path), allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    return validate_arrays(arrays, geometry, metadata)

def save_frame(path, arrays, metadata=None):
    validate_arrays(arrays,metadata=metadata)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp,str(path))
    finally:
        if os.path.exists(temp): os.unlink(temp)
