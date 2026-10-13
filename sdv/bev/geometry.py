"""Shared CARLA world / global raster / ego-aligned BEV coordinates.

The inverse crop is adapted from LRS ``make_local_bev_ego_aligned`` and
``global_pixel_to_local_bev`` at commit
fe9cbc2f8f38fa5f7808e6d161e645160958c6a2. CARLA yaw is DEGREES at this public
boundary. World +X forward at yaw=0 maps up; world +Y maps right. Global
rasters map world +X to columns and +Y to rows, without a hidden mirror.

Pixel coordinates refer to pixel centers. An anchor is a fraction of width
or height, preserving LRS's 200 / 2 = 100 center. An anchor of 1 places ego
at the far image boundary, one pixel beyond the final pixel center.

MIT License — Copyright (c) 2026 woomin jun
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite number".format(name))
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


@dataclass(frozen=True)
class BEVGeometry:
    width: int = 200
    height: int = 200
    meters_per_pixel: float = 0.2
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value < 32767:
                raise ValueError("{} must be an integer between 1 and 32766".format(name))
            object.__setattr__(self, name, int(value))
        resolution = _finite(self.meters_per_pixel, "meters_per_pixel")
        if resolution <= 0:
            raise ValueError("meters_per_pixel must be positive")
        object.__setattr__(self, "meters_per_pixel", resolution)
        for name in ("anchor_x", "anchor_y"):
            value = _finite(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError("{} must be between 0 and 1".format(name))
            object.__setattr__(self, name, value)

    @property
    def anchor(self):
        return self.width * self.anchor_x, self.height * self.anchor_y

    def metadata(self):
        return {
            "width": self.width, "height": self.height,
            "meters_per_pixel": self.meters_per_pixel,
            "anchor_x": self.anchor_x, "anchor_y": self.anchor_y,
            "ego_anchor_pixels": list(self.anchor),
            "pixel_coordinates": "integer_pixel_centers",
            "anchor_definition": "fraction_of_width_and_height; fraction_1_is_outside_last_pixel_center",
            "orientation": "ego_forward_up_ego_right_right",
            "yaw_unit": "degrees", "world_unit": "meters",
        }


def _pose(ego_pose):
    if isinstance(ego_pose, Mapping):
        values = [ego_pose[key] for key in ("x", "y", "z", "yaw")]
    else:
        values = list(ego_pose)
        if len(values) != 4:
            raise ValueError("ego_pose must contain x, y, z, yaw_degrees")
    return tuple(_finite(value, name) for name, value in zip(("x", "y", "z", "yaw"), values))


def _points(points):
    result = np.asarray(points, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 2 or not np.isfinite(result).all():
        raise ValueError("points must be finite coordinates with final dimension 2")
    return result


def _offset(world_offset):
    result = _points(world_offset)
    if result.shape != (2,):
        raise ValueError("world_offset must contain exactly x and y")
    return result


def _resolution(global_mpp):
    result = _finite(global_mpp, "global_mpp")
    if result <= 0:
        raise ValueError("global_mpp must be positive")
    return result


def world_to_global(points_xy, world_offset, global_mpp):
    """Return subpixel global raster coordinates; never truncate world position."""
    return (_points(points_xy) - _offset(world_offset)) / _resolution(global_mpp)


def global_to_world(points_uv, world_offset, global_mpp):
    return _points(points_uv) * _resolution(global_mpp) + _offset(world_offset)


def world_to_local(points_xy, ego_pose, geometry):
    x, y, _z, yaw = _pose(ego_pose)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rotation = np.array([[-s, c], [-c, -s]], dtype=np.float64)
    return (_points(points_xy) - (x, y)).dot(rotation.T) / geometry.meters_per_pixel + geometry.anchor


def local_to_world(points_uv, ego_pose, geometry):
    x, y, _z, yaw = _pose(ego_pose)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rotation = np.array([[-s, -c], [c, -s]], dtype=np.float64)
    return ((_points(points_uv) - geometry.anchor) * geometry.meters_per_pixel).dot(rotation.T) + (x, y)


def local_to_global_affine(ego_pose, geometry, world_offset, global_mpp):
    """Destination-local to source-global 2×3 matrix for WARP_INVERSE_MAP.

    Float32 construction and operation ordering preserve the verified LRS
    nearest-neighbor crop for its original square centered configuration.
    """
    x, y, _z, yaw = _pose(ego_pose)
    center = world_to_global((x, y), world_offset, global_mpp)
    u0, v0 = geometry.anchor
    k = geometry.meters_per_pixel / _resolution(global_mpp)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([
        [-s * k, -c * k, center[0] + k * (s * u0 + c * v0)],
        [c * k, -s * k, center[1] + k * (-c * u0 + s * v0)],
    ], dtype=np.float32)


def warp_global(image, ego_pose, geometry, world_offset, global_mpp, border_value=0):
    """Crop a global label/height raster with nearest-neighbor interpolation.

    Integer semantic IDs are retained exactly. OpenCV-unsupported integer
    dtypes are converted through float64 only within its exact integer range.
    NaN floating heights remain NaN; use warp_height for companion validity.
    """
    import cv2

    source = np.asarray(image)
    if source.ndim not in (2, 3) or min(source.shape[:2]) < 1:
        raise ValueError("image must be a non-empty 2D or HWC raster")
    original_dtype = source.dtype
    if original_dtype == np.bool_:
        source = source.astype(np.uint8)
    elif original_dtype.kind in "iu" and original_dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("int16")):
        if np.any(source > 2 ** 53) or (original_dtype.kind == "i" and np.any(source < -(2 ** 53))):
            raise ValueError("integer raster IDs exceed the exact float64 range")
        source = source.astype(np.float64)
    elif original_dtype == np.float16:
        source = source.astype(np.float32)
    elif original_dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("int16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("unsupported raster dtype: {}".format(original_dtype))
    output = cv2.warpAffine(
        source, local_to_global_affine(ego_pose, geometry, world_offset, global_mpp),
        (geometry.width, geometry.height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_value),
    )
    if source.ndim == 3 and source.shape[2] == 1 and output.ndim == 2:
        output = output[..., None]
    return output.astype(original_dtype, copy=False)


def warp_height(height_m, ego_pose, geometry, world_offset, global_mpp, validity=None):
    """Return float32 meters and boolean validity, preserving missing elevation.

    Invalid and outside pixels are NaN, including when their placeholder
    source height is zero. Zero meters is valid when its validity is true.
    """
    source = np.asarray(height_m, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("height_m must be a 2D raster")
    valid = np.isfinite(source)
    if validity is not None:
        explicit = np.asarray(validity)
        if explicit.shape != source.shape or explicit.dtype.kind not in "biu" or not np.isin(explicit, (0, 1)).all():
            raise ValueError("validity must be a same-size binary mask")
        valid &= explicit.astype(bool)
    local_valid = warp_global(valid, ego_pose, geometry, world_offset, global_mpp, border_value=0)
    local_height = warp_global(np.where(valid, source, np.float32(0)), ego_pose, geometry, world_offset, global_mpp)
    local_height[~local_valid] = np.nan
    return local_height, local_valid
