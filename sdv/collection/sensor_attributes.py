"""Shared CARLA blueprint attributes for preview and collection.

The old LiDAR constructors ignored fields saved by the sensor editor. Reading
those fields here makes the displayed configuration effective on both paths.
Missing fields retain the original constructor values; no research data or
coordinate transformations are changed. This module is stdlib-only for the
separate CARLA Python interpreter (including Python 3.7).
"""
import math
from collections.abc import Mapping


LIDAR_LEGACY_DEFAULTS = {
    'channels': 64,
    'range': 100,
    'points_per_second': 1200000,
    'rotation_frequency': 60,
}


def _numeric_attribute(name, value, integral=False, angle=False):
    if isinstance(value, bool):
        raise ValueError('{}: 숫자를 지정하세요.'.format(name))
    try:
        parsed = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('{}: 유한한 숫자를 지정하세요.'.format(name))
    if not math.isfinite(parsed):
        raise ValueError('{}: 유한한 숫자를 지정하세요.'.format(name))
    if angle:
        if not -90 <= parsed <= 90:
            raise ValueError('{}: -90~90도 범위를 지정하세요.'.format(name))
    elif parsed <= 0:
        raise ValueError('{}: 0보다 큰 값을 지정하세요.'.format(name))
    if integral:
        if not parsed.is_integer():
            raise ValueError('{}: 정수를 지정하세요.'.format(name))
        return str(int(parsed))
    return format(parsed, '.15g')


def _set_attributes(blueprint, attributes):
    # Validate the complete input first. Invalid fields must not leave a
    # partially configured blueprint that a caller could accidentally spawn.
    for name, value in attributes.items():
        blueprint.set_attribute(name, value)
    return attributes


def configure_camera_blueprint(blueprint, config):
    """Apply RGB/semantic/depth camera resolution and horizontal field of view."""
    if not isinstance(config, Mapping):
        raise ValueError('카메라 설정은 객체 형식이어야 합니다.')
    missing = {'width', 'height', 'fov'} - set(config)
    if missing:
        raise ValueError('카메라 설정 누락: {}'.format(', '.join(sorted(missing))))
    attributes = {
        'image_size_x': _numeric_attribute('width', config['width'], integral=True),
        'image_size_y': _numeric_attribute('height', config['height'], integral=True),
        'fov': _numeric_attribute('fov', config['fov']),
    }
    if float(attributes['fov']) >= 180:
        raise ValueError('fov: 0도 초과, 180도 미만을 지정하세요.')
    return _set_attributes(blueprint, attributes)


def configure_lidar_blueprint(blueprint, config):
    """Apply common ray_cast and ray_cast_semantic attributes with old defaults."""
    if not isinstance(config, Mapping):
        raise ValueError('LiDAR 설정은 객체 형식이어야 합니다.')
    attributes = {
        name: _numeric_attribute(name, config.get(name, default),
                                 integral=name in {'channels', 'points_per_second'})
        for name, default in LIDAR_LEGACY_DEFAULTS.items()
    }
    for name in ('upper_fov', 'lower_fov'):
        if name in config:
            attributes[name] = _numeric_attribute(name, config[name], angle=True)
    return _set_attributes(blueprint, attributes)
