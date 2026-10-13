"""Blueprint contracts without requiring a CARLA server or Python package."""
import pytest

from sdv.collection.sensor_attributes import configure_camera_blueprint, configure_lidar_blueprint


class Blueprint:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, name, value):
        self.attributes[name] = value


def test_lidar_missing_parameters_preserve_original_defaults():
    blueprint = Blueprint()
    configure_lidar_blueprint(blueprint, {'x': 0, 'z': 2.5})
    assert blueprint.attributes == {'channels': '64', 'range': '100',
                                    'points_per_second': '1200000', 'rotation_frequency': '60'}


def test_lidar_applies_all_configured_parameters():
    blueprint = Blueprint()
    configure_lidar_blueprint(blueprint, {'channels': 32, 'range': 75.5,
                                         'points_per_second': 160000, 'rotation_frequency': 10,
                                         'upper_fov': 12.5, 'lower_fov': -35})
    assert blueprint.attributes == {'channels': '32', 'range': '75.5',
                                    'points_per_second': '160000', 'rotation_frequency': '10',
                                    'upper_fov': '12.5', 'lower_fov': '-35'}


def test_lidar_partial_config_keeps_other_original_defaults():
    blueprint = Blueprint()
    configure_lidar_blueprint(blueprint, {'range': 45, 'lower_fov': -25})
    assert blueprint.attributes['range'] == '45'
    assert blueprint.attributes['channels'] == '64'
    assert blueprint.attributes['points_per_second'] == '1200000'
    assert blueprint.attributes['rotation_frequency'] == '60'
    assert blueprint.attributes['lower_fov'] == '-25'
    assert 'upper_fov' not in blueprint.attributes


def test_camera_rgb_depth_segmentation_share_resolution_and_fov_contract():
    blueprint = Blueprint()
    configure_camera_blueprint(blueprint, {'width': 640, 'height': 360, 'fov': 90.5,
                                          'yaw': 45, 'seg': True})
    assert blueprint.attributes == {'image_size_x': '640', 'image_size_y': '360', 'fov': '90.5'}


@pytest.mark.parametrize('field,value', [('channels', 0), ('channels', 4.5),
                                       ('range', -1), ('range', float('inf')),
                                       ('points_per_second', True),
                                       ('rotation_frequency', float('nan')),
                                       ('upper_fov', 91), ('lower_fov', 'not a number')])
def test_invalid_lidar_attributes_fail_before_blueprint_mutation(field, value):
    blueprint = Blueprint()
    with pytest.raises(ValueError):
        configure_lidar_blueprint(blueprint, {field: value})
    assert blueprint.attributes == {}


@pytest.mark.parametrize('config', [{'width': 640, 'height': 360},
                                    {'width': 0, 'height': 360, 'fov': 90},
                                    {'width': 640.5, 'height': 360, 'fov': 90},
                                    {'width': 640, 'height': 360, 'fov': 180}])
def test_invalid_camera_attributes_fail_before_blueprint_mutation(config):
    blueprint = Blueprint()
    with pytest.raises(ValueError):
        configure_camera_blueprint(blueprint, config)
    assert blueprint.attributes == {}
