"""Geometric contract checks independent of CARLA/GPU availability."""

import math

import numpy as np
import pytest

from sdv.bev.geometry import (
    BEVGeometry, global_to_world, local_to_global_affine, local_to_world,
    warp_global, warp_height, world_to_global, world_to_local,
)


@pytest.mark.parametrize("yaw", [0, 90, 180, -90, 37.125, 359.999])
def test_carla_forward_is_up_and_right_is_right(yaw):
    geometry = BEVGeometry(240, 160, 0.25, 0.3, 0.8)
    pose = (123.456, -72.137, 4.12, yaw)
    theta = math.radians(yaw)
    forward = np.array([math.cos(theta), math.sin(theta)])
    right = np.array([-math.sin(theta), math.cos(theta)])
    xy = np.asarray(pose[:2])
    actual = world_to_local(np.array([xy, xy + 5 * forward, xy + 3 * right]), pose, geometry)
    anchor = np.array(geometry.anchor)
    expected = np.array([anchor, anchor + [0, -20], anchor + [12, 0]])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("yaw", [0, 90, -90, 180, 37.125, 721.5])
def test_subpixel_roundtrip_large_world_offset(yaw):
    geometry = BEVGeometry(321, 193, 0.2, 0.37, 0.85)
    pose = {"x": 712345.8123, "y": -923456.9432, "z": 14.25, "yaw": yaw}
    pixels = np.array([[0.25, 0.125], [37.125, 64.99], [320.9, 192.0005]])
    world = local_to_world(pixels, pose, geometry)
    np.testing.assert_allclose(world_to_local(world, pose, geometry), pixels, atol=6e-10)
    offset = (712000.123456, -924000.876543)
    global_pixels = world_to_global(world, offset, 0.4)
    np.testing.assert_allclose(global_to_world(global_pixels, offset, 0.4), world, atol=1e-10)


def test_inverse_affine_matches_world_coordinates_with_distinct_resolution():
    geometry = BEVGeometry(96, 64, 0.2, 0.3, 0.75)
    pose = (32.127, -41.555, 6.5, 27.5)
    pixels = np.array([[0, 0], [95, 0], [95, 63], [0, 63], list(geometry.anchor)])
    matrix = local_to_global_affine(pose, geometry, (-70.22, -92.34), 0.5)
    actual = np.column_stack((pixels, np.ones(len(pixels)))).dot(matrix.T)
    expected = world_to_global(local_to_world(pixels, pose, geometry), (-70.22, -92.34), 0.5)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("yaw", [0, 90, 180, -90])
def test_exact_class_ids_and_orientation_after_crop(yaw):
    pytest.importorskip("cv2")
    image = np.zeros((100, 100), dtype=np.uint8)
    geometry = BEVGeometry(40, 30, 0.2, 0.4, 0.8)
    pose = (10, 10, 0, yaw)
    theta = math.radians(yaw)
    forward = np.array([math.cos(theta), math.sin(theta)])
    right = np.array([-math.sin(theta), math.cos(theta)])
    points = np.array([[10, 10], np.array([10, 10]) + 2 * forward, np.array([10, 10]) + right])
    source_pixels = np.rint(world_to_global(points, (0, 0), 0.2)).astype(int)
    image[source_pixels[:, 1], source_pixels[:, 0]] = [4, 19, 255]
    local = warp_global(image, pose, geometry, (0, 0), 0.2)
    assert local.shape == (30, 40)
    assert local.dtype == np.uint8
    u, v = map(int, geometry.anchor)
    assert local[v, u] == 4
    assert local[v - 10, u] == 19
    assert local[v, u + 5] == 255
    assert set(np.unique(local)) == {0, 4, 19, 255}


def test_height_nan_validity_and_zero_meters_are_distinct():
    pytest.importorskip("cv2")
    height = np.arange(25, dtype=np.float32).reshape((5, 5)) / 10
    height[2, 2] = np.nan
    validity = np.ones(height.shape, dtype=bool)
    validity[3, 3] = False
    geometry = BEVGeometry(7, 7, 1, 0, 0)
    pose = (0, 0, 0, -90)
    cropped, valid = warp_height(height, pose, geometry, (0, 0), 1, validity)
    assert cropped.dtype == np.float32 and valid.dtype == np.bool_
    assert cropped[0, 0] == 0 and valid[0, 0]
    assert np.isnan(cropped[2, 2]) and not valid[2, 2]
    assert np.isnan(cropped[3, 3]) and not valid[3, 3]
    assert np.isnan(cropped[6, 6]) and not valid[6, 6]
    assert np.isnan(cropped[~valid]).all()
    assert valid.sum() == 23


@pytest.mark.parametrize("dtype,value", [(np.uint8, 29), (np.uint16, 65000), (np.uint32, 4000000000), (np.int32, -100000), (np.bool_, True)])
def test_integer_raster_identity_preserved(dtype, value):
    pytest.importorskip("cv2")
    image = np.full((8, 8), value, dtype=dtype)
    output = warp_global(image, (0, 0, 0, -90), BEVGeometry(8, 8, 1, 0, 0), (0, 0), 1)
    assert output.dtype == image.dtype
    np.testing.assert_array_equal(output, image)


@pytest.mark.parametrize("kwargs", [{"width": 0}, {"height": 1.5}, {"width": True}, {"meters_per_pixel": 0}, {"meters_per_pixel": float("nan")}, {"anchor_x": -0.01}, {"anchor_y": 1.01}])
def test_geometry_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BEVGeometry(**kwargs)


def test_anchor_at_one_is_declared_image_boundary_not_clamped():
    geometry = BEVGeometry(200, 100, 0.2, 0.5, 1)
    assert geometry.anchor == (100, 100)
    assert geometry.metadata()["ego_anchor_pixels"] == [100, 100]
    assert "outside" in geometry.metadata()["anchor_definition"]


def test_pose_and_world_resolution_are_validated():
    with pytest.raises(ValueError):
        world_to_global([[0, 0]], (0, 0), -0.2)
    with pytest.raises(ValueError):
        world_to_local([[0, 0]], (0, 0, 0, float("nan")), BEVGeometry())
    with pytest.raises(ValueError):
        local_to_world([[0, 0]], (0, 0, 90), BEVGeometry())
    with pytest.raises(ValueError):
        world_to_global([[0, float("inf")]], (0, 0), 0.2)


def test_invalid_height_mask_shape_is_rejected():
    with pytest.raises(ValueError):
        warp_height(np.zeros((5, 5)), (0, 0, 0, 0), BEVGeometry(), (0, 0), 0.2, np.zeros((2, 2)))


def test_subpixel_origin_is_not_truncated():
    geometry = BEVGeometry(200, 200, 0.2)
    pose = (10.099, -20.077, 0, 0)
    matrix = local_to_global_affine(pose, geometry, (0, -40), 0.2)
    center = matrix.dot([100, 100, 1])
    np.testing.assert_allclose(center, [50.495, 99.615], atol=1e-5)


def test_hwc_single_channel_shape_preserved():
    pytest.importorskip("cv2")
    image = np.ones((4, 4, 1), dtype=np.uint8)
    result = warp_global(image, (0, 0, 0, -90), BEVGeometry(4, 4, 1, 0, 0), (0, 0), 1)
    assert result.shape == image.shape
