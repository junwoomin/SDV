import hashlib
import json
from pathlib import Path
import shutil

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
from sdv.bev.global_map import (GlobalMap, GlobalMapCatalog, catalog, default_map_root,
                                load_global_map)


def test_actual_supplied_catalog_has_14_verified_variants_without_array_decode():
    result = catalog()
    assert result["errors"] == []
    assert len(result["town_records"]) == 7
    for town in result["town_records"]:
        assert town["available_ppm"] == [5, 10]
        for variant in town["variants"]:
            assert variant["available"]
            assert variant["errors"] == []
            assert variant["metric_elevation_available"] is False
            assert variant["meters_per_pixel"] == 1 / variant["ppm"]
            assert variant["source_path"].endswith(town["town"])
    t4 = next(r for r in result["town_records"] if r["town"] == "Town04")
    assert t4["variants"][0]["height_layers"] == ["low", "high"]
    assert t4["variants"][1]["width_px"] == 10294


def test_lazy_selected_loading_and_aliases_equal_exact_supplied_masks():
    value = GlobalMapCatalog().get("Town01", 5)
    assert len(value._cache) == 0
    selection = value.select_layer({"x": 0, "y": 0, "z": 0}, channels=["drivable", "lane", "crosswalk"])
    assert len(value._cache) == 0
    assert set(selection["arrays"]) == {"drivable", "lane", "crosswalk"}
    lane = selection["arrays"]["lane"]
    assert list(value._cache) == ["lane_all_BEV.png"]
    raw = np.asarray(Image.open(default_map_root() / "data5/Town01/lane_all_BEV.png"))
    np.testing.assert_array_equal(lane, (raw > 0).astype(np.uint8))
    assert not lane.flags.writeable
    assert selection["arrays"]["crosswalk"].sum() == 0
    assert selection["elevation_m"] is None


def test_rgba_alpha_is_not_occupancy():
    value = load_global_map(None, "Town03", 5)
    selection = value.select_layer((0, 0, 0, 0), channels=["drivable"])
    decoded = selection["arrays"]["drivable"]
    image = np.asarray(Image.open(default_map_root() / "data5/Town03/DAS_BEV.png"))
    assert np.all(image[:, :, 3] == 255)
    assert decoded.min() == 0 and decoded.max() == 1
    np.testing.assert_array_equal(decoded, (image[:, :, 0] > 0).astype(np.uint8))


def test_high_layer_does_not_borrow_low_stopline_and_has_no_metric_height():
    value = load_global_map(None, "Town05", 5)
    chosen = value.select_layer({"ground_z_m": 9}, channels=["drivable", "lane", "stopline", "crosswalk"])
    assert chosen["layer_id"] == "high"
    assert set(chosen["arrays"]) == {"drivable", "lane"}
    assert chosen["metadata"]["unavailable_channels"] == ["crosswalk", "stopline"]
    assert chosen["metadata"]["height_separation_available"]
    assert chosen["metadata"]["height_threshold_m"] == 7
    assert chosen["metadata"]["source_files"]["drivable"] == "DAS_BEV_high.png"
    assert chosen["elevation_m"] is None


@pytest.mark.parametrize("ppm", [5, 10])
def test_actual_native_town05_road37_ramp_uses_high_image_support(ppm):
    # Actual supplied Town05 OpenDRIVE parsed with CARLA 0.9.15:
    # Road37, lane-3, s=60. HBLR's Z>7 switch alone incorrectly selects low.
    value = load_global_map(None, "Town05", ppm)
    pose = {"x": -52.895118713378906, "y": -207.51962280273438,
            "z": 6.3, "ground_z_m": 6.094699859619141, "road_id": 37, "lane_id": -3,
            "ground_verified": True, "ground_lane_type": "Driving", "ground_source": "carla_waypoint_exact"}
    chosen = value.select_layer(pose, channels=[])
    assert chosen["layer_id"] == "high"
    assert chosen["metadata"]["threshold_overridden_by_support"] is True
    assert chosen["metadata"]["exclusive_support_used"] is True
    assert chosen["metadata"]["ground_z_m_used"] == pose["ground_z_m"]
    # A tall vehicle's top is never the ground selector.
    pose["z"] = 100
    assert value.select_layer(pose, channels=[])["layer_id"] == "high"
    pose["ground_verified"] = False
    assert value.select_layer(pose, channels=[])["layer_id"] == "low"
    pose["ground_verified"] = True
    # Identical XY alone cannot promote an unverified/sidewalk actor upstairs.
    pose["ground_lane_type"] = "Sidewalk"
    assert value.select_layer(pose, channels=[])["layer_id"] == "low"


def test_unknown_ground_height_warns_and_missing_z_fails_multilayer():
    value = load_global_map(None, "Town04", 5)
    selection = value.select_layer({"z": 6}, channels=[])
    assert selection["layer_id"] == "high"
    assert selection["metadata"]["ground_z_m_used"] is None
    assert any("uncertain" in warning for warning in selection["warnings"])
    with pytest.raises(ValueError, match="finite ground"):
        value.select_layer({}, channels=[])
    assert value.select_layer({}, height_aware=False, channels=[])["layer_id"] == "low"


@pytest.mark.parametrize("ppm", [5, 10])
def test_actual_town05_sidewalk_underbridge_never_follows_upper_das(ppm):
    # CARLA0.9.15 Map.get_waypoint(project_to_road=False, LaneType.Any)
    # identifies this exact point as Road9, Lane-5, Sidewalk, ground Z=0.
    value = load_global_map(None, "Town05", ppm)
    point = [-240.8455963, -99.4904327, 0.0]
    comparison = value.layer_for_points([point], ground_verified=True, lane_type="Driving")
    assert comparison["high_support"][0] and not comparison["low_support"][0]
    pose = dict(zip(("x", "y", "ground_z_m"), point))
    pose.update(ground_verified=True, ground_lane_type="Sidewalk", ground_source="carla_waypoint_exact", road_id=9, lane_id=-5)
    result = value.select_layer(pose, channels=[])
    assert result["layer_id"] == "low"
    assert result["metadata"]["exclusive_support_used"] is False


def test_bounded_cache_and_unchanged_source_bytes():
    value = load_global_map(None, "Town01", 5, max_cache_bytes=6 * 1024 * 1024)
    source_path = default_map_root() / "data5/Town01/DAS_BEV.png"
    before = source_path.read_bytes()
    selected = value.select_layer((0, 0, 0, 0), channels=["drivable", "lane"])
    selected["arrays"]["drivable"]
    selected["arrays"]["lane"]
    assert len(value._cache) == 1
    assert value._bytes <= 6 * 1024 * 1024
    assert source_path.read_bytes() == before


def test_missing_and_corrupt_source_rejected(tmp_path):
    bundled = default_map_root()
    shutil.copytree(str(bundled / "data5/Town01"), str(tmp_path / "data5/Town01"))
    value = load_global_map(tmp_path, "Town01", 5)
    path = tmp_path / "data5/Town01/lane_all_BEV.png"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="changed"):
        value.select_layer((0, 0, 0, 0), channels=["lane"])["arrays"]["lane"]
    with pytest.raises(ValueError, match="unavailable"):
        load_global_map(tmp_path, "Town01", 5)


def test_unknown_map_resolution_and_catalog_tamper_rejected(tmp_path):
    with pytest.raises(ValueError, match="No verified"):
        load_global_map(None, "Town06", 5)
    with pytest.raises(ValueError, match="No verified"):
        load_global_map(None, "Town01", 20)
    manifest = json.loads((default_map_root() / "catalog.json").read_text())
    manifest["variants"][0]["meters_per_pixel"] = .123
    (tmp_path / "catalog.json").write_text(json.dumps(manifest))
    assert "integrity mismatch" in catalog(tmp_path)["errors"][0]
