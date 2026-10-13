import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from sdv.bev.cache import GlobalRasterCache


def surface(sid="road-a", x=0, z=0, slope=0, neighbors=(), road=1, lane=-1, **kwargs):
    value = {"id": sid, "road_id": road, "section_id": 0, "lane_id": lane,
             "polygon_xyz": [[x, -2, z], [x + 4, -2, z + slope * 4],
                             [x + 4, 2, z + slope * 4], [x, 2, z]],
             "center_xyz": [x + 2, 0, z + slope * 2], "neighbors": list(neighbors),
             "kind": "driving", "is_junction": False}
    value.update(kwargs)
    return value


def build(tmp_path, surfaces, **kwargs):
    return GlobalRasterCache.build(tmp_path, "TownTest", "<OpenDRIVE />", surfaces, **kwargs)


def test_flat_masks_real_coordinates_and_empty_area(tmp_path):
    cache = build(tmp_path, [surface()])
    patch = cache.patches["road-a"]
    arrays = cache.arrays("road-a")
    assert arrays["height"].dtype == np.float32
    assert set(np.unique(arrays["drivable"])) == {0, 1}
    assert np.isnan(arrays["height"][arrays["validity"] == 0]).all()
    assert (arrays["height"][arrays["validity"] > 0] == 0).all()
    assert arrays["road"].shape == (patch["bounds_px"][3] - patch["bounds_px"][1], patch["bounds_px"][2] - patch["bounds_px"][0])
    assert cache.lookup_ground(2, 0, 0)["ground_z"] == 0
    assert cache.lookup_ground(200, 0, 0) is None
    assert cache.query((100, 100, 101, 101)) == []


def test_continuous_ramp_height_no_z_bands(tmp_path):
    cache = build(tmp_path, [surface(z=3, slope=.75)])
    for x in (0, .25, 1.234, 3.75, 4):
        hit = cache.lookup_ground(x, .3, 3 + .75 * x)
        assert hit["ground_z"] == pytest.approx(3 + .75 * x, abs=1e-10)
        assert hit["layer_id"] == "road-a"
    arrays = cache.arrays("road-a")
    origin = cache.patches["road-a"]["offset_world"]
    xs = origin[0] + np.arange(arrays["height"].shape[1]) * .2
    expected = np.broadcast_to(3 + .75 * xs, arrays["height"].shape)
    np.testing.assert_allclose(arrays["height"][arrays["validity"] > 0], expected[arrays["validity"] > 0], atol=1e-6)


def test_stacked_roads_keep_distinct_surfaces_and_refuse_ambiguous(tmp_path):
    cache = build(tmp_path, [surface("lower", z=0, road=1), surface("upper", z=6, road=2)])
    assert cache.lookup_ground(2, 0, .2)["layer_id"] == "lower"
    assert cache.lookup_ground(2, 0, 6.2)["layer_id"] == "upper"
    assert cache.lookup_ground(2, 0, 3, tolerance_m=4) is None
    assert cache.lookup_ground(2, 0, 3, road_id=2, tolerance_m=4)["layer_id"] == "upper"
    assert cache.lookup_ground(2, 0, .2, road_id=2) is None
    assert cache.lookup_ground(2, 0, 0, lane_id=99) is None
    assert cache.graph_distances("lower") == {"lower": 0.0}


def test_graph_distances_physical_bidirectional_and_roi(tmp_path):
    a = surface("a", neighbors=["ramp"])
    ramp = surface("ramp", x=4, slope=.5, neighbors=["top"])
    top = surface("top", x=8, z=2)
    cache = build(tmp_path, [a, ramp, top])
    distances = cache.graph_distances("top")
    assert distances["ramp"] == pytest.approx(np.sqrt(17))
    assert distances["a"] == pytest.approx(2 * np.sqrt(17))
    assert cache.graph_distances("top", ["top", "ramp"]).keys() == {"top", "ramp"}
    assert cache.graph_distances("top", max_distance=5).keys() == {"top", "ramp"}
    assert cache.graph_distances("top", max_distance=0) == {"top": 0.0}
    with pytest.raises(ValueError, match="nonnegative"):
        cache.graph_distances("top", max_distance=-1)


def test_build_report_immutable_and_part_of_identity(tmp_path):
    report = {"skipped_edge_count": 1, "skipped_edges": [{"id": "bad-edge", "reason": "jump", "distance_m": 7.317}]}
    cache = build(tmp_path, [surface()], build_report=report)
    assert cache.metadata["build_report"] == report
    report["skipped_edge_count"] = 0
    assert cache.metadata["build_report"]["skipped_edge_count"] == 1
    assert GlobalRasterCache.load(cache.path).metadata["build_report"]["skipped_edges"][0]["distance_m"] == 7.317
    other = build(tmp_path, [surface()], build_report=report)
    assert other.path != cache.path


def test_cache_signature_reuse_and_geometry_options_invalidation(tmp_path):
    options = {"adapter_version": "test-1", "spacing_m": 2}
    first = build(tmp_path, [surface()], build_options=options)
    second = build(tmp_path, [surface()], build_options=options)
    assert first.path == second.path
    assert GlobalRasterCache.find(tmp_path, "TownTest", "<OpenDRIVE />", build_options=options).path == first.path
    assert GlobalRasterCache.find(tmp_path, "TownTest", "<OpenDRIVE changed='1' />", build_options=options) is None
    assert GlobalRasterCache.find(tmp_path, "TownTest", "<OpenDRIVE />", meters_per_pixel=.25, build_options=options) is None
    assert GlobalRasterCache.find(tmp_path, "TownTest", "<OpenDRIVE />", build_options={}) is None
    third = build(tmp_path, [surface(z=1)], build_options=options)
    assert third.path != first.path
    assert GlobalRasterCache.find(tmp_path, "TownTest", "<OpenDRIVE />", build_options=options) is None


def test_bounded_lazy_arrays_and_tampering(tmp_path):
    cache = build(tmp_path, [surface("a"), surface("b", x=8)])
    cache = GlobalRasterCache.load(cache.path, max_cached_patches=1)
    cache.arrays("a")
    assert list(cache._arrays) == ["a"]
    cache.arrays("b")
    assert list(cache._arrays) == ["b"]
    assert not cache.arrays("b")["height"].flags.writeable
    path = cache.path / cache.patches["a"]["file"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        cache.arrays("a")
    meta = cache.path / "metadata.json"
    data = json.loads(meta.read_text()); data["world_offset"] = [0, 0]
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        GlobalRasterCache.load(cache.path)


def test_no_dense_whole_town_allocation(tmp_path):
    cache = build(tmp_path, [surface("near"), surface("far", x=1000000)])
    assert cache.arrays("near")["height"].shape == cache.arrays("far")["height"].shape
    assert [p["id"] for p in cache.query((999999, -3, 1000005, 3))] == ["far"]
    assert len(cache._tiles) < 20


def test_concave_surface_does_not_fill_hull(tmp_path):
    polygon = [[0, 0, 0], [4, 0, 0], [4, 1, 0], [1, 1, 0], [1, 4, 0], [0, 4, 0]]
    cache = build(tmp_path, [surface(polygon_xyz=polygon)])
    assert cache.lookup_ground(.5, 3, 0) is not None
    assert cache.lookup_ground(3, 3, 0) is None


def test_solid_geometry_and_unresolved_broken_paint_validity(tmp_path):
    s = surface(lane_markings=[
        {"points_xyz": [[0, -1, 0], [4, -1, 0]], "width_m": .2, "type": "Solid"},
        {"points_xyz": [[0, 1, 0], [4, 1, 0]], "width_m": .2, "type": "Broken"}])
    cache = build(tmp_path, [s])
    arr = cache.arrays("road-a")
    assert arr["lane"].sum() > 0
    assert arr["lane_validity"].sum() < arr["validity"].sum()
    assert not np.any(arr["lane"][arr["lane_validity"] == 0])
    assert cache.patches["road-a"]["unsupported_markings"] == {"Broken": 1}


@pytest.mark.parametrize("bad", [
    {"polygon_xyz": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
    {"polygon_xyz": [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]]},
    {"neighbors": ["absent"]}, {"polygon_xyz": [[0, 0, 0], [1, 0, 0], [1, 1, float("nan")]]}
])
def test_invalid_geometry_rejected_without_publishing(tmp_path, bad):
    with pytest.raises(ValueError):
        build(tmp_path, [surface(**bad)])
    assert not list(tmp_path.rglob("metadata.json"))
