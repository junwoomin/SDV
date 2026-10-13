"""Real PNG decoding/cropping with controlled annotations, not CARLA captures."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('cv2')
from sdv.bev.global_map import GlobalMap
from sdv.bev.global_generator import GlobalMapBEVGenerator
from sdv.bev.geometry import world_to_local, warp_global
from sdv.bev.io import load_frame, save_frame


def _asset_map(tmp_path, town='Town04'):
    folder = tmp_path/'controlled-pngs'
    folder.mkdir()
    shape = (120, 160)
    low = np.ones(shape, dtype=np.uint8)*255
    lane = np.zeros(shape, dtype=np.uint8)
    # World point (4, 1): source col=(4+16)*5, row=(1+12)*5.
    lane[65,100] = 255
    records = {}
    for name, array in [('DAS_BEV.png',low), ('all_lane_BEV.png',lane),
                         ('DAS_BEV_high.png',low), ('all_lane_BEV_high.png',lane),
                         ('stop_line_BEV.png',np.zeros(shape,dtype=np.uint8))]:
        path = folder/name
        Image.fromarray(array).save(str(path))
        payload = path.read_bytes()
        records[name] = {'sha256':hashlib.sha256(payload).hexdigest(),
            'bytes':len(payload), 'size':[shape[1],shape[0]], 'mode':'L'}
    path = folder/'world_offset.npy'
    np.save(str(path), np.array([-16.,-12.]))
    payload = path.read_bytes()
    records[path.name] = {'sha256':hashlib.sha256(payload).hexdigest(), 'bytes':len(payload)}
    record = {'town':town, 'ppm':5, 'directory':'controlled-pngs', 'width_px':160,
        'height_px':120, 'meters_per_pixel':.2, 'world_offset_xy':[-16.,-12.],
        'source_signature':'controlled-fixture-not-real-town', 'files':records}
    return GlobalMap(tmp_path, record)


def _actor(identity, x=0., y=0., ground=0., height=2.):
    return {'actor_id':identity, 'object_class':'vehicle', 'x':x, 'y':y,
        'z':ground+.1, 'yaw':0., 'ground_z_m':ground,
        'ground_source':'controlled native Map-shaped observation',
        'road_id':1, 'lane_id':-1, 'bbox_min_z_m':ground+.1,
        'bbox_max_z_m':ground+.1+height, 'object_height_m':height,
        'footprint_world_xyz':[[x-1,y-.5,ground], [x+1,y-.5,ground],
                               [x+1,y+.5,ground], [x-1,y+.5,ground]]}


@pytest.fixture
def global_map(tmp_path):
    return _asset_map(tmp_path)


def test_png_is_cropped_directly_and_unknown_elevation_remains_unknown(global_map, tmp_path):
    generator = GlobalMapBEVGenerator(global_map, {'width':64,'height':64})
    pose = _actor(1)
    source = global_map.select_layer(pose, channels=['lane'])['arrays']['lane']
    before = source.copy()
    result = generator.generate(pose, [pose], frame_id=123, timestamp=12.3)
    arrays, meta = result['arrays'], result['metadata']
    expected = warp_global(source, pose, generator.geometry, global_map.world_offset, .2)
    np.testing.assert_array_equal(arrays['lane'], expected)
    np.testing.assert_array_equal(source, before)
    assert not source.flags.writeable
    assert meta['schema_version'] == 4
    assert arrays['validity'].all() and not arrays['elevation_validity'].any()
    assert np.isnan(arrays['elevation_m']).all()
    assert meta['elevation_available'] is False
    assert meta['static_label_source'] == 'provided_global_raster_png'
    assert Path(meta['global_map']['source_path']).is_absolute()
    assert meta['global_map']['files']['lane']['sha256'] == global_map.record['files']['all_lane_BEV.png']['sha256']
    assert set(meta['available_static_channels']) == {'drivable','lane'}
    assert set(meta['unavailable_static_channels']) == {'road_boundary','sidewalk','junction'}
    assert not arrays['static_stopline'].any()
    assert not arrays['stop_red'].any() and not arrays['stop_green'].any()
    saved = tmp_path/'frame.npz'
    save_frame(saved, arrays, metadata=meta)
    restored = load_frame(saved, meta['geometry'], metadata=meta)
    np.testing.assert_array_equal(restored['semantic'], arrays['semantic'])
    assert all(np.isfinite(v) and v >= 0 for v in meta['timings'].values())
    assert meta['timings']['total_ms'] >= sum(v for k,v in meta['timings'].items() if k != 'total_ms')-.001


@pytest.mark.parametrize('yaw', [0., 90., -90., 37.])
@pytest.mark.parametrize('rotate', [True, False])
def test_rotation_switch_and_affine_keep_real_ego_yaw(global_map, yaw, rotate):
    generator = GlobalMapBEVGenerator(global_map, {'width':64,'height':64,'rotate_with_ego':rotate})
    ego = dict(_actor(1), yaw=yaw)
    result = generator.generate(ego, [ego])
    metadata = result['metadata']
    effective = dict(ego, yaw=yaw if rotate else 0.)
    expected = warp_global(global_map.select_layer(ego, channels=['lane'])['arrays']['lane'],
        effective, generator.geometry, global_map.world_offset, .2)
    np.testing.assert_array_equal(result['arrays']['lane'], expected)
    assert metadata['ego_pose']['yaw'] == yaw
    affine = np.asarray(metadata['geometry']['world_to_bev_pixel'])
    points = np.array([[0.,0.], [4.,1.], [-2.1,3.2]])
    np.testing.assert_allclose(np.column_stack((points,np.ones(3))) @ affine.T,
        world_to_local(points, effective, generator.geometry), atol=1e-10)
    assert metadata['geometry']['orientation'] == ('ego_forward_up_ego_right_right' if rotate else 'world_x_up_world_y_right')


def test_actor_layer_uses_ground_not_top_and_unknown_ground_is_invalid(global_map):
    generator = GlobalMapBEVGenerator(global_map, {'width':64,'height':64})
    ego = _actor(1, x=-4.)
    tall = _actor(2, x=0., ground=0., height=10.)
    bridge = _actor(3, x=0., ground=8.)
    unknown = _actor(4, x=2.)
    unknown['ground_z_m'] = None
    result = generator.generate(ego, [ego,tall,bridge,unknown])
    arrays = result['arrays']
    records = {row['actor_id']:row for row in result['metadata']['actors']}
    assert records[2]['included'] and records[2]['map_layer_id'] == 'low'
    assert not records[3]['included'] and records[3]['map_layer_id'] == 'high'
    assert not records[4]['included'] and records[4]['reason'] == 'ground_height_unavailable_for_layer_selection'
    assert (arrays['actor_id'] == 2).any() and not (arrays['actor_id'] == 3).any()
    rows, cols = generator._polygon_indices(unknown['footprint_world_xyz'], ego)
    assert not arrays['occupancy_validity'][rows,cols].any()


def test_unknown_single_plane_does_not_invent_vertical_filter(tmp_path):
    generator = GlobalMapBEVGenerator(_asset_map(tmp_path, 'Town01'), {'width':64,'height':64})
    ego = _actor(1, x=-3.)
    distant_height = _actor(2, x=2., ground=8.)
    result = generator.generate(ego, [ego,distant_height])
    assert (result['arrays']['actor_id'] == 2).any()
    assert result['metadata']['height_separation_available'] is False
    assert any('single-plane' in warning for warning in result['metadata']['warnings'])


def test_canvas_padding_is_not_negative_training_data(global_map):
    generator = GlobalMapBEVGenerator(global_map, {'width':64,'height':64})
    ego = _actor(1, x=-15., y=-11.)
    result = generator.generate(ego, [])
    coverage = result['arrays']['validity']
    assert coverage.any() and not coverage.all()
    expected = warp_global(np.ones(global_map.shape,dtype=np.uint8), ego,
        generator.geometry, global_map.world_offset, .2)
    np.testing.assert_array_equal(coverage, expected)
    assert not result['arrays']['drivable'][coverage == 0].any()
    assert np.all(result['arrays']['surface_id'][coverage == 0] == -1)


def test_repeated_frames_reuse_png_and_canvas_buffers(global_map, monkeypatch):
    generator = GlobalMapBEVGenerator(global_map, {'width':64,'height':64})
    ego = _actor(1)
    generator.generate(ego, [ego])
    cached = {key:id(value) for key,value in global_map._cache.items()}
    coverage = next(iter(generator._coverage_images.values()))
    def no_disk_read(*args, **kwargs):
        raise AssertionError('online repeat must not reload a cached PNG')
    monkeypatch.setattr(Path, 'read_bytes', no_disk_read)
    generator.generate(dict(ego,x=.25), [ego])
    assert cached == {key:id(value) for key,value in global_map._cache.items()}
    assert next(iter(generator._coverage_images.values())) is coverage


def test_coarser_local_resolution_warns_about_nearest_aliasing(global_map):
    generator = GlobalMapBEVGenerator(global_map, {'width':64,'height':64,'meters_per_pixel':.25})
    pose = _actor(1)
    result = generator.generate(pose, [])
    assert any('thin lane/stopline' in message for message in result['metadata']['warnings'])
    expected = warp_global(global_map.select_layer(pose,channels=['lane'])['arrays']['lane'],
        pose, generator.geometry, global_map.world_offset, global_map.meters_per_pixel)
    np.testing.assert_array_equal(result['arrays']['lane'],expected)


def test_object_height_toggle_preserves_actor_identity(global_map):
    ego = _actor(1)
    tall, low = _actor(2, x=3., height=3.), _actor(3, x=3., height=2.)
    outputs = [GlobalMapBEVGenerator(global_map, {'width':64,'height':64,'object_height':enabled}).generate(
        ego,[ego,tall,low])['arrays'] for enabled in (True,False)]
    np.testing.assert_array_equal(outputs[0]['actor_id'],outputs[1]['actor_id'])
    assert np.isnan(outputs[1]['object_height_m']).all()


@pytest.mark.parametrize('ppm', [5,10])
def test_supplied_town05_ramp_support_preserves_native_ground_and_route(ppm):
    """Actual PNGs + audited CARLA OpenDRIVE points, no live driving claim.

    Native Town05 road37/lane-3 at s=60 and65 were independently queried
    during source audit. Both lie below HBLR's7m threshold but in supplied
    high-only DAS. The actor shape here remains an explicit test fixture.
    """
    from sdv.bev.global_map import default_map_root, load_global_map
    root = default_map_root()
    if not (root/'catalog.json').is_file():
        pytest.skip('The supplied real global PNG assets are not installed')
    actual_map = load_global_map(root, 'Town05', ppm)
    start = [-52.895118713378906, -207.51962280273438, 6.094699859619141]
    finish = [-57.895111083984375, -207.52630615234375, 6.7271928787231445]
    ego = dict(_actor(1,x=start[0],y=start[1],ground=start[2]), yaw=180.,
        road_id=37,lane_id=-3,ground_verified=True,ground_lane_type='Driving',
        ground_source='carla_waypoint_exact')
    generator = GlobalMapBEVGenerator(actual_map, {'width':64,'height':64,'global_ppm':ppm})
    result = generator.generate(ego,[ego],route=[start,finish],route_ground_verified=True)
    assert result['metadata']['map_layer_id'] == 'high'
    assert result['metadata']['global_map']['threshold_overridden_by_support'] is True
    assert result['arrays']['drivable'][32,32] == 1
    assert result['arrays']['route'].sum() > 10
    assert result['metadata']['actors'][0]['included']
    assert result['metadata']['ground_z_m'] == pytest.approx(start[2])
    assert np.isnan(result['arrays']['elevation_m']).all()
    # The old threshold remains the explicit fallback without native Driving
    # evidence; it is not silently presented as a verified layer assignment.
    uncertain = dict(ego,ground_verified=False)
    assert actual_map.select_layer(uncertain,channels=[])['layer_id'] == 'low'


def test_supplied_underbridge_sidewalk_cannot_use_high_only_drivable_override():
    from sdv.bev.global_map import default_map_root, load_global_map
    root = default_map_root()
    if not (root/'catalog.json').is_file():
        pytest.skip('The supplied real global PNG assets are not installed')
    actual_map = load_global_map(root, 'Town05', 5)
    # Actual native road9/lane-5 Sidewalk at Z=0 under a high-only DAS road.
    pose = {'x':-240.8455963,'y':-99.4904327,'z':0.,'yaw':0.,
        'ground_z_m':0.,'ground_verified':True,'ground_lane_type':'Sidewalk',
        'road_id':9,'lane_id':-5}
    selected = actual_map.select_layer(pose,channels=[])
    assert selected['layer_id'] == 'low'
    assert selected['metadata']['threshold_overridden_by_support'] is False
