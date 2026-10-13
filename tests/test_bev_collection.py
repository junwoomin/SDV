"""CARLA-shaped fixtures exercise production collection boundaries; no server claim."""
import json
import math
from types import SimpleNamespace as NS

import pytest
np = pytest.importorskip('numpy')
pytest.importorskip('cv2')

from sdv.bev.schema import normalize_bev_settings
from sdv.collection.bev_adapter import (surface_patches, actor_record, build_cache,
    CarlaBEVAdapter, save_ground_truth, compatibility_images, legacy_route_pixels)


def vec(x=0., y=0., z=0.):
    return NS(x=float(x), y=float(y), z=float(z))


def transform(x=0., y=0., z=0., yaw=0.):
    return NS(location=vec(x, y, z), rotation=NS(yaw=yaw, pitch=0., roll=0.),
              get_right_vector=lambda: vec(-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0))


class MapFixture:
    name = '/Game/Carla/Maps/TownFixture'
    def __init__(self):
        self.calls = 0
        self.waypoints = {}
    def point(self, x, road=1, lane=-1):
        key = (float(x), road, lane)
        if key not in self.waypoints:
            z = 0 if road == 1 else 6
            y = 0 if lane == -1 else 3
            wp = NS(s=float(x)+10, road_id=road, section_id=0, lane_id=lane,
                transform=transform(x, y, z), lane_width=4 if lane == -1 else 2,
                lane_type='Driving' if lane == -1 else 'Sidewalk', is_junction=False,
                left_lane_marking=NS(type='Solid', width=.12), right_lane_marking=NS(type='Broken', width=.12))
            wp.next = lambda d: [self.point(x+d, road, lane)] if x+d <= 10 else []
            wp.get_left_lane = lambda: self.point(x, road, -1) if lane == -2 else None
            wp.get_right_lane = lambda: self.point(x, road, -2) if lane == -1 else None
            self.waypoints[key] = wp
        return self.waypoints[key]
    def generate_waypoints(self, spacing):
        self.calls += 1
        # Match real 0.9.15: only driving is generated, sidewalk via lane links.
        return [self.point(x, road) for road in (1, 2) for x in np.arange(-8, 10.1, spacing)]
    def to_opendrive(self):
        return '<OpenDRIVE fixture="controlled_geometry_not_a_real_town" />'
    def get_waypoint(self, loc, project_to_road=False, **kwargs):
        assert project_to_road is False
        if abs(loc.y) > 4 or abs(loc.x) > 10:
            return None
        return self.point(loc.x, 1 if loc.z < 3 else 2, -1 if loc.y < 2 else -2)


def actor(identity, x=0, y=0, z=0, yaw=0, height=2, pedestrian=False):
    tf = transform(x, y, z, yaw)
    extent = vec(.3 if pedestrian else 2, .25 if pedestrian else 1, height/2)
    def vertices(at):
        c, s = math.cos(math.radians(at.rotation.yaw)), math.sin(math.radians(at.rotation.yaw))
        return [vec(at.location.x + c*dx - s*dy, at.location.y + s*dx + c*dy,
                    at.location.z + dz + height/2)
                for dx in (-extent.x, extent.x) for dy in (-extent.y, extent.y) for dz in (-extent.z, extent.z)]
    box = NS(extent=extent, location=vec(z=height/2), rotation=NS(yaw=0., pitch=0., roll=0.), get_world_vertices=vertices)
    a = NS(id=identity, type_id='walker.pedestrian.1' if pedestrian else 'vehicle.test', attributes={}, bounding_box=box)
    # Deliberately different LIVE pose; code must not use this.
    a.get_transform = lambda: transform(900, 900, 99)
    state = NS(get_transform=lambda: tf, get_velocity=lambda: vec(3, 0, 0))
    return a, state


def snapshot(states):
    return NS(frame=123, timestamp=NS(elapsed_seconds=12.3), find=lambda identity: states.get(identity))


def test_surface_builder_discovers_sidewalks_and_keeps_overpass_graphs_separate():
    patches = surface_patches(MapFixture())
    assert {p['kind'] for p in patches} == {'driving', 'sidewalk'}
    by_id = {p['id']: p for p in patches}
    for patch in patches:
        assert len(patch['polygon_xyz']) == 4
        assert patch['neighbors']
        assert all(by_id[other]['road_id'] == patch['road_id'] for other in patch['neighbors'])
        assert {line['type'] for line in patch['lane_markings']} == {'Solid', 'Broken'}
    assert any(by_id[n]['kind'] == 'sidewalk' for p in patches if p['kind']=='driving' for n in p['neighbors'])


def test_cache_reuse_skips_all_geometry_traversal(tmp_path):
    m = MapFixture(); world = NS(get_map=lambda: m)
    settings = normalize_bev_settings({'mode':'height_aware'})
    first = build_cache(world, tmp_path, settings)
    assert m.calls == 1
    second = build_cache(world, tmp_path, settings)
    assert m.calls == 1 and first.path == second.path
    build_cache(world, tmp_path, dict(settings, meters_per_pixel=.25))
    assert m.calls == 2


def test_discontinuous_opendrive_successor_is_reported_not_filled():
    # Reproduces Town01 sidewalk road25/lane-3 -> road335/lane1: next(2)
    # can span a 7.317 m physical gap. Such a link is not a road polygon.
    m = MapFixture()
    source = m.point(0, lane=-2)
    source.next = lambda distance: [m.point(8, lane=-2)]
    report = []
    patches = surface_patches(m, diagnostics=report)
    assert patches and len(report) == 1
    assert report[0]['reason'] == 'discontinuous_successor_geometry'
    assert report[0]['distance_m'] == 8
    assert not any(p['id'].startswith('1:0:-2:10.0000>') for p in patches)


def test_positive_lane_road_entry_tail_uses_actual_topology_endpoint():
    # CARLA generate_waypoints samples increasing s, while positive-lane next
    # decreases s. Without the actual road-end seed the final odd metre is lost.
    def point(s):
        wp = NS(s=s,road_id=1,section_id=0,lane_id=1,lane_type='Driving',
            lane_width=4,transform=transform(s,0,0,yaw=180),is_junction=False,
            left_lane_marking=NS(type='Solid',width=.12),right_lane_marking=NS(type='Solid',width=.12))
        wp.next = lambda d: [point(s-d)] if s-d > 0 else []
        wp.get_left_lane = wp.get_right_lane = lambda:None
        return wp
    m = NS(generate_waypoints=lambda spacing:[point(s) for s in (.0001,2.0001,4.0001,6.0001,8.0001)],
           get_topology=lambda:[(point(9.9),point(.0001))])
    patches = surface_patches(m)
    assert max(vertex[0] for patch in patches for vertex in patch['polygon_xyz']) == pytest.approx(9.9)


def test_patch_road_identity_stops_at_native_boundary_but_graph_connects():
    def point(x):
        road = 1 if x < 3 else 2
        wp = NS(s=x if road==1 else x-3,road_id=road,section_id=0,lane_id=-1,lane_type='Driving',
            lane_width=4,transform=transform(x),is_junction=False,
            left_lane_marking=NS(type='Solid',width=.12),right_lane_marking=NS(type='Solid',width=.12))
        wp.next = lambda d: [point(x+d)] if x+d <= 8 else []
        wp.get_left_lane = wp.get_right_lane = lambda:None
        return wp
    m = NS(generate_waypoints=lambda spacing:[point(x) for x in (0,2,3,5,7)],
           get_topology=lambda:[(point(0),point(3)),(point(3),point(8))])
    patches = surface_patches(m)
    by_id = {p['id']:p for p in patches}
    assert max(v[0] for p in patches if p['road_id']==1 for v in p['polygon_xyz']) <= 3
    assert any(by_id[n]['road_id']==2 for p in patches if p['road_id']==1 for n in p['neighbors'])


def test_positive_lane_cut_includes_exact_native_zero_endpoint():
    def point(s):
        wp = NS(s=s if s>=0 else -s,road_id=1 if s>=0 else 2,section_id=0,lane_id=1,
            lane_type='Driving',lane_width=4,transform=transform(s,0,0,yaw=180),is_junction=False,
            left_lane_marking=NS(type='Solid',width=.12),right_lane_marking=NS(type='Solid',width=.12))
        wp.next = lambda d:[point(s-d)]
        wp.get_left_lane = wp.get_right_lane = lambda:None
        return wp
    m = NS(generate_waypoints=lambda spacing:[point(1.11),point(3.11)],
        get_topology=lambda:[(point(3.11),point(0))],get_waypoint_xodr=lambda road,lane,s:point(s))
    patches = surface_patches(m)
    # A >0 assertion catches the observed 3–8 micrometre gap without relaxing
    # triangle containment or ground-matching tolerances anywhere else.
    assert min(patch['end_s_m'] for patch in patches if patch['road_id']==1) == 0.0


@pytest.mark.parametrize('opposite_z,opposite_road,connected', [(0.,1,True),(6.,1,False),(0.,2,False)])
def test_native_lateral_median_links_visibility_without_filling_median(opposite_z, opposite_road, connected):
    def point(x,lane):
        wp = NS(s=x+10,road_id=opposite_road if lane==-3 else 1,section_id=0,lane_id=lane,
            lane_type='Shoulder' if lane==-2 else 'Driving',lane_width=2 if lane==-2 else 4,
            transform=transform(x,3 if lane==-2 else (7 if lane==-3 else 0),opposite_z if lane==-3 else 0),
            is_junction=False,left_lane_marking=NS(type='Solid',width=.12),right_lane_marking=NS(type='Solid',width=.12))
        wp.next = lambda d:[point(x+d,lane)] if x+d<=6 else []
        wp.get_left_lane = lambda:point(x,lane+1) if lane<-1 else None
        wp.get_right_lane = lambda:point(x,lane-1) if lane>-3 else None
        return wp
    m = NS(generate_waypoints=lambda spacing:[point(x,lane) for x in (0,2,4,6) for lane in (-1,-3)])
    patches = surface_patches(m)
    indexed = {p['id']:p for p in patches}
    linked = any(indexed[n]['lane_id']==-3 for p in patches if p['lane_id']==-1 for n in p['neighbors'])
    assert linked is connected
    assert all(p['lane_id'] != -2 for p in patches)
    # Unsupported median remains outside every rasterized road polygon.
    assert not any(min(v[1] for v in p['polygon_xyz']) < 3 < max(v[1] for v in p['polygon_xyz']) for p in patches)


def test_snapshot_full_bbox_and_ground_are_separate_from_vehicle_height():
    a, state = actor(7, x=2, z=0, yaw=90, height=3)
    record = actor_record(a, snapshot({7: state}), MapFixture())
    assert record['position'] == [2, 0, 0]
    assert record['ground_z_m'] == 0
    assert record['ground_verified'] and record['ground_source'] == 'carla_waypoint_exact'
    assert record['ground_lane_type'] == 'Driving' and record['ground_road_s_m'] == 12
    assert record['bbox_min_z_m'] == 0 and record['bbox_max_z_m'] == 3
    assert record['object_height_m'] == 3
    assert record['ground_contact_world_xyz'] == [2, 0, 0]
    points = np.array(record['footprint_world_xyz'])
    np.testing.assert_allclose(points[:, 0].max()-points[:, 0].min(), 2)
    np.testing.assert_allclose(points[:, 1].max()-points[:, 1].min(), 4)
    assert record['frame_id'] == 123 and len(record['bbox_vertices_world_xyz']) == 8
    assert record['ground_contact_measured'] is False
    assert actor_record(a, snapshot({}), MapFixture()) is None


def test_pedestrian_off_map_has_unknown_ground_not_nearest_road():
    a, state = actor(8, x=20, y=20, pedestrian=True)
    record = actor_record(a, snapshot({8: state}), MapFixture())
    assert record['object_class'] == 'pedestrian'
    assert record['ground_z_m'] is None and record['lane_id'] is None
    assert record['ground_verified'] is False


def test_native_waypoint_on_wrong_deck_is_not_verified_from_xy_or_roof():
    a, state = actor(8, height=7)
    m = MapFixture()
    m.get_waypoint = lambda location, **kwargs: m.point(location.x, road=2)
    record = actor_record(a, snapshot({8:state}), m)
    assert record['bbox_max_z_m'] == 7
    assert record['ground_candidate_z_m'] == 6
    assert record['ground_z_m'] is None and record['ground_verified'] is False
    assert record['ground_height_error_m'] == 6
    assert record['ground_height_tolerance_m'] == .75


def test_ground_lookup_uses_bottom_face_center_and_preserves_sidewalk_kind():
    a, state = actor(8, y=3, height=12, pedestrian=True)
    record = actor_record(a, snapshot({8:state}), MapFixture())
    assert record['ground_verified'] is True and record['ground_z_m'] == 0
    assert record['ground_lane_type'] == 'Sidewalk'
    assert record['bbox_max_z_m'] == 12  # top does not select the bridge at z=6


def test_motorcycle_is_not_relabelled_as_cyclist_from_wheel_count():
    a, state = actor(8)
    a.attributes = {'number_of_wheels': '2'}
    a.semantic_tags = [18]
    assert actor_record(a, snapshot({8:state}), MapFixture())['object_class'] == 'vehicle'
    a.semantic_tags = [19]
    assert actor_record(a, snapshot({8:state}), MapFixture())['object_class'] == 'cyclist'


def test_bottom_face_center_differs_from_lowest_corner_on_slope():
    a, state = actor(5)
    # Actual returned vertex order from CARLA BoundingBox.h; pitch makes the
    # lower corner z=-1 at x=-2, while bottom-face center is x=0,z=0.
    a.bounding_box.get_world_vertices = lambda tf: [vec(dx,dy,dz + dx*.5)
        for dx in (-2,2) for dy in (-1,1) for dz in (0,2)]
    record = actor_record(a, snapshot({5:state}), MapFixture())
    assert record['bbox_min_z_m'] == -1
    assert record['ground_contact_world_xyz'] == [0,0,0]


def test_actual_carla_box_rotation_and_snapshot_transform_without_server():
    carla = pytest.importorskip('carla')
    bbox = carla.BoundingBox(carla.Location(x=1,y=2,z=3), carla.Vector3D(x=2,y=1,z=3))
    bbox.rotation = carla.Rotation(pitch=5,roll=10,yaw=12)
    pose = carla.Transform(carla.Location(x=2,y=0,z=1), carla.Rotation(pitch=20,roll=15,yaw=30))
    state = NS(get_transform=lambda:pose, get_velocity=lambda:carla.Vector3D())
    a = NS(id=91,type_id='vehicle.test',bounding_box=bbox,semantic_tags=[14])
    record = actor_record(a, snapshot({91:state}), MapFixture())
    up = bbox.rotation.get_up_vector()
    local_bottom = carla.Location(x=bbox.location.x-up.x*bbox.extent.z,
        y=bbox.location.y-up.y*bbox.extent.z,z=bbox.location.z-up.z*bbox.extent.z)
    expected = pose.transform(local_bottom)
    np.testing.assert_allclose(record['ground_contact_world_xyz'], [expected.x,expected.y,expected.z], atol=2e-6)
    assert abs(record['bbox_min_z_m']-expected.z) > .1
    assert record['bbox_extent_m'] == {'x':2.,'y':1.,'z':3.}


def test_collector_ground_truth_stop_state_and_npz_roundtrip(tmp_path):
    ego, ego_state = actor(1)
    car, car_state = actor(2, x=5)
    bridge, bridge_state = actor(3, x=5, z=6, height=3)
    pedestrian, pedestrian_state = actor(4, x=2, y=3, pedestrian=True)
    m = MapFixture()
    light = NS(id=9, type_id='traffic.traffic_light', state='Red', calls=0)
    def stop_waypoints():
        light.calls += 1
        return [m.point(6)]
    light.get_stop_waypoints = stop_waypoints
    light.get_state = lambda: light.state
    world = NS(get_map=lambda: m, get_actors=lambda: [ego, car, bridge, pedestrian, light])
    adapter = CarlaBEVAdapter(world, tmp_path/'cache', normalize_bev_settings({'mode':'height_aware','width':128,'height':128}))
    snap = snapshot({1:ego_state, 2:car_state, 3:bridge_state, 4:pedestrian_state})
    agent = NS(_local_planner=NS(_waypoints_queue=[(m.point(x), 'LANEFOLLOW') for x in (0, 2, 4, 6)]))
    result = adapter.generate(ego, snap, agent)
    arrays = result['arrays']
    assert arrays['ego'].sum() and arrays['vehicle'].sum() and arrays['pedestrian'].sum()
    assert arrays['stop_red'].sum() and not arrays['stop_green'].sum()
    assert arrays['object_max_z_m'][arrays['vehicle'] > 0].max() == 2
    assert result['metadata']['frame_id'] == 123
    light.state = 'Green'
    next_result = adapter.generate(ego, snap, agent)
    assert light.calls == 1  # fixed geometry cached, state never cached
    assert next_result['arrays']['stop_green'].sum() and not next_result['arrays']['stop_red'].sum()
    saved = save_ground_truth(tmp_path/'scene', '00001_a', result)
    from sdv.bev.io import load_frame
    restored = load_frame(tmp_path/'scene'/saved['path'])
    np.testing.assert_array_equal(restored['semantic'], arrays['semantic'])
    assert saved['online_input'] is False and saved['role'] == 'simulator_ground_truth'
    images = compatibility_images(result)
    np.testing.assert_array_equal(images['vehicle_mask'], arrays['vehicle']*255)
    assert images['bev'].shape == (128,128,3)
    assert len(legacy_route_pixels([(2,0)], adapter.last_ego_record, adapter.settings)) == 1
    json.dumps(saved, allow_nan=False)


def test_manifest_records_requested_geometry_and_label_role():
    from sdv.collection.metadata import collection_schema
    settings = normalize_bev_settings({'mode':'height_aware','width':256,'height':128,'anchor_y':.75})
    data = collection_schema(settings)
    assert data['schema_version'] == 3
    assert data['coordinates']['bev']['width'] == 256
    assert data['coordinates']['bev']['height'] == 128
    assert data['coordinates']['bev']['ego_anchor_pixels'] == [128.,96.]
    assert data['encodings']['bev_gt']['online_input'] is False


def test_static_stopline_export_does_not_invent_dynamic_signal_state():
    names = ['semantic','drivable','lane','vehicle','pedestrian','route',
             'stop_red','stop_yellow','stop_green','static_stopline']
    arrays = {name:np.zeros((4,4),dtype=np.uint8) for name in names}
    arrays['static_stopline'][0,1] = 1
    arrays['stop_red'][2,3] = 1
    result = {'arrays':arrays,'preview_rgb':np.zeros((4,4,3),dtype=np.uint8)}
    images = compatibility_images(result)
    assert images['stopline'][0,1] == 255 and images['stopline'][2,3] == 255
    assert not images['light_topview'][0,1].any()
    np.testing.assert_array_equal(images['light_topview'][2,3],[255,0,0])


def test_metadata_motion_uses_snapshot_state(tmp_path):
    from sdv.collection.metadata import MetadataRecorder
    ego, state = actor(1, x=2)
    state.get_acceleration = lambda: vec(4)
    state.get_angular_velocity = lambda: vec(z=5)
    m = MapFixture()
    # Metadata API uses legacy default projection; separate fixture wrapper.
    road = NS(name=m.name, get_waypoint=lambda location: m.point(location.x))
    record = MetadataRecorder().capture(NS(get_map=lambda: road, get_actors=lambda:[ego]), ego,
        NS(), tmp_path, 'sample', snapshot({1:state}), {})
    assert record['ego']['world_position']['x'] == 2
    assert record['ego']['acceleration_m_s2']['x'] == 4
    assert record['ego']['state_source'] == 'WorldSnapshot'


def _global_png_catalog(root):
    """Small controlled PNGs exercise real asset loading, never claim Town GT."""
    import hashlib
    from PIL import Image
    from sdv.bev.global_map import CATALOG_VERSION
    folder = root/'controlled-pngs'
    folder.mkdir(parents=True)
    files = {}
    for name, fill in [('DAS_BEV.png',255),('DAS_BEV_high.png',255),
                       ('all_lane_BEV.png',0),('all_lane_BEV_high.png',0)]:
        path = folder/name
        Image.fromarray(np.full((160,160),fill,dtype=np.uint8)).save(str(path))
        payload = path.read_bytes()
        files[name] = {'sha256':hashlib.sha256(payload).hexdigest(),
            'bytes':len(payload),'size':[160,160],'mode':'L'}
    path = folder/'world_offset.npy'
    np.save(str(path),np.array([-16.,-16.]))
    payload = path.read_bytes()
    files[path.name] = {'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
    record = {'town':'Town04','ppm':5,'directory':'controlled-pngs','width_px':160,
        'height_px':160,'meters_per_pixel':.2,'world_offset_xy':[-16.,-16.],'files':files}
    record['source_signature'] = hashlib.sha256(json.dumps(record,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
    (root/'catalog.json').write_text(json.dumps({'format':CATALOG_VERSION,'variants':[record]}))
    return root


@pytest.mark.parametrize('rotate', [True,False])
def test_global_collection_uses_verified_png_without_rebuilding_geometry(tmp_path, monkeypatch, rotate):
    from sdv.collection import bev_adapter as boundary
    from sdv.collection.metadata import collection_schema, atomic_json
    from sdv.bev.io import load_frame
    root = _global_png_catalog(tmp_path/'assets')
    settings = normalize_bev_settings({'mode':'global_raster','map_root':str(root),
        'width':128,'height':128,'rotate_with_ego':rotate})
    monkeypatch.delenv('SDV_BEV_MAP_ROOT', raising=False)
    m = MapFixture(); m.name = '/Game/Carla/Maps/Town04'
    def forbidden(*args, **kwargs):
        raise AssertionError('PNG collection must never reconstruct map geometry')
    m.to_opendrive = m.generate_waypoints = forbidden
    monkeypatch.setattr(boundary,'build_cache',forbidden)
    monkeypatch.setattr(boundary,'surface_patches',forbidden)
    boundary.validate_global_maps(['Town04'],settings)
    ego, ego_state = actor(1,yaw=90)
    other, other_state = actor(2,x=5,height=8)
    bridge, bridge_state = actor(3,x=5,z=6)
    unknown, unknown_state = actor(4,y=5,pedestrian=True)
    light = NS(id=9,type_id='traffic.traffic_light',get_stop_waypoints=lambda:[m.point(6)],get_state=lambda:'Red')
    world = NS(get_map=lambda:m,get_actors=lambda:[ego,other,bridge,unknown,light])
    adapter = CarlaBEVAdapter(world,tmp_path/'unused-geometry-cache',settings)
    snap = snapshot({1:ego_state,2:other_state,3:bridge_state,4:unknown_state})
    agent = NS(_local_planner=NS(_waypoints_queue=[(m.point(x),'LANEFOLLOW') for x in (0,2,4)]))
    result = adapter.generate(ego,snap,agent)
    assert adapter.cache is None and not (tmp_path/'unused-geometry-cache').exists()
    meta, arrays = result['metadata'], result['arrays']
    assert meta['schema_version'] == 4 and meta['frame_id'] == 123 and meta['timestamp'] == 12.3
    assert meta['ego_pose']['yaw'] == 90 and meta['route_ground_verified'] is True
    actors = {value['actor_id']:value for value in meta['actors']}
    assert actors[2]['included'] and not actors[3]['included'] and not actors[4]['included']
    assert np.isnan(arrays['elevation_m']).all() and not arrays['elevation_validity'].any()
    assert arrays['vehicle'].any() and arrays['stop_red'].any() and arrays['route'].any()
    assert adapter.traffic([light])[0]['ground_lane_type'] == 'Driving'
    scene = tmp_path/'scene'
    atomic_json(scene/'collection_manifest.json',collection_schema(settings))
    boundary.annotate_scene_map(scene,adapter)
    saved = save_ground_truth(scene,'00001_a',result)
    atomic_json(scene/'info/00001_a.json',{'frame':123,'bev_gt':saved})
    restored = load_frame(scene/saved['path'],metadata=saved)
    np.testing.assert_array_equal(restored['semantic'],arrays['semantic'])
    manifest = json.loads((scene/'collection_manifest.json').read_text())
    assert manifest['schema_version'] == 4
    assert manifest['bev_map']['online_geometry_rebuilt'] is False
    assert manifest['bev_map']['files']['DAS_BEV.png']['sha256']
    assert manifest['encodings']['bev_gt']['metric_road_elevation_available'] is False
    assert saved['online_input'] is False
    expected = [[64.,54.]] if rotate else [[74.,64.]]
    # Independent world-fixed versus ego-yaw projection of world point (0,2).
    np.testing.assert_allclose(np.array(legacy_route_pixels([(0,2)],adapter.last_ego_record,settings)).reshape(-1,2),expected,atol=1e-8)


def test_global_map_validation_uses_frozen_job_root_and_never_falls_back(tmp_path, monkeypatch):
    from sdv.collection.bev_adapter import validate_global_maps, resolved_map_root
    root = _global_png_catalog(tmp_path/'frozen-job-assets')
    settings = normalize_bev_settings({'map_root':'wrong-path'})
    monkeypatch.setenv('SDV_BEV_MAP_ROOT',str(root))
    assert resolved_map_root(settings) == root
    validate_global_maps(['Town04'],settings)
    with pytest.raises(ValueError,match='Town05 / 5 PPM'):
        validate_global_maps(['Town05'],settings)
    # Explicit old modes do not require PNG assets.
    validate_global_maps(['Town99'],normalize_bev_settings({'mode':'height_aware'}))


def test_collection_preflight_reports_missing_town_or_resolution(tmp_path):
    pytest.importorskip('fastapi')
    from studio.api import Runtime
    from studio.config import Config
    root = _global_png_catalog(tmp_path/'assets/maps')
    config = Config(tmp_path)
    config.save_settings({'bev_generator':{'map_root':'assets/maps'},
                          'driving': {'agent': 'existing'}})
    runtime = Runtime.__new__(Runtime)
    runtime.config = config
    runtime.jobs = NS(active=lambda:None)
    dependencies = ['carla','numpy','yaml','cv2','scipy','shapely','torch','pygame','PIL','networkx']
    runtime.capabilities = lambda **kwargs: {'dependencies':{key:{'available':True} for key in dependencies}}
    runtime.carla_state = lambda: {'connected':True,'maps':['Town04','Town05']}
    sensors = dict(config.sensors(),towns=['Town04'])
    assert runtime.preflight('collection',override=sensors)['ready'] is True
    missing = runtime.preflight('collection',override=dict(sensors,towns=['Town05']))
    assert not missing['ready'] and any('Town05 / 5 PPM' in value for value in missing['blockers'])
    config.save_settings({'bev_generator':{'global_ppm':10}})
    unavailable = runtime.preflight('collection',override=sensors)
    assert not unavailable['ready'] and any('Town04 / 10 PPM' in value for value in unavailable['blockers'])
