"""Cross-boundary checks using explicitly controlled geometry, not CARLA runs.

The actual generator produces each label fixture. Tests then exercise the
quality report and planning exporter, rather than inventing numeric BEV masks.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('cv2')
from sdv.bev.cache import GlobalRasterCache
from sdv.bev.generator import HeightAwareBEVGenerator
from sdv.bev.io import load_frame
from sdv.bev.quality import validate_frame
from sdv.collection.bev_adapter import save_ground_truth
from studio.planning import export_sequence
from test_sequence_api import sequence_app


@pytest.fixture
def generated(tmp_path):
    surface = {'id': 'controlled-road', 'road_id': 7, 'section_id': 0,
        'lane_id': -1, 'polygon_xyz': [[-12,-6,2], [12,-6,2], [12,6,2], [-12,6,2]],
        'center_xyz': [0,0,2], 'neighbors': [], 'kind': 'driving', 'is_junction': False}
    cache = GlobalRasterCache.build(tmp_path/'cache', 'ControlledPipelineFixture',
        '<OpenDRIVE fixture="controlled-not-real-town" />', [surface])
    ego = {'actor_id': 10, 'object_class': 'vehicle', 'x': 0, 'y': 0, 'z': 2.1,
        'yaw': 0, 'ground_contact_world_xyz': [0,0,2], 'road_id': 7,
        'section_id': 0, 'lane_id': -1, 'bbox_min_z_m': 2.1, 'bbox_max_z_m': 4.1,
        'object_height_m': 2.0,
        'footprint_world_xyz': [[-2,-1,2.1], [2,-1,2.1], [2,1,2.1], [-2,1,2.1]]}
    return HeightAwareBEVGenerator(cache, {'width':64, 'height':64}).generate(
        ego, [ego], frame_id=123, timestamp=12.3)


def test_generated_saved_frame_satisfies_shared_quality_contract(generated, tmp_path):
    receipt = save_ground_truth(tmp_path/'scene', '00001_a', generated)
    restored = load_frame(tmp_path/'scene'/receipt['path'], receipt['geometry'])
    assert restored['ego'].any() and restored['validity'].any()
    assert validate_frame(restored, receipt, frame_id=123, timestamp=12.3) == []
    assert receipt['online_input'] is False
    assert receipt['role'] == 'simulator_ground_truth'


@pytest.mark.parametrize('fault,code', [
    ('frame', 'bev_frame_alignment'), ('timestamp', 'bev_timestamp_alignment'),
    ('schema', 'bev_schema'), ('provenance', 'bev_provenance'),
    ('semantic', 'bev_semantic_mapping'), ('height', 'bev_bbox_height'),
    ('ground', 'bev_actor_road_height'), ('surface', 'bev_surface_lookup'),
    ('actor', 'bev_actor_metadata'), ('dtype', 'bev_array_contract'),
])
def test_quality_detects_corrupted_generated_frame(generated, fault, code):
    result = copy.deepcopy(generated)
    arrays, meta = result['arrays'], result['metadata']
    row, col = np.argwhere(arrays['ego'] > 0)[0]
    if fault == 'frame': meta['frame_id'] += 1
    elif fault == 'timestamp': meta['timestamp'] += .1
    elif fault == 'schema': meta['schema_version'] = 2
    elif fault == 'provenance': meta['online_input'] = True
    elif fault == 'semantic': arrays['semantic'][row,col] = 0
    elif fault == 'height': arrays['object_height_m'][row,col] += 1
    elif fault == 'ground': arrays['object_ground_z_m'][row,col] += 2
    elif fault == 'surface': arrays['surface_id'][row,col] = 999999
    elif fault == 'actor': meta['actors'] = []
    elif fault == 'dtype': arrays['semantic'] = arrays['semantic'].astype(np.float32)
    actual = {item['code'] for item in validate_frame(arrays, meta, frame_id=123, timestamp=12.3)}
    assert code in actual


@pytest.mark.parametrize('field,value,code', [
    ('geometry', {'width':64}, 'bev_array_contract'),
    ('actors', ['wrong-type'], 'bev_actor_metadata'),
    ('surface_table', ['wrong-type'], 'bev_surface_lookup'),
])
def test_malformed_metadata_is_reported_without_python_exception(generated, field, value, code):
    metadata = copy.deepcopy(generated['metadata'])
    metadata[field] = value
    issues = validate_frame(generated['arrays'], metadata, frame_id=123, timestamp=12.3)
    assert code in {item['code'] for item in issues}


def test_invalid_zip_is_a_frame_error_not_a_server_error(sequence_app, generated):
    _app, client, scene, identifiers = sequence_app
    path = scene/'info/10_a.json'
    raw = json.loads(path.read_text())
    raw['bev_gt'] = save_ground_truth(scene, '10_a', generated)
    path.write_text(json.dumps(raw))
    (scene/'bev_gt/10_a.npz').write_bytes(b'not a ZIP archive')
    with pytest.raises(ValueError, match='Invalid BEV archive'):
        load_frame(scene/'bev_gt/10_a.npz')
    base = '/api/datasets/'+identifiers['scene_a']
    response = client.get(base+'/frame', params={'index':1, 'channel':'bev'})
    assert response.status_code == 422, response.text
    assert 'Invalid BEV archive' in response.text
    for endpoint in ('/frames/1', '/sequence'):
        response = client.get(base+endpoint)
        assert response.status_code == 200, response.text
        assert any('Invalid BEV archive' in str(v) for v in response.json()['warnings'])
    response = client.get(base+'/quality')
    assert response.status_code == 200, response.text
    errors = [item for item in response.json()['issues'] if item['code'] == 'invalid_bev_gt']
    assert len(errors) == 1 and errors[0]['severity'] == 'error'
    assert errors[0]['frame_key'] == '10_a'
    # An unrelated committed RGB image is still available.
    assert client.get(base+'/frame', params={'index':0, 'channel':'rgb'}).status_code == 200


def test_sequence_quality_matches_bev_to_recorded_world_frame(sequence_app, generated):
    _app, client, scene, identifiers = sequence_app
    path = scene/'info/10_a.json'
    raw = json.loads(path.read_text())
    raw['bev_gt'] = save_ground_truth(scene, '10_a', generated)
    raw['sdv_metadata'] = {'schema_version':3, 'synchronization':{
        'carla_frame':123, 'timestamp_s':12.3, 'aligned':True,
        'sensors':{'front_cam_rgb':{'frame':123, 'timestamp_s':12.3}}}}
    path.write_text(json.dumps(raw))
    route = '/api/datasets/'+identifiers['scene_a']+'/quality'
    response = client.get(route)
    assert response.status_code == 200, response.text
    assert not [issue for issue in response.json()['issues'] if issue['code'].startswith('bev_')]
    # Same file key is insufficient evidence: a stale raster's actual simulator
    # frame and timestamp must be rejected against the committed RGB record.
    raw['bev_gt']['frame_id'] = 122
    raw['bev_gt']['timestamp'] = 12.2
    path.write_text(json.dumps(raw))
    response = client.get(route)
    assert response.status_code == 200, response.text
    issues = {issue['code']:issue for issue in response.json()['issues'] if issue['code'].startswith('bev_')}
    for code in ('bev_frame_alignment', 'bev_timestamp_alignment'):
        assert issues[code]['severity'] == 'error'
        assert issues[code]['frame_key'] == '10_a'


def test_planning_export_keeps_all_gt_files_out_of_online_inputs(tmp_path, generated):
    channels = ('front_cam_rgb', 'rear_cam_rgb', 'lidar', 'info', 'bev_gt', 'bev',
        'das', 'lane', 'vehicle_mask', 'walker_mask', 'route', 'stopline',
        'light_topview', 'front_cam_seg_ids', 'front_cam_seg', 'front_cam_depth_m',
        'front_cam_depth', 'front_cam_bbs')
    frames, info = [], {}
    for sample in (1, 2, 3):
        key = '{}_a'.format(sample)
        frames.append({'index':sample-1, 'frame_key':key, 'frame_id':sample*2,
            'timestamp':sample*.2, 'files':{name:name+'/'+key for name in channels}})
        info[key] = {'ego_vehicle_loc':{'x':float(sample), 'y':0., 'z':2.},
            'ego_vehicle_rot':{'pitch':0., 'roll':0., 'yaw':0.},
            'bev_gt':copy.deepcopy(generated['metadata']), 'current_speed':10.,
            'current_control':{'steer':0., 'throttle':.2, 'brake':0.}}
    data = {'frames':frames, 'path':tmp_path, 'manifest':{'fps':10, 'save_every_ticks':2},
        'calibration':{'source':'controlled-fixture'}}
    service = SimpleNamespace(_inventory=lambda _identifier:data,
        _info=lambda _data, frame:info[frame['frame_key']])
    exported = export_sequence(service, 'controlled-sequence', horizon=1)
    assert exported['schema'] == 'sdv-planning-observations-v2'
    assert exported['record_count'] == 2
    for record in exported['records']:
        inputs, gt = record['inputs'], record['simulator_ground_truth']
        assert set(inputs['files']) == {'front_cam_rgb', 'rear_cam_rgb', 'lidar'}
        assert set(gt['files']) == set(channels)-set(inputs['files'])
        assert 'bev_gt' not in inputs and 'bev_metadata' not in inputs
        assert gt['bev_metadata']['online_input'] is False and gt['online_input'] is False
        future = record['targets']['future_observations'][0]
        assert future['frame_id'] > record['frame_id']
        assert future['timestamp_s'] > record['timestamp_s']
        assert 'offline ground truth' in future['source']
    json.dumps(exported, allow_nan=False)
