"""HTTP contracts for recorded sequence review and immutable task settings.

The three tiny frames below are explicit, controlled annotations used to check
alignment and security. Real uploaded collection verification is recorded
separately in docs/pipeline-validation.json; these are not CARLA observations.
"""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from studio.api import create_app
from test_studio_jobs import namespace_psutil_adapter

HEADERS = {'X-SDV-Request': '1'}


def image(path, color, size=(32, 24)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(str(path))


def record(scene, key, rgb, boxes=None):
    image(scene/'front_cam_rgb'/(key+'.png'), rgb)
    image(scene/'front_cam_seg'/(key+'.png'), (128, 64, 128))
    depth=scene/'front_cam_depth'/(key+'.png')
    depth.parent.mkdir(parents=True,exist_ok=True)
    Image.new('L',(32,24),10).save(str(depth))
    image(scene/'das'/(key+'.png'), (255, 255, 255), (16, 16))
    image(scene/'lane'/(key+'.png'), (0, 0, 0), (16, 16))
    info = scene/'info'/(key+'.json')
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(json.dumps({'current_speed':float(int(key.split('_')[0])),
        'ego_vehicle_loc':{'x':float(int(key.split('_')[0])), 'y':2.0, 'z':0.0},
        'ego_vehicle_rot':{'pitch':0.0, 'yaw':0.0, 'roll':0.0},
        'current_control':{'throttle':0.5, 'steer':0.25, 'brake':0.0},
        'traffic_light':False, 'collision':False}))
    annotation=scene/'front_cam_bbs'/(key+'.npy')
    annotation.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(annotation), np.asarray(boxes or [], dtype=np.float32))


@pytest.fixture
def sequence_app(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    monkeypatch.setattr(app.state.runtime, 'capabilities', lambda *args, **kwargs: {'dependencies':{}, 'cuda_available':False})
    app.state.jobs.set_training_gate(lambda job:{'ready':False, 'blockers':['Explicit QA gate']})
    scene=tmp_path/'data/collections/run_a/Town01/scene_a'
    scene.mkdir(parents=True)
    app.state.config.yaml(scene/'sensor_config.yaml', app.state.config.sensors())
    (scene/'collection_manifest.json').write_text(json.dumps({'format':'sdv-native-masks-v1','fps':10,'save_every_ticks':2,'town':'Town01','scene':'scene_a'}))
    # Creation order differs from numeric frame order deliberately.
    record(scene, '10_a', (100, 1, 1))
    record(scene, '2_a', (20, 2, 2))
    record(scene, '11_a', (110, 3, 3))
    second=tmp_path/'data/collections/run_b/Town01/scene_b'
    second.mkdir(parents=True)
    app.state.config.yaml(second/'sensor_config.yaml', app.state.config.sensors())
    record(second, '2_a', (220, 22, 22))
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        rows=client.get('/api/datasets').json()['datasets']
        ids={row['name']:row['id'] for row in rows}
        yield app, client, scene, ids


def png(response):
    assert response.status_code==200,response.text[:500]
    assert response.headers['content-type'].startswith('image/png')
    return Image.open(io.BytesIO(response.content)).convert('RGB')


def test_sequence_count_numeric_order_and_sessions_do_not_mix(sequence_app):
    app, client, scene, ids=sequence_app
    first=client.get('/api/datasets/'+ids['scene_a']+'/sequence')
    assert first.status_code==200,first.text
    payload=first.json()
    assert payload['total']==3
    assert payload['id']==ids['scene_a']
    assert client.get('/api/datasets/'+ids['scene_b']+'/sequence').json()['total']==1
    for index,(key,color) in enumerate((('2_a',(20,2,2)),('10_a',(100,1,1)),('11_a',(110,3,3)))):
        response=client.get('/api/datasets/'+ids['scene_a']+'/frames/'+str(index))
        assert response.status_code==200,response.text
        frame=response.json()
        assert frame['frame_key']==key
        assert frame['index']==index
        assert frame['scene']['ego']['speed_kmh']==float(key.split('_')[0])
        actual=png(client.get('/api/datasets/'+ids['scene_a']+'/frame',params={'index':index,'channel':'rgb'}))
        assert actual.getpixel((10,10))==color
    other=png(client.get('/api/datasets/'+ids['scene_b']+'/frame',params={'index':0,'channel':'rgb'}))
    assert other.getpixel((10,10))==(220,22,22)


def test_legacy_metadata_remains_unknown_and_empty_boxes_are_valid(sequence_app):
    app,client,scene,ids=sequence_app
    frame=client.get('/api/datasets/'+ids['scene_a']+'/frames/0').json()
    assert frame['timestamp'] is None
    assert frame['boxes']==[]
    assert frame['scene']['road'].get('road_id') is None
    assert frame['scene']['traffic'].get('traffic_light_state') is None
    assert frame['scene']['ego']['steering']==.25
    base='/api/datasets/'+ids['scene_a']+'/frame'
    rgb=png(client.get(base,params={'index':0,'channel':'rgb'}))
    od=png(client.get(base,params={'index':0,'channel':'od'}))
    assert np.array_equal(np.asarray(rgb),np.asarray(od))


@pytest.mark.parametrize('channel',['rgb','seg','depth','od','bev'])
def test_actual_channel_images_are_decodable_and_aligned(sequence_app,channel):
    app,client,scene,ids=sequence_app
    result=png(client.get('/api/datasets/'+ids['scene_a']+'/frame',params={'index':1,'channel':channel}))
    assert result.width>0 and result.height>0
    if channel!='bev':
        assert result.size==(32,24)


@pytest.mark.parametrize('params',[
    {'index':-1,'channel':'rgb'},{'index':3,'channel':'rgb'},
    {'index':0,'channel':'../info'},{'index':0,'channel':'rgb','camera':'../private'},
    {'index':0,'channel':'rgb','camera':'/tmp/private'},
    {'index':0,'channel':'od','classes':'__import__("os")'},
    {'index':0,'channel':'od','colors':'not-json'},
    {'index':0,'channel':'depth','depth_min':'NaN'},
    {'index':0,'channel':'depth','depth_max':'Infinity'},
])
def test_frame_query_rejects_unsafe_or_invalid_input(sequence_app,params):
    app,client,scene,ids=sequence_app
    response=client.get('/api/datasets/'+ids['scene_a']+'/frame',params=params)
    assert response.status_code in (400,404,422),(params,response.status_code,response.text[:100])


def test_frame_missing_channel_does_not_fall_back_to_another_frame(sequence_app):
    app,client,scene,ids=sequence_app
    (scene/'front_cam_seg/10_a.png').unlink()
    app.state.catalog.invalidate()
    response=client.get('/api/datasets/'+ids['scene_a']+'/frame',params={'index':1,'channel':'seg'})
    assert response.status_code==404,response.text
    frame=client.get('/api/datasets/'+ids['scene_a']+'/frames/1').json()
    assert frame['frame_key']=='10_a'
    assert any('seg' in str(item) for item in frame['missing'])
    assert png(client.get('/api/datasets/'+ids['scene_a']+'/frame',params={'index':0,'channel':'seg'})).size==(32,24)


def test_recorded_channel_symlink_escape_is_rejected(sequence_app,tmp_path):
    app,client,scene,ids=sequence_app
    external=tmp_path/'secret.png'
    image(external,(1,2,3))
    target=scene/'front_cam_rgb/10_a.png'
    target.unlink();target.symlink_to(external)
    response=client.get('/api/datasets/'+ids['scene_a']+'/frame',params={'index':1,'channel':'rgb'})
    assert response.status_code in (400,404,422),response.text


def test_numpy_object_annotation_cannot_execute_pickle(sequence_app,tmp_path):
    app,client,scene,ids=sequence_app
    marker=tmp_path/'pickle-executed'
    class Unsafe:
        def __reduce__(self):
            return (eval, ("__import__('pathlib').Path(%r).write_text('unsafe')"%str(marker),))
    np.save(str(scene/'front_cam_bbs/10_a.npy'), np.asarray([Unsafe()],dtype=object))
    response=client.get('/api/datasets/'+ids['scene_a']+'/frames/1')
    assert response.status_code in (200,400,422),response.text
    assert not marker.exists()
    if response.status_code==200:
        assert response.json().get('warnings') or response.json().get('errors')


def test_unknown_dataset_and_oversized_frame_indices_are_rejected(sequence_app):
    app,client,scene,ids=sequence_app
    for route in ('/sequence','/frames/0','/frame?index=0&channel=rgb','/quality'):
        assert client.get('/api/datasets/does-not-exist'+route).status_code==404
    assert client.get('/api/datasets/'+ids['scene_a']+'/frames/999999999999').status_code in (400,404,422)


def test_advanced_training_settings_are_persisted_in_each_queue_snapshot(sequence_app,monkeypatch):
    app,client,scene,ids=sequence_app
    monkeypatch.setattr(app.state.runtime,'preflight',lambda *args,**kwargs:{'ready':True,'blockers':[]})
    first=app.state.config.training_defaults()
    first['task_settings']={
        'segmentation':{'mode':'carla','class_ids':[1,3,14],'excluded':'ignore'},
        'depth':{'min_m':1,'max_m':80,'out_of_range':'mask','visual_min_m':0,'visual_max_m':100,'colormap':'viridis'},
        'detection':{'class_ids':[0]},
        'bev':{'layers':['das','lane'],'visual_layers':['das','lane','route']},
    }
    response=client.post('/api/training/queue',json={'name':'Immutable task settings','config':first},headers=HEADERS)
    assert response.status_code==200,response.text
    queued=response.json(); original=copy.deepcopy(queued['config']['task_settings'])
    second=copy.deepcopy(first)
    second['task_settings']['segmentation'].update(class_ids=[1,24],excluded='background')
    second['task_settings']['depth'].update(min_m=2,max_m=50,colormap='turbo')
    result=client.put('/api/config/training',json=second,headers=HEADERS)
    assert result.status_code==200,result.text
    assert client.get('/api/config/training').json()['task_settings']['depth']['max_m']==50
    durable=client.get('/api/jobs/'+queued['id']).json()
    assert durable['config']['task_settings']==original
    saved=Path(durable['run_dir'])/'config/train_config.yaml'
    import yaml
    assert yaml.safe_load(saved.read_text())['task_settings']==original


@pytest.mark.parametrize('task,field,value',[
    ('segmentation','class_ids',[29]),('segmentation','class_ids',[1,1]),
    ('segmentation','excluded','invented'),('depth','min_m',-1),
    ('depth','max_m',0),('depth','invalid','fill_mean'),('depth','colormap','made-up'),
    ('detection','class_ids',[77]),('bev','layers',['route']),
])
def test_advanced_training_invalid_values_do_not_overwrite_settings(sequence_app,task,field,value):
    app,client,scene,ids=sequence_app
    original=client.get('/api/config/training').json()
    invalid=copy.deepcopy(original)
    invalid.setdefault('task_settings',{})
    invalid['task_settings'].setdefault(task,{})[field]=value
    response=client.put('/api/config/training',json=invalid,headers=HEADERS)
    assert response.status_code in (400,422),response.text
    assert client.get('/api/config/training').json()==original


def test_planning_export_requires_same_origin_write_header(sequence_app):
    app,client,scene,ids=sequence_app
    url='/api/datasets/'+ids['scene_a']+'/planning'
    assert client.post(url,json={'horizon_frames':2,'stride':1}).status_code==403
    assert client.post(url,json={'horizon_frames':2,'stride':1},headers=dict(HEADERS,Origin='https://outside.example')).status_code==403
    for body in ({'horizon_frames':-1},{'stride':0},{'horizon_frames':True}):
        assert client.post(url,json=body,headers=HEADERS).status_code in (400,422)


def test_positive_box_overlay_coordinate_color_filter_and_toggle(sequence_app):
    # Explicit controlled positive annotation: uploaded collection has zero boxes.
    app,client,scene,ids=sequence_app
    np.save(str(scene/'front_cam_bbs/10_a.npy'),np.asarray([[3,6,26,20,0]],dtype=np.float32))
    base='/api/datasets/'+ids['scene_a']
    frame=client.get(base+'/frames/1').json()
    box=frame['boxes'][0]
    assert box['class_id']==0 and box['class_name']=='Vehicle'
    assert box['xyxy']==[3,6,26,20]
    assert box['object_id'] is None and box['type']=='2d'
    rendered=png(client.get(base+'/frame',params={'index':1,'channel':'od','colors':json.dumps({'0':'#00ff00'})}))
    assert rendered.getpixel((3,15))==(0,255,0)
    original=png(client.get(base+'/frame',params={'index':1,'channel':'rgb'}))
    for query in ({'show_boxes':'false'},{'classes':'1'},{'classes':''}):
        hidden=png(client.get(base+'/frame',params=dict(index=1,channel='od',**query)))
        assert np.array_equal(np.asarray(original),np.asarray(hidden))


def test_malformed_metadata_is_reported_without_crashing_other_frames(sequence_app):
    app,client,scene,ids=sequence_app
    target=scene/'info/10_a.json'
    value=json.loads(target.read_text());value.update(sdv_metadata=['invalid'],ego_vehicle_rot=['invalid'])
    target.write_text(json.dumps(value))
    result=client.get('/api/datasets/'+ids['scene_a']+'/frames/1')
    assert result.status_code==200,result.text
    assert result.json()['scene']['ego']['yaw_deg'] is None
    assert client.get('/api/datasets/'+ids['scene_a']+'/frames/0').json()['scene']['ego']['speed_kmh']==2.0


@pytest.mark.parametrize('camera',['front_cam','left_cam'])
def test_collection_scene_uses_committed_frame_not_newest_file(sequence_app,camera):
    app,client,scene,ids=sequence_app
    if camera!='front_cam':
        for folder in scene.glob('front_cam_*'):
            folder.rename(scene/folder.name.replace('front_cam',camera,1))
        app.state.catalog.invalidate()
    job=app.state.jobs._build_job('collection','Controlled collection receipt',None,'Completed')
    job['output_dir']=str(scene.parent.parent)
    job['last_committed_frame']={'info':str(scene/'info/10_a.json'),'sample':10,'town':'Town01'}
    job['last_event']={'event':'collection_finished'}
    with app.state.jobs.db() as db:
        db.execute('INSERT INTO jobs(id,status,created_at,payload) VALUES(?,?,?,?)',(job['id'],job['status'],job['created_at'],json.dumps(job)))
        app.state.jobs._save(job,db)
    response=client.get('/api/jobs/'+job['id']+'/scene')
    assert response.status_code==200,response.text
    frame=response.json()
    assert frame['frame_key']=='10_a' and frame['index']==1
    assert frame['scene']['ego']['speed_kmh']==10.0
    assert frame['dataset_id']==ids['scene_a']
    assert frame['camera']==camera
    rendered=png(client.get('/api/datasets/'+frame['dataset_id']+'/frame',params={'index':frame['index'],'channel':'rgb','camera':frame['camera']}))
    assert rendered.getpixel((10,10))==(100,1,1)
    assert frame['frame_id'] is None and frame['timestamp'] is None
