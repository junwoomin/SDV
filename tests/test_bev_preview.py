"""Controlled schema-v3 fixtures exercise real HTTP and numeric rendering paths.

These small annotations are test geometry, not captured CARLA observations.
"""
import json
import numpy as np
import pytest

from sdv.bev.schema import BINARY_CHANNELS, HEIGHT_CHANNELS, SEMANTIC_CLASSES
from sdv.bev.io import save_frame, semantic_from_channels
from studio.bev_preview import read_bev
from studio.task_catalog import task_catalog
from test_sequence_api import sequence_app, png, HEADERS


def numeric_record(scene, key='10_a'):
    shape=(64,64)
    arrays={name:np.zeros(shape,dtype=np.uint8) for name in BINARY_CHANNELS}
    arrays.update({name:np.full(shape,np.nan,dtype=np.float32) for name in HEIGHT_CHANNELS})
    arrays.update(actor_id=np.zeros(shape,dtype=np.int32),surface_id=np.full(shape,-1,dtype=np.int32))
    arrays['validity'][:40,:]=1
    arrays['occupancy_validity'][:40,:]=1
    arrays['drivable'][:40,:]=1
    arrays['road'][:40,:]=1
    arrays['elevation_m'][:40,:]=2.5
    arrays['elevation_m'][0,0]=7.5
    arrays['vehicle'][10:20,10:20]=1
    arrays['actor_id'][10:20,10:20]=42
    arrays['object_height_m'][10:20,10:20]=1.8
    arrays['object_ground_z_m'][10:20,10:20]=2.5
    arrays['object_min_z_m'][10:20,10:20]=2.6
    arrays['object_max_z_m'][10:20,10:20]=4.4
    arrays['route'][5:8,2:60]=1
    arrays['lane'][30:32,:]=1
    arrays['semantic']=semantic_from_channels(arrays)
    save_frame(scene/'bev_gt'/(key+'.npz'),arrays)
    path=scene/'info'/(key+'.json')
    raw=json.loads(path.read_text())
    raw['bev_gt']={'schema_version':3,'geometry':{'width':64,'height':64,'meters_per_pixel':.2,'anchor_x':.5,'anchor_y':.5},
        'ego_pose':{'x':3,'y':4,'z':2.8,'yaw':90},'ground_z_m':2.5,'map_layer_id':'surface_7',
        'road_id':17,'lane_id':-1,'provenance':'controlled_test_geometry',
        'actors':[{'actor_id':42,'object_class':'vehicle','ground_z_m':2.5,'min_z_m':2.6,'max_z_m':4.4,'height_m':1.8,'map_layer_id':'surface_7'}]}
    path.write_text(json.dumps(raw))
    return arrays


@pytest.fixture
def numeric_app(sequence_app):
    app,client,scene,ids=sequence_app
    arrays=numeric_record(scene)
    return app,client,scene,ids,arrays


def test_height_metadata_and_semantic_palette_are_from_same_record(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    base='/api/datasets/'+ids['scene_a']
    frame=client.get(base+'/frames/1').json()
    assert frame['frame_key']=='10_a'
    assert frame['bev_gt']['schema_version']==3
    assert frame['bev_gt']['height_ranges']['elevation_m']=={'minimum_m':2.5,'maximum_m':7.5,'valid_pixels':2560}
    assert frame['scene']['bev']['ground_z_m']==2.5
    assert frame['scene']['bev']['ego_z_m']==2.8
    assert frame['scene']['bev']['map_layer_id']=='surface_7'
    assert frame['scene']['bev']['actors'][0]['actor_id']==42
    image=np.asarray(png(client.get(base+'/frame',params={'index':1,'channel':'bev'})))
    palette=np.asarray([r['color'] for r in SEMANTIC_CLASSES],dtype=np.uint8)
    assert np.array_equal(image,palette[arrays['semantic']])
    # Rendering must not mutate the numeric source labels.
    assert np.array_equal(read_bev(scene/'bev_gt/10_a.npz')['semantic'],arrays['semantic'])


@pytest.mark.parametrize('channel',['bev_height','bev_object_height','bev_occupancy','bev_road','bev_route'])
def test_numeric_views_decode_and_missing_frame_never_substitutes(numeric_app,channel):
    app,client,scene,ids,arrays=numeric_app
    base='/api/datasets/'+ids['scene_a']+'/frame'
    image=png(client.get(base,params={'index':1,'channel':channel}))
    assert image.size==(64,64)
    missing=client.get(base,params={'index':0,'channel':channel})
    assert missing.status_code==404,missing.text
    legacy=client.get('/api/datasets/'+ids['scene_a']+'/frames/0').json()
    assert legacy['bev_gt'] is None
    assert legacy['scene']['bev']['ground_z_m'] is None
    assert legacy['scene']['bev']['map_layer_id'] is None
    assert channel not in legacy['image_urls']


def test_road_elevation_unknown_is_not_a_zero_height(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    base='/api/datasets/'+ids['scene_a']+'/frame'
    image=png(client.get(base,params={'index':1,'channel':'bev_height'}))
    assert image.getpixel((10,10))==(26,88,130)
    assert image.getpixel((0,0))==(251,191,89)
    assert image.getpixel((50,50))==(10,18,31)
    assert np.isnan(read_bev(scene/'bev_gt/10_a.npz')['elevation_m'][50,50])


def test_mask_layer_toggle_uses_numeric_channels(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    base='/api/datasets/'+ids['scene_a']+'/frame'
    selected=png(client.get(base,params={'index':1,'channel':'bev','layers':'vehicle'}))
    assert selected.getpixel((10,10))==tuple(SEMANTIC_CLASSES[10]['color'])
    assert selected.getpixel((30,30))==(10,18,31)
    empty=png(client.get(base,params={'index':1,'channel':'bev','layers':''}))
    assert len(empty.getcolors(100000))==1
    assert client.get(base,params={'index':1,'channel':'bev','layers':'invented'}).status_code==422


def test_schema_v3_rejects_shape_dtype_and_geometry_mismatch(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    arrays['vehicle']=arrays['vehicle'].astype(np.float32)
    np.savez_compressed(scene/'bev_gt/10_a.npz',**arrays)
    base='/api/datasets/'+ids['scene_a']
    response=client.get(base+'/frame',params={'index':1,'channel':'bev_height'})
    assert response.status_code==422,response.text
    frame=client.get(base+'/frames/1').json()
    assert frame['bev_gt'] is None
    assert any('dtype' in warning for warning in frame['warnings'])
    assert 'bev' not in frame['image_urls']


def test_task_catalog_reads_numeric_labels_and_actual_geometry(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    result=task_catalog(app.state.config.root,dict(app.state.config.settings(),train_data=str(scene),val_data=str(scene)))
    bev=result['bev']
    assert bev['schema_version']==3
    assert bev['numeric_frames']==2  # Explicitly requested for both splits.
    assert bev['recorded_geometries'][0]['width']==64
    assert bev['recorded_geometries'][0]['meters_per_pixel']==.2
    assert 'object_height_m' in bev['height_channels']
    assert next(row for row in bev['layers'] if row['id']=='vehicle')['available']
    assert {row['id'] for row in bev['semantic_classes']}==set(np.unique(arrays['semantic']).tolist())


def test_bev_collection_settings_round_trip(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    config=client.get('/api/settings').json()['bev_generator']
    config.update(width=128,height=256,anchor_y=.8,object_height=False,actor_classes=['vehicle'])
    response=client.put('/api/settings',json={'bev_generator':config},headers=HEADERS)
    assert response.status_code==200,response.text
    assert client.get('/api/settings').json()['bev_generator']==config
    config['layer_policy']='invented'
    assert client.put('/api/settings',json={'bev_generator':config},headers=HEADERS).status_code==422


def test_disabled_collection_channels_are_not_claimed_as_trainable(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    path=scene/'info/10_a.json'
    raw=json.loads(path.read_text())
    raw['bev_gt']['settings']={'semantic_layers':['drivable'],'actor_classes':['vehicle']}
    path.write_text(json.dumps(raw))
    # The archive still has schema-required zero masks, but the persisted
    # capture settings determine whether a task was actually collected.
    result=task_catalog(app.state.config.root,dict(app.state.config.settings(),train_data=str(scene),val_data=str(scene)))
    layers={v['id']:v for v in result['bev']['layers']}
    assert layers['walker']['available'] is False
    assert layers['vehicle']['available'] is True
    sequence=client.get('/api/datasets/'+ids['scene_a']+'/sequence').json()
    observed={v['id']:v['available'] for v in sequence['layers']}
    assert observed['pedestrian'] is False and observed['lane'] is False


def test_recorded_geometry_is_checked_before_rendering(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    path=scene/'info/10_a.json'
    raw=json.loads(path.read_text());raw['bev_gt']['geometry']['width']=128
    path.write_text(json.dumps(raw))
    response=client.get('/api/datasets/'+ids['scene_a']+'/frame',params={'index':1,'channel':'bev'})
    assert response.status_code==422,response.text
    assert 'geometry' in response.text


@pytest.mark.parametrize('field,value',[('settings',['wrong-type']),('geometry',{'width':64}),('geometry','not-a-dict')])
def test_malformed_imported_bev_metadata_returns_warning_and_422(numeric_app,field,value):
    app,client,scene,ids,arrays=numeric_app
    path=scene/'info/10_a.json'
    raw=json.loads(path.read_text());raw['bev_gt'][field]=value
    path.write_text(json.dumps(raw))
    base='/api/datasets/'+ids['scene_a']
    result=client.get(base+'/frames/1')
    assert result.status_code==200,result.text
    assert result.json()['bev_gt'] is None and result.json()['warnings']
    result=client.get(base+'/frame',params={'index':1,'channel':'bev'})
    assert result.status_code==422,result.text
