"""HTTP/UI data contracts using explicitly controlled PNG map fixtures."""
import hashlib
import json

import numpy as np
from PIL import Image

from sdv.bev.global_map import CATALOG_VERSION
from sdv.bev.io import save_frame
from test_bev_preview import numeric_app
from test_sequence_api import sequence_app
from test_studio_jobs import namespace_psutil_adapter


def controlled_map(root):
    folder=root/'controlled/Town01'
    folder.mkdir(parents=True)
    Image.new('L',(32,24),255).save(folder/'DAS_BEV.png')
    Image.new('L',(32,24),0).save(folder/'all_lane_BEV.png')
    np.save(folder/'world_offset.npy',np.asarray([-10.,-20.]))
    files={}
    for path in folder.iterdir():
        row={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix=='.png':row.update(size=[32,24],mode='L')
        files[path.name]=row
    manifest={'format':CATALOG_VERSION,'variants':[{'town':'Town01','ppm':5,'meters_per_pixel':.2,
        'directory':'controlled/Town01','files':files,'width_px':32,'height_px':24,
        'world_offset_xy':[-10.,-20.],'source_signature':'controlled-test-source',
        'resolution_evidence':'Explicit controlled test geometry; not captured CARLA source'}]}
    variant=manifest['variants'][0]
    payload={key:value for key,value in variant.items() if key!='source_signature'}
    variant['source_signature']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    (root/'catalog.json').write_text(json.dumps(manifest))


def test_global_map_endpoint_reports_only_verified_native_variants(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    root=app.state.config.root/'assets/bev_maps'
    controlled_map(root)
    response=client.get('/api/bev/maps')
    assert response.status_code==200,response.text
    result=response.json()
    assert result['selected']['global_ppm']==5
    town=result['town_records'][0]
    assert town['town']=='Town01' and town['available_ppm']==[5]
    variant=town['variants'][0]
    assert variant['meters_per_pixel']==.2
    assert variant['static_layers']==['drivable','lane']
    assert variant['height_layers']==['low']
    assert variant['metric_elevation_available'] is False
    (root/'controlled/Town01/DAS_BEV.png').write_bytes(b'corrupted PNG')
    invalid=client.get('/api/bev/maps').json()['town_records'][0]
    assert invalid['available_ppm']==[]
    assert invalid['variants'][0]['errors']


def test_global_map_endpoint_rejects_paths_outside_project(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    for path in ('../../outside','/etc'):
        result=client.get('/api/bev/maps',params={'root':path})
        assert result.status_code==422,result.text


def test_source_images_do_not_imply_metric_road_elevation(numeric_app):
    app,client,scene,ids,arrays=numeric_app
    arrays['elevation_m'][:]=np.nan
    arrays['elevation_validity']=np.zeros_like(arrays['validity'])
    path=scene/'info/10_a.json'
    raw=json.loads(path.read_text());meta=raw['bev_gt']
    meta.update(schema_version=4,global_map={'town':'Town01','native_ppm':5,'source_path':'controlled/Town01',
        'selected_height_layer':'low','height_layers':['low'],'files':{'drivable':'DAS_BEV.png','lane':'all_lane_BEV.png'}})
    meta['geometry']['orientation']='world_x_up_world_y_right'
    save_frame(scene/'bev_gt/10_a.npz',arrays,metadata=meta)
    path.write_text(json.dumps(raw))
    base='/api/datasets/'+ids['scene_a']
    frame=client.get(base+'/frames/1').json()
    assert frame['bev_gt']['schema_version']==4
    views={row['id']:row['available'] for row in frame['bev_gt']['views']}
    assert views['bev'] and views['bev_object_height'] and not views['bev_height']
    assert 'bev_height' not in frame['image_urls']
    assert frame['scene']['bev']['native_ppm']==5
    assert frame['scene']['bev']['orientation']=='world_x_up_world_y_right'
    layers={row['id']:row['available'] for row in frame['bev_gt']['layers']}
    assert layers['drivable'] and layers['lane'] and not layers['sidewalk']
    assert client.get(base+'/frame',params={'index':1,'channel':'bev_height'}).status_code==404
    arrays['object_height_m'][:]=np.nan
    save_frame(scene/'bev_gt/10_a.npz',arrays,metadata=meta)
    frame=client.get(base+'/frames/1').json()
    assert not next(row['available'] for row in frame['bev_gt']['views'] if row['id']=='bev_object_height')


def test_source_stopline_is_independent_of_signal_state(numeric_app):
    from studio.bev_preview import layer_catalog, render_bev
    from sdv.bev.schema import SEMANTIC_CLASSES
    app,client,scene,ids,arrays=numeric_app
    arrays['static_stopline']=np.zeros_like(arrays['validity'])
    arrays['static_stopline'][1,1]=1
    row=next(v for v in layer_catalog(arrays) if v['id']=='static_stopline')
    assert row['available'] and 'state unknown' in row['label']
    assert render_bev(arrays, layers=['static_stopline']).getpixel((1,1))==(240,244,250)
    arrays['stop_red'][1,1]=1
    assert render_bev(arrays, layers=['stoplines']).getpixel((1,1))==tuple(SEMANTIC_CLASSES[9]['color'])
    assert render_bev(arrays, layers=[]).getpixel((1,1))==(10,18,31)
