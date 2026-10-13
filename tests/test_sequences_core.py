import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from studio.catalog import Catalog
from studio.sequences import SequenceCatalog
from studio.dataset_quality import validate_sequence
from studio.planning import export_sequence, world_to_ego


@pytest.fixture
def sequence(tmp_path):
    scene = tmp_path / 'collections' / 'Town01' / 'scene_test'
    for folder in ('front_cam_rgb','front_cam_seg','front_cam_bbs','das','lane','info'):
        (scene/folder).mkdir(parents=True)
    (scene/'collection_manifest.json').write_text(json.dumps({'format':'sdv-native-masks-v1','scene':'scene_test','town':'Town01','save_every_ticks':2}))
    (scene/'sensor_config.yaml').write_text(yaml.safe_dump({'sensors':{'front_cam':{'width':4,'height':3}}}))
    for number in (0,1,2,3):
        name = '{:05d}_a'.format(number)
        Image.new('RGB',(4,3),(100,110,120)).save(scene/'front_cam_rgb'/(name+'.png'))
        Image.new('RGB',(4,3),(128,64,128)).save(scene/'front_cam_seg'/(name+'.png'))
        Image.new('L',(4,3),255).save(scene/'das'/(name+'.png'))
        Image.new('L',(4,3),0).save(scene/'lane'/(name+'.png'))
        np.save(str(scene/'front_cam_bbs'/(name+'.npy')),np.empty((0,),dtype=np.float32))
        info = {'ego_vehicle_loc':{'x':number,'y':0,'z':0},'ego_vehicle_rot':{'pitch':0,'yaw':0,'roll':0},
                'current_speed':18,'current_control':{'throttle':.5,'steer':0,'brake':0},
                'sdv_metadata':{'synchronization':{'carla_frame':100+2*number,'timestamp_s':number*.2,'aligned':True,
                    'sensors':{'front_cam_rgb':{'frame':100+2*number,'timestamp_s':number*.2}}}}}
        (scene/'info'/(name+'.json')).write_text(json.dumps(info))
    catalog=Catalog(tmp_path,{'data_root':'collections'})
    identifier=catalog.datasets()[0]['id']
    return SequenceCatalog(catalog),identifier,scene


def test_missing_frame_channel_never_substitutes_neighbor(sequence):
    service,did,scene=sequence
    (scene/'front_cam_seg'/'00001_a.png').unlink()
    index=service.sequence(did)
    assert index['total']==4
    assert index['frames'][1]['missing']==['front_cam_seg']
    assert 'seg' not in service.frame(did,1)['image_urls']
    with pytest.raises(FileNotFoundError): service.render(did,1,'seg')


def test_empty_boxes_are_known_zero_and_not_missing(sequence):
    service,did,_=sequence
    frame=service.frame(did,0)
    assert frame['boxes']==[] and 'od' in frame['image_urls']
    assert service.render(did,0,'od')==service.render(did,0,'rgb')
    assert frame['scene']['traffic']['signal_state'] is None


def test_bev_layer_toggle_and_empty_selection(sequence):
    service,did,_=sequence
    def pixel(layers): return Image.open(io.BytesIO(service.render(did,0,'bev',layers=layers))).getpixel((0,0))
    assert pixel(['das'])==(51,65,85)
    assert pixel([])==(10,18,31)
    assert pixel(['lane'])==(10,18,31)


def test_quality_distinguishes_simulator_stride_from_missing_samples(sequence):
    service,did,_=sequence
    report=validate_sequence(service,did)
    codes=[item['code'] for item in report['issues']]
    assert 'native_frame_interval' not in codes
    assert 'sample_id_gap' not in codes
    assert report['summary']['semantic_classes'][0]['name']=='Road'
    assert report['summary']['semantic_classes'][0]['pixels']==48
    assert report['counts']['error']==0


def test_future_targets_observed_not_planner_and_no_cross_gap(sequence):
    service,did,scene=sequence
    result=export_sequence(service,did,horizon=2)
    target=result['records'][0]['targets']['future_observations'][0]
    assert target['position_ego_m']==[1,0,0]
    assert target['offset_s']==pytest.approx(.2)
    assert result['records'][0]['targets']['complete_horizon'] is True
    assert result['record_count']==3
    for folder in scene.iterdir():
        if folder.is_dir():
            for path in folder.glob('00001_a.*'): path.unlink()
    service._cache.clear()
    result=export_sequence(service,did,horizon=2)
    assert result['omitted'][0]['frame_key']=='00000_a'
    assert 'Missing' in result['omitted'][0]['reason']


def test_future_ego_rotation_full_carla_matrix():
    assert world_to_ego([0,2,0],[0,0,0],{'pitch':0,'yaw':90,'roll':0})==pytest.approx([2,0,0])
    assert world_to_ego([0,0,2],[0,0,0],{'pitch':90,'yaw':0,'roll':0})==pytest.approx([2,0,0])


def test_malformed_metadata_reported_per_frame(sequence):
    service,did,scene=sequence
    (scene/'info'/'00000_a.json').write_text(json.dumps({'sdv_metadata':[], 'ego_vehicle_rot':[1,2]}))
    # A nonempty malformed metadata type is reported, while optional empty
    # legacy values remain unavailable and never become invented measurements.
    (scene/'info'/'00001_a.json').write_text(json.dumps({'sdv_metadata':[1], 'ego_vehicle_rot':[1,2]}))
    assert service.frame(did,0)['scene']['ego']['yaw_deg'] is None
    assert service.frame(did,1)['warnings']
    assert any(i['code']=='invalid_metadata' for i in validate_sequence(service,did)['issues'])


def test_metadata_cache_survives_inventory_refresh_and_tracks_file_changes(sequence, monkeypatch):
    service,did,scene=sequence
    import studio.sequences as module
    original, reads = module.read_json, []
    def observed(path, *args, **kwargs):
        if path.parent.name == 'info': reads.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module, 'read_json', observed)
    service.sequence(did)
    assert len(reads)==4
    service._cache.clear()
    service.sequence(did)
    assert len(reads)==4
    path=scene/'info'/'00001_a.json'
    value=json.loads(path.read_text()); value['current_speed']=99.25
    path.write_text(json.dumps(value))
    service._cache.clear()
    assert service.frame(did,1)['scene']['ego']['speed_kmh']==99.25
    assert len(reads)==5
    assert service._info_bytes<=8*1024*1024
    # Simulate JSON eviction while retaining the compact sequence index. A
    # normal live poll should not parse the complete history again.
    service._info_cache.clear(); service._info_bytes=0
    service._cache[did]=(0,service._cache[did][1])
    service.sequence(did)
    assert len(reads)==5
