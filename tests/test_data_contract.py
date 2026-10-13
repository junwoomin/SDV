"""Label contracts test actual IDs, invalid pixels and depth provenance."""
import json
import pytest
from sdv.data_contract import (normalize_task_settings, new_task_settings,
    segmentation_spec, detection_spec, semantic_records, SEMANTIC_COLORS,
    depth_encoding_for_scene, decode_segmentation, remap_segmentation,
    prepare_depth, decode_depth, IGNORE_INDEX)


def test_old_config_keeps_six_channel_order_and_new_has_only_discovered_classes():
    assert [x['name'] for x in segmentation_spec()] == ['Pedestrians','RoadLines','Roads','Sidewalks','Vehicles','Other']
    settings = new_task_settings([1, 7, 14], [])
    assert [x['id'] for x in segmentation_spec(settings)] == [1, 7, 14]
    assert detection_spec(settings) == []


def test_unknown_fields_and_invalid_numeric_depth_rejected():
    for value in ({'unknown': {}}, {'depth': {'max_m': float('nan')}},
                  {'depth': {'min_m': 80, 'max_m': 1}},
                  {'segmentation': {'class_ids': [30]}},
                  {'detection': {'class_ids': [2]}}, {'bev': {'layers': ['road_boundary']}}):
        with pytest.raises(ValueError): normalize_task_settings(value)


def test_source_provenance_prevents_guessing_depth_scale(tmp_path):
    assert depth_encoding_for_scene(tmp_path) == 'legacy_100m_u8'
    (tmp_path/'collection_manifest.json').write_text(json.dumps({'format':'sdv-native-masks-v1'}))
    assert depth_encoding_for_scene(tmp_path) == 'meters_u8'
    assert depth_encoding_for_scene(tmp_path, path='depth.npy') == 'meters_f32'
    assert depth_encoding_for_scene(tmp_path, requested='legacy_100m_u8') == 'legacy_100m_u8'


def test_semantic_schema_uses_actual_carla_ids_and_rgb():
    records = {x['id']:x for x in semantic_records([1,7,12,14,24])}
    assert records[1]['name'] == 'Road'
    assert records[7]['name'] == 'Traffic Light'
    assert records[14]['color'] == [0,0,142]
    assert records[24]['name'] == 'Road Line'


def test_exact_palette_and_ids_roundtrip_without_invented_nearest_color():
    np=pytest.importorskip('numpy')
    pixels=np.array([SEMANTIC_COLORS],dtype=np.uint8)
    assert decode_segmentation(pixels).tolist() == [list(range(29))]
    assert decode_segmentation(np.array([[0,14,28,29,255]])).tolist()==[[0,14,28,255,255]]
    assert decode_segmentation(np.array([[[0,0,141]]],dtype=np.uint8))[0,0]==IGNORE_INDEX


def test_exclusions_and_unknown_pixels_do_not_change_output_contract():
    np=pytest.importorskip('numpy')
    settings=new_task_settings([1,14], [1])
    labels=np.array([[0,1,14,255]],dtype=np.uint8)
    assert remap_segmentation(labels,settings).tolist()==[[255,0,1,255]]
    settings['segmentation']['excluded']='background'
    assert [x['id'] for x in segmentation_spec(settings)]==[0,1,14]
    assert remap_segmentation(labels,settings).tolist()==[[0,1,2,255]]
    assert detection_spec(settings)[0]['name']=='Pedestrian'
    assert detection_spec(settings)[0]['output_index']==0


def test_depth_units_saturation_mask_clip_and_visual_range_independent():
    np=pytest.importorskip('numpy')
    original=np.array([[0,1,50,254,255]],dtype=np.uint8)
    values,valid=decode_depth(original,'meters_u8')
    assert values.tolist()==[[0.,1.,50.,254.,255.]]
    assert valid.tolist()==[[False,True,True,True,False]]
    settings={'depth':{'encoding':'meters_u8','min_m':2,'max_m':80,'out_of_range':'clip','visual_min_m':10,'visual_max_m':40}}
    clipped,valid=prepare_depth(original,settings)
    assert clipped.tolist()==[[0.,2.,50.,80.,255.]]
    assert valid.tolist()==[[False,True,True,True,False]]
    settings['depth']['out_of_range']='mask'
    masked,valid=prepare_depth(original,settings)
    assert np.array_equal(masked,values)
    assert valid.tolist()==[[False,False,True,False,False]]
