"""Availability checks inspect isolated file inventories, not model performance."""
from pathlib import Path

import pytest

from studio.training_options import training_options


FRAMES = ('00001', '00002')
THREE = {'front_cam': True, 'left_cam': True, 'right_cam': True}


def modality(scene, name, frames=FRAMES):
    folder = scene / name
    folder.mkdir(parents=True, exist_ok=True)
    extension = '.npy' if name == 'lidar' or name.endswith('_bbs') else '.png'
    for frame in frames:
        # Content decoding is deliberately left to the existing launch gate.
        (folder / (frame + extension)).write_bytes(b'inventory fixture')
    return folder


def dataset(root, cameras=('front_cam',), labels=(), bev=None, lidar=False):
    scenes = []
    for split in ('train', 'val'):
        scene = root / 'data' / split / ('scene_' + split)
        for camera in cameras:
            modality(scene, camera + '_rgb')
            for label in labels:
                modality(scene, camera + '_' + label)
        if bev:
            for name in (('bev',) if bev == 'legacy' else ('das', 'lane')) + ('vehicle_mask', 'walker_mask'):
                modality(scene, name)
        if lidar:
            modality(scene, 'lidar')
        scenes.append(scene)
    return scenes


def test_empty_splits_disable_all_options_with_bilingual_reasons(tmp_path):
    result = training_options(tmp_path, {})
    for group in ('sensors', 'tasks'):
        assert all(not entry['available'] for entry in result[group].values())
        assert all(entry['reason'] and entry['reason_en'] for entry in result[group].values())
    assert result['sensors']['front_cam']['reason_code'] == 'no_dataset_scenes'


def test_front_only_data_enables_only_present_sensor(tmp_path):
    dataset(tmp_path)
    result = training_options(tmp_path, {})
    assert result['sensors']['front_cam']['available']
    assert not result['sensors']['left_cam']['available']
    assert not result['sensors']['lidar']['available']
    assert not result['tasks']['Seg']['available']
    assert result['splits']['train']['samples'] == 2
    assert result['scope'] == 'file_presence_and_frame_alignment'


def test_complete_data_enables_all_selected_tasks_and_sensors(tmp_path):
    dataset(tmp_path, tuple(THREE), ('seg', 'depth', 'bbs'), 'native', True)
    result = training_options(tmp_path, {}, dict(THREE, lidar=True))
    assert all(result['tasks'][name]['available'] for name in ('BEV', 'Seg', 'Depth', 'od'))
    assert all(result['sensors'][name]['available'] for name in (*THREE, 'lidar'))
    assert not result['sensors']['rear_cam']['available']


def test_task_labels_follow_selected_cameras_and_validate_both_splits(tmp_path):
    scenes = dataset(tmp_path, ('front_cam', 'left_cam'), ('seg',))
    (scenes[1] / 'left_cam_seg' / '00002.png').unlink()
    front_only = training_options(tmp_path, {}, {'front_cam': True})
    both = training_options(tmp_path, {}, {'front_cam': True, 'left_cam': True})
    assert front_only['tasks']['Seg']['available']
    assert both['sensors']['left_cam']['available']
    assert not both['tasks']['Seg']['available']
    assert both['tasks']['Seg']['reason_code'] == 'frame_mismatch'
    assert both['tasks']['Seg']['details'][0]['split'] == 'val'


@pytest.mark.parametrize('change', ['missing', 'extra', 'same_count_wrong_ids'])
def test_rgb_frame_ids_must_align_in_every_scene(tmp_path, change):
    scenes = dataset(tmp_path, ('front_cam', 'left_cam'))
    folder = scenes[1] / 'left_cam_rgb'
    if change != 'extra':
        (folder / '00002.png').unlink()
    if change != 'missing':
        (folder / '99999.png').write_bytes(b'fixture')
    result = training_options(tmp_path, {}, {'front_cam': True, 'left_cam': True})
    assert result['sensors']['front_cam']['available']
    assert not result['sensors']['left_cam']['available']
    assert result['sensors']['left_cam']['reason_code'] == 'frame_mismatch'
    assert result['tasks']['Seg']['reason_code'] == 'selected_sensor_unavailable'


def test_one_incomplete_scene_is_not_hidden_by_complete_scenes(tmp_path):
    dataset(tmp_path, ('front_cam', 'left_cam'), ('seg',))
    additional = tmp_path / 'data' / 'train' / 'another_scene'
    modality(additional, 'front_cam_rgb')
    result = training_options(tmp_path, {}, {'front_cam': True, 'left_cam': True})
    assert result['splits']['train']['scenes'] == 2
    assert not result['sensors']['left_cam']['available']


def test_bev_accepts_legacy_and_native_scenes_without_palette_conversion(tmp_path):
    scenes = dataset(tmp_path, tuple(THREE), bev='native')
    for name in ('das', 'lane'):
        for file in (scenes[0] / name).iterdir():
            file.unlink()
        (scenes[0] / name).rmdir()
    modality(scenes[0], 'bev')
    result = training_options(tmp_path, {}, THREE)
    assert result['tasks']['BEV']['available']
    (scenes[1] / 'walker_mask' / '00002.png').unlink()
    assert not training_options(tmp_path, {}, THREE)['tasks']['BEV']['available']


def test_bev_accepts_single_camera_after_actual_decoder_contract_validation(tmp_path):
    dataset(tmp_path, bev='legacy')
    result = training_options(tmp_path, {})
    assert result['tasks']['BEV']['available']


def test_complete_native_masks_take_precedence_over_partial_legacy_palette(tmp_path):
    scenes = dataset(tmp_path, tuple(THREE), bev='native')
    modality(scenes[0], 'bev', ('00001',))
    assert training_options(tmp_path, {}, THREE)['tasks']['BEV']['available']


def test_zero_byte_file_and_later_write_are_detected_without_stale_cache(tmp_path):
    scenes = dataset(tmp_path, labels=('depth',))
    target = scenes[1] / 'front_cam_depth' / '00002.png'
    assert training_options(tmp_path, {})['tasks']['Depth']['available']
    target.write_bytes(b'')
    result = training_options(tmp_path, {})
    assert result['tasks']['Depth']['reason_code'] == 'empty_file'
    target.write_bytes(b'repaired inventory fixture')
    assert training_options(tmp_path, {})['tasks']['Depth']['available']


def test_missing_front_in_a_scene_disables_all_options(tmp_path):
    dataset(tmp_path)
    modality(tmp_path / 'data' / 'val' / 'orphan_scene', 'left_cam_rgb')
    result = training_options(tmp_path, {})
    assert not any(value['available'] for value in result['sensors'].values())
    assert result['sensors']['front_cam']['reason_code'] == 'front_frames_required'


def test_symbolic_link_modality_is_rejected_without_following_it(tmp_path):
    scenes = dataset(tmp_path)
    (scenes[0] / 'left_cam_rgb').symlink_to(scenes[0] / 'front_cam_rgb', target_is_directory=True)
    result = training_options(tmp_path, {})
    assert result['sensors']['left_cam']['reason_code'] == 'unsafe_path'


def test_split_escape_is_disabled_with_a_reason(tmp_path):
    result = training_options(tmp_path, {'train_data': '../outside'})
    assert result['sensors']['front_cam']['reason_code'] == 'invalid_dataset_path'


def test_extra_non_image_file_is_not_silently_ignored_by_positional_loader(tmp_path):
    scenes = dataset(tmp_path)
    (scenes[0] / 'front_cam_rgb' / 'notes.txt').write_text('extra loader input')
    result = training_options(tmp_path, {})
    assert result['sensors']['front_cam']['reason_code'] == 'unsupported_file'


def test_draft_paths_are_inspected_without_changing_saved_settings(tmp_path):
    dataset(tmp_path)
    settings = {'train_data': 'data/train', 'val_data': 'data/val'}
    assert training_options(tmp_path, settings)['sensors']['front_cam']['available']
    draft = dict(settings, val_data='data/missing')
    assert not training_options(tmp_path, draft)['sensors']['front_cam']['available']
    assert settings['val_data'] == 'data/val'


def test_front_camera_dependency_is_explained(tmp_path):
    dataset(tmp_path, ('front_cam', 'left_cam'), ('seg',))
    result = training_options(tmp_path, {}, {'left_cam': True})
    assert result['sensors']['left_cam']['available']
    assert result['tasks']['Seg']['reason_code'] == 'front_camera_required'


def test_nested_collection_and_direct_scene_paths_are_supported(tmp_path):
    scenes=dataset(tmp_path,labels=('seg',))
    for split,scene in zip(('train','val'),scenes):
        target=tmp_path/'nested'/split/'Town01'/'collection'/scene.name
        target.parent.mkdir(parents=True)
        scene.rename(target)
    settings={'train_data':'nested/train','val_data':'nested/val/Town01/collection/scene_val'}
    result=training_options(tmp_path,settings)
    assert result['tasks']['Seg']['available']
    assert result['splits']['train']['scenes']==1
    assert result['splits']['val']['samples']==2


def test_modern_raw_labels_and_partial_backfill_use_frame_keys(tmp_path):
    scenes=dataset(tmp_path)
    for scene in scenes:
        modality(scene,'front_cam_seg_ids',('00001',))
        modality(scene,'front_cam_seg',('00002',))
        folder=scene/'front_cam_depth_m';folder.mkdir()
        for frame in FRAMES:(folder/(frame+'.npy')).write_bytes(b'inventory fixture')
    advanced={'segmentation':{'mode':'carla','class_ids':[1]},'detection':{'class_ids':[]}}
    result=training_options(tmp_path,{},task_settings=advanced)
    assert result['tasks']['Seg']['available']
    assert result['tasks']['Depth']['available']
    assert not result['tasks']['od']['available']


def test_only_selected_bev_layers_are_required(tmp_path):
    scenes=dataset(tmp_path,tuple(THREE))
    for scene in scenes:modality(scene,'das')
    result=training_options(tmp_path,{},THREE,{'bev':{'layers':['das']}})
    assert result['tasks']['BEV']['available']
    assert not training_options(tmp_path,{},THREE)['tasks']['BEV']['available']
