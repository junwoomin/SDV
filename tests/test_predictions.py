"""Stable real-data probes and safe saved-image galleries (no CARLA required)."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from sdv.training.predictions import fixed_samples
from studio.predictions import prediction_frame, prediction_index


def source_dataset(root, count=9):
    folder = root / 'scene' / 'front_cam_rgb'
    folder.mkdir(parents=True)
    for index in range(count):
        (folder / ('{:05d}.png'.format(index))).write_bytes(('RGB actual file identity ' + str(index)).encode())
    return root


def test_fixed_four_identical_across_dataset_copies_and_loader_orders(tmp_path):
    root = source_dataset(tmp_path / 'val')
    a = fixed_samples(root, tmp_path)
    assert [item['index'] for item in a['samples']] == [0, 3, 5, 8]
    copied = tmp_path / 'copied'
    shutil.copytree(str(root), str(copied))
    b = fixed_samples(copied, tmp_path)
    assert a['dataset_fingerprint'] == b['dataset_fingerprint']
    assert a['samples'] == b['samples']
    paths = list(reversed(sorted(root.glob('*/front_cam_rgb/*'))))
    reverse = fixed_samples(root, tmp_path, paths)
    assert [item['sample_id'] for item in a['samples']] == [item['sample_id'] for item in reverse['samples']]
    assert [item['index'] for item in reverse['samples']] == [8, 5, 3, 0]
    manifest = tmp_path / '.runtime/studio/prediction_samples' / (a['dataset_fingerprint'] + '.json')
    assert json.loads(manifest.read_text())['count'] == 4


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4])
def test_small_dataset_does_not_repeat_or_invent_frames(tmp_path, count):
    result = fixed_samples(source_dataset(tmp_path / 'val', count), tmp_path)
    assert result['count'] == count
    assert len({sample['sample_id'] for sample in result['samples']}) == count
    assert bool(result['reason']) == (count < 4)


def test_data_label_change_changes_identity_independent_of_selected_tasks(tmp_path):
    root = source_dataset(tmp_path / 'val')
    before = fixed_samples(root, tmp_path)
    label = root / 'scene/front_cam_depth/00000.png'
    label.parent.mkdir()
    label.write_bytes(b'ground truth depth')
    added = fixed_samples(root, tmp_path)
    assert before['dataset_fingerprint'] != added['dataset_fingerprint']
    assert before['samples'] == added['samples']
    label.write_bytes(b'changed ground truth depth')
    after = fixed_samples(root, tmp_path)
    assert after['dataset_fingerprint'] != added['dataset_fingerprint']


def test_file_digest_cache_preserves_content_identity(tmp_path, monkeypatch):
    import sdv.training.predictions as module
    root = source_dataset(tmp_path / 'val')
    before = fixed_samples(root, tmp_path)
    monkeypatch.setattr(module, '_digest', lambda path: pytest.fail('Unchanged source was needlessly hashed'))
    assert fixed_samples(root, tmp_path)['dataset_fingerprint'] == before['dataset_fingerprint']


def test_dataset_escape_symlink_is_rejected(tmp_path):
    root = source_dataset(tmp_path / 'val')
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'outside')
    (root / 'scene/front_cam_rgb/outside.png').symlink_to(outside)
    with pytest.raises(ValueError, match='링크'):
        fixed_samples(root, tmp_path)


def saved_manifest(output, epochs=(1, 2)):
    entries = []
    for epoch in epochs:
        images = {}
        for key in ('input', 'gt', 'pred'):
            relative = 'predictions/val/epoch_{:04d}/{}_seg.png'.format(epoch, key)
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'PNG path fixture')
            images[key] = relative
        entries.append({'id': str(epoch), 'capture_id': epoch, 'split': 'val', 'epoch': epoch,
                        'global_step': epoch * 10, 'sample_id': 'fixed-a', 'task': 'Seg',
                        'images': images})
    blob = {'version': 1, 'entries': entries,
            'datasets': {'val': {'count': 1, 'samples': [{'sample_id': 'fixed-a'}],
                                 'dataset_fingerprint': 'hash', 'reason': 'Only one real sample'}}}
    (output / 'predictions/index.json').write_text(json.dumps(blob), encoding='utf-8')
    return blob


def test_gallery_job_root_prefix_pagination_filter_and_fixed_samples(tmp_path):
    output = tmp_path / 'job/output'
    saved_manifest(output)
    result = prediction_index(tmp_path / 'job', limit=1)
    assert result['total'] == 2 and result['entries'][0]['epoch'] == 2
    assert result['entries'][0]['images']['pred'].startswith('output/predictions/')
    assert result['fixed_samples'] == [{'sample_id': 'fixed-a'}]
    assert result['fixed_sample_count'] == 1 and result['reason']
    assert result['epochs'] == [1, 2] and result['tasks'] == ['Seg']
    assert prediction_index(output, epoch=1, task='Seg')['total'] == 1
    assert prediction_index(output, split='train')['total'] == 0
    assert prediction_index(output, sample_id='different')['total'] == 0
    assert prediction_index(output, offset=1)['entries'][0]['epoch'] == 1


def test_frame_only_serves_manifest_listed_images(tmp_path):
    saved_manifest(tmp_path)
    result = prediction_index(tmp_path)
    relative = result['entries'][0]['images']['pred']
    assert prediction_frame(tmp_path, relative).is_file()
    unknown = tmp_path / 'private.png'
    unknown.write_bytes(b'not listed')
    with pytest.raises(FileNotFoundError): prediction_frame(tmp_path, 'private.png')
    with pytest.raises(ValueError): prediction_frame(tmp_path, '../private.png')
    with pytest.raises(ValueError): prediction_frame(tmp_path, str(unknown))


def test_gallery_rejects_escaping_symlink_and_corrupt_manifest(tmp_path):
    blob = saved_manifest(tmp_path / 'output')
    external = tmp_path / 'outside.png'
    external.write_bytes(b'external')
    pred = tmp_path / 'output' / blob['entries'][0]['images']['pred']
    pred.unlink(); pred.symlink_to(external)
    with pytest.raises(ValueError): prediction_index(tmp_path / 'output')
    path = tmp_path / 'output/predictions/index.json'
    path.write_text('invalid json')
    with pytest.raises(ValueError, match='JSON'): prediction_index(tmp_path / 'output')


def test_metadata_file_size_limit(tmp_path, monkeypatch):
    import studio.predictions as module
    saved_manifest(tmp_path)
    monkeypatch.setattr(module, 'MAX_MANIFEST_BYTES', 10)
    with pytest.raises(ValueError, match='32 MB'): prediction_index(tmp_path)


def test_missing_images_do_not_become_placeholder_predictions(tmp_path):
    blob = saved_manifest(tmp_path)
    (tmp_path / blob['entries'][0]['images']['pred']).unlink()
    result = prediction_index(tmp_path)
    assert result['total'] == 2 and result['unavailable_entries'] == 1
    assert len(result['entries']) == 1 and result['entries'][0]['epoch'] == 2
    assert result['next_offset'] == 2 and not result['has_more']


def test_missing_page_images_still_advance_metadata_pagination(tmp_path):
    blob = saved_manifest(tmp_path, epochs=(1, 2, 3))
    (tmp_path / blob['entries'][2]['images']['pred']).unlink()
    first = prediction_index(tmp_path, limit=1)
    assert first['entries'] == [] and first['unavailable_entries'] == 1
    assert first['total'] == 3 and first['next_offset'] == 1 and first['has_more']
    second = prediction_index(tmp_path, offset=first['next_offset'], limit=1)
    assert second['entries'][0]['epoch'] == 2
    assert second['next_offset'] == 2 and second['has_more']
    third = prediction_index(tmp_path, offset=second['next_offset'], limit=1)
    assert third['entries'][0]['epoch'] == 1
    assert third['next_offset'] == 3 and not third['has_more']


def test_cached_gallery_still_rejects_changed_image_symlink(tmp_path):
    blob = saved_manifest(tmp_path / 'output')
    assert prediction_index(tmp_path / 'output')['total'] == 2
    external = tmp_path / 'external.png'
    external.write_bytes(b'outside')
    selected = tmp_path / 'output' / blob['entries'][1]['images']['pred']
    selected.unlink(); selected.symlink_to(external)
    with pytest.raises(ValueError): prediction_index(tmp_path / 'output')
    with pytest.raises(ValueError): prediction_frame(tmp_path / 'output', blob['entries'][1]['images']['pred'])


def test_gallery_and_frame_cache_follow_atomic_manifest_updates(tmp_path):
    blob = saved_manifest(tmp_path)
    old_image = blob['entries'][0]['images']['pred']
    assert prediction_index(tmp_path)['total'] == 2
    assert prediction_frame(tmp_path, old_image).exists()
    updated = saved_manifest(tmp_path, epochs=(2, 3))
    result = prediction_index(tmp_path)
    assert result['epochs'] == [2, 3] and result['entries'][0]['epoch'] == 3
    with pytest.raises(FileNotFoundError): prediction_frame(tmp_path, old_image)


def test_large_history_validates_only_requested_page_images(tmp_path, monkeypatch):
    import studio.predictions as module
    blob = saved_manifest(tmp_path, epochs=(1,))
    record = blob['entries'][0]
    blob['entries'] = [dict(record, id=str(index), capture_id=index) for index in range(10000)]
    (tmp_path / 'predictions/index.json').write_text(json.dumps(blob), encoding='utf-8')
    original, calls = module._listed_image, []
    def measured(root, relative):
        calls.append(relative)
        return original(root, relative)
    monkeypatch.setattr(module, '_listed_image', measured)
    result = prediction_index(tmp_path, limit=10)
    assert result['total'] == 10000 and len(result['entries']) == 10
    assert len(calls) == 30  # Input/GT/pred for only ten returned records.
    calls.clear()
    assert prediction_index(tmp_path, offset=100, limit=10)['total'] == 10000
    assert len(calls) == 30


def test_manifest_caches_are_bounded_and_returned_rows_do_not_mutate_cache(tmp_path):
    import studio.predictions as module
    for number in range(5):
        root = tmp_path / str(number)
        saved_manifest(root)
        rows = prediction_index(root)
        rows['entries'][0]['images']['pred'] = '../outside.png'
        assert prediction_index(root)['entries'][0]['images']['pred'].startswith('predictions/')
    assert len(module._MANIFEST_CACHE) <= module._CACHE_ITEMS
    assert len(module._ENTRY_CACHE) <= module._CACHE_ITEMS


@pytest.mark.parametrize('offset,limit', [(-1, 100), (0, 0), (0, 201)])
def test_gallery_bounds(offset, limit, tmp_path):
    with pytest.raises(ValueError): prediction_index(tmp_path, offset=offset, limit=limit)


def test_legacy_training_images_are_read_without_invented_epoch_or_identity(tmp_path):
    output = tmp_path / 'job/output'
    for kind in ('gt', 'pred'):
        folder = output / 'save_samples_seg' / kind
        folder.mkdir(parents=True)
        (folder / 'seg_20.png').write_bytes(b'actual saved image placeholder for path tests')
    result = prediction_index(tmp_path / 'job')
    entry = result['entries'][0]
    assert result['legacy'] and entry['legacy']
    assert entry['epoch'] is None and entry['global_step'] is None and entry['split'] == 'unknown'
    assert entry['dataset_fingerprint'] is None
    assert set(entry['images']) == {'pred', 'gt'}
    assert prediction_frame(tmp_path / 'job', entry['images']['pred']).is_file()


def _torch_fixture(tmp_path, count=6, cameras=1):
    torch = pytest.importorskip('torch', reason='Actual CPU prediction rendering requires research dependencies')
    np = pytest.importorskip('numpy')
    pytest.importorskip('PIL')
    pytest.importorskip('cv2')
    from PIL import Image
    root = tmp_path / 'val'
    paths = []
    for index in range(count):
        path = root / 'scene/front_cam_rgb' / ('{:05d}.png'.format(index))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((128, 128, 3), index * 20, dtype=np.uint8)).save(str(path))
        paths.append(str(path))
    class Dataset:
        data_root = root
        image_front = paths
        sensor_keys = ['front_cam'] if cameras == 1 else ['left_cam', 'front_cam', 'right_cam']
        def __getitem__(self, index):
            imgs = torch.ones(cameras, 3, 128, 128) * index / 10
            eye = torch.eye(3).repeat(cameras, 1, 1)
            zero = torch.zeros(cameras, 3)
            seg = torch.zeros(cameras, 6, 128, 128); seg[:, index % 6] = 1
            dep = torch.ones(cameras, 1, 128, 128) * (index + 1)
            bev = torch.ones(1, 200, 200) * (index % 2)
            return (imgs, imgs, bev, bev, bev, bev, None, None,
                    eye, zero, torch.eye(4).repeat(cameras, 1, 1), eye,
                    eye, zero, seg, dep, dep.bool(), torch.zeros(cameras, 0, 5))
    def collater(samples):
        return tuple(None if items[0] is None else torch.stack(items) for items in zip(*samples))
    return torch, np, Dataset(), collater


def test_real_render_state_rng_preservation_and_epoch_history(tmp_path):
    torch, np, dataset, collater = _torch_fixture(tmp_path)
    from sdv.training.predictions import PredictionRecorder
    import random
    class AnalyticModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            self.bn = torch.nn.BatchNorm2d(3)
        def forward(self, images, *args):
            batch, cameras, _, height, width = images.shape
            bev = torch.ones(batch, 1, 200, 200) * self.weight
            seg = images.new_zeros(batch, cameras, 6, height, width)
            seg[:, :, 1] = self.weight
            depth = images[:, :, :1] * self.weight + 10
            return (bev, bev, bev, bev, seg, depth, None, None)
    model = AnalyticModel().train(); model.bn.eval()
    recorder = PredictionRecorder(tmp_path / 'output', tmp_path, dataset)
    random.seed(81); np.random.seed(83); torch.manual_seed(87)
    states = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
    bn_before = model.bn.running_mean.clone()
    first = recorder.capture(model, dataset, collater, epoch=1, global_step=10)
    assert first['count'] == 24 and len(first['samples']) == 4
    assert model.training and not model.bn.training
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert torch.equal(model.bn.running_mean, bn_before)
    before = prediction_index(tmp_path / 'output')
    path = prediction_frame(tmp_path / 'output', before['entries'][0]['images']['pred'])
    bytes_before = path.read_bytes()
    model.weight.data.fill_(2)
    recorder.capture(model, dataset, collater, epoch=2, global_step=20)
    after = prediction_index(tmp_path / 'output')
    assert after['total'] == 48 and after['epochs'] == [1, 2]
    assert path.read_bytes() == bytes_before
    assert {entry['sample_id'] for entry in after['entries']} == {sample['sample_id'] for sample in first['samples']}
    assert all(set(entry['images']) == {'input', 'gt', 'pred'} for entry in after['entries'])


def test_real_render_failure_restores_all_modes_and_rng(tmp_path):
    torch, np, dataset, collater = _torch_fixture(tmp_path)
    from sdv.training.predictions import PredictionRecorder
    class FailedModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, *args): raise RuntimeError('actual inference failed')
    model = FailedModel().train()
    state = torch.get_rng_state().clone()
    recorder = PredictionRecorder(tmp_path / 'output', tmp_path, dataset)
    with pytest.raises(RuntimeError, match='inference failed'):
        recorder.capture(model, dataset, collater)
    assert model.training and torch.equal(torch.get_rng_state(), state)
    assert not recorder.index_path.exists()


def test_actual_sdv_model_seg_depth_od_predictions_render(tmp_path, monkeypatch):
    torch, _, dataset, collater = _torch_fixture(tmp_path, count=1)
    pytest.importorskip('torchvision')
    pytest.importorskip('efficientnet_pytorch')
    from sdv.training.modeling.model import models
    from sdv.training.predictions import PredictionRecorder
    config = {'width': 128, 'height': 128, 'lidar': False,
              'sensor_use': {'front_cam': True, 'lidar': False},
              'model': {'encoder': 'ResNet18', 'decoder': 'FPN', 'bev_decoder': None},
              'model_conf': {'BEV': False, 'Seg': True, 'Depth': True, 'od': True}}
    def forbidden(*args, **kwargs): pytest.fail('CPU fixture must not download weights')
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        model = models(config, pretrained=False).eval()
        recorder = PredictionRecorder(tmp_path / 'output', tmp_path, dataset)
        result = recorder.capture(model, dataset, collater, epoch=1)
        assert result['count'] == 3
        gallery = prediction_index(tmp_path / 'output')
        assert set(gallery['tasks']) == {'Seg', 'Depth', 'od'}
        from PIL import Image
        for entry in gallery['entries']:
            for relative in entry['images'].values():
                with Image.open(prediction_frame(tmp_path / 'output', relative)) as image:
                    assert image.size == (128, 128)
    finally:
        torch.set_num_threads(threads)


def test_actual_sdv_multicamera_bev_and_image_predictions_render(tmp_path):
    torch, _, dataset, collater = _torch_fixture(tmp_path, count=1, cameras=3)
    pytest.importorskip('torchvision')
    pytest.importorskip('efficientnet_pytorch')
    from sdv.training.modeling.model import models
    from sdv.training.predictions import PredictionRecorder
    config = {'width': 128, 'height': 128, 'lidar': False,
              'sensor_use': {'left_cam': True, 'front_cam': True, 'right_cam': True, 'lidar': False},
              'model': {'encoder': 'ResNet18', 'decoder': 'FPN', 'bev_decoder': 'UNet'},
              'model_conf': {'BEV': True, 'Seg': True, 'Depth': True, 'od': True}}
    threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        model = models(config, pretrained=False).eval()
        recorder = PredictionRecorder(tmp_path / 'output', tmp_path, dataset)
        result = recorder.capture(model, dataset, collater)
        assert result['count'] == 13  # four BEV maps and three image tasks × three cameras
        gallery = prediction_index(tmp_path / 'output')
        assert set(gallery['tasks']) == {'DAS', 'Lane', 'Vehicle', 'Walker', 'Seg', 'Depth', 'od'}
        from PIL import Image
        for entry in gallery['entries']:
            with Image.open(prediction_frame(tmp_path / 'output', entry['images']['pred'])) as image:
                assert image.size == ((200, 200) if entry['camera'] is None else (128, 128))
            if entry['camera'] is None:
                assert entry['images']['input'].endswith('/front_cam_input.png')
        assert {entry['camera'] for entry in gallery['entries'] if entry['task'] == 'Seg'} == set(dataset.sensor_keys)
    finally:
        torch.set_num_threads(threads)
