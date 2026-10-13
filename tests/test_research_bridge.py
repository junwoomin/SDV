"""CPU-only boundary tests; these do not claim CARLA/CUDA execution coverage."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]


def function_from_file(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_collater_absent_optional_modalities():
    torch = SimpleNamespace(stack=np.stack, from_numpy=lambda x: x, as_tensor=np.asarray,
                            ones=np.ones, float32=np.float32, bool=np.bool_)
    collater = function_from_file('sdv/training/experiment.py', 'collater', {'torch': torch, 'np': np})
    arr = np.zeros((1, 2, 2))
    sample = (arr, arr, None, None, None, None, None, None,
              arr, arr, arr, arr, arr, arr, None, None, None, [])
    result = collater([sample, sample])
    assert result[0].shape[0] == 2
    assert result[2] is None and result[6] is None and result[14] is None
    assert result[-1].shape == (2, 1, 0, 5)


def test_collater_six_column_annotations_remove_only_batch_index():
    torch = SimpleNamespace(stack=np.stack, from_numpy=lambda x: x, as_tensor=np.asarray,
                            ones=np.ones, float32=np.float32, bool=np.bool_)
    collater = function_from_file('sdv/training/experiment.py', 'collater', {'torch': torch, 'np': np})
    arr = np.zeros((1, 2, 2))
    annots = [np.array([[0, 10, 20, 30, 40, 1]], dtype=np.float32)]
    sample = (arr, arr, arr, arr, arr, arr, None, None,
              arr, arr, arr, arr, arr, arr, arr, arr, arr, annots)
    result = collater([sample])
    assert result[-1].tolist() == [[[[10, 20, 30, 40, 1]]]]


def test_legacy_metric_lists_align_and_keep_od_separate():
    tree = ast.parse((ROOT / 'sdv/training/experiment.py').read_text())
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'append' and isinstance(call.func.value, ast.Name):
                if call.func.value.id in {'val_DAS_losses', 'val_OD_losses', 'train_OD_losses'}:
                    # These regression checks expect a direct metric variable;
                    # ast.unparse would unnecessarily require Python 3.9+.
                    assert len(call.args) == 1 and isinstance(call.args[0], ast.Name)
                    assignments.append((call.func.value.id, call.args[0].id))
    assert assignments.count(('val_DAS_losses', 'avg_val_DAS_loss')) == 1
    assert ('val_OD_losses', 'avg_val_OD_loss') in assignments
    assert ('train_OD_losses', 'avg_train_OD_loss') in assignments
    assert len(assignments) == 3


def test_bev_forward_uses_voxel_pooling_output():
    forward = function_from_file('sdv/training/modeling/model.py', 'forward', {}, class_name='models')
    class Tensor:
        shape = (1, 1, 3, 32, 32)
        device = 'cpu'
        def view(self, *args): return self
        def reshape(self, *args): return self
        def to(self, *args, **kwargs): return self
        def permute(self, *args): return self
    tensor = Tensor()
    pooled = object()
    def decode(value):
        assert value is pooled
        return (1, 2, 3, 4)
    model = SimpleNamespace(cfg={'model': {'decoder': None, 'bev_decoder': 'ResNet'},
                                 'model_conf': {'Seg': False, 'Depth': False, 'od': False}, 'lidar': False},
                            camencode=lambda x: (tensor, []), camC=128, D=10, downsample=16,
                            dx=tensor, bev_enabled=True,
                            get_geometry=lambda *args: 'geometry', voxel_pooling=lambda *args: pooled,
                            bevencode=decode)
    assert forward(model, tensor, tensor, tensor, tensor, tensor, tensor, None, None)[:4] == (1, 2, 3, 4)


def test_native_bev_preserves_overlapping_labels(tmp_path):
    # Folder names containing "bev" must not be rewritten when selecting labels.
    scene = tmp_path / 'bev_dataset' / 'scene_a'
    arrays = {}
    for name in ('front_cam_rgb', 'das', 'lane', 'vehicle_mask', 'walker_mask'):
        file = scene / name / '00001.png'
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
        arrays[str(file)] = np.zeros((500, 500), dtype=np.uint8)
    arrays[str(scene / 'das' / '00001.png')][200, 200] = 255
    arrays[str(scene / 'lane' / '00001.png')][200, 200] = 255
    cv2 = SimpleNamespace(IMREAD_GRAYSCALE=0, imread=lambda path, *args: arrays.get(path))
    namespace = {'Path': Path, 'cv2': cv2, 'np': np,
                 'torch': SimpleNamespace(from_numpy=lambda value: value, zeros=np.zeros)}
    get_bev = function_from_file('sdv/training/modeling/data.py', 'get_bev',
                                 namespace, class_name='bev_Dataset')
    def aligned(idx, folder, suffixes=('.png', '.jpg', '.jpeg'), required=True):
        path = scene / folder / '00001.png'
        return path if path.is_file() else None
    dataset = SimpleNamespace(bev_schema=None, image_front=[str(scene / 'front_cam_rgb' / '00001.png')],
        train_config={'model_conf': {'BEV': True}},
        task_settings={'bev': {'layers': ['das', 'lane', 'vehicle', 'walker']}}, _aligned=aligned)
    das, lane, vehicle, walker = get_bev(dataset, 0)
    assert das.shape == (1, 200, 200)
    assert das[0, 50, 50] == 1 and lane[0, 50, 50] == 1


def test_events_use_observed_values_and_valid_json(tmp_path, monkeypatch):
    from sdv.telemetry import emit
    path = tmp_path / 'job.jsonl'
    mirror = tmp_path / 'results' / 'events.jsonl'
    monkeypatch.setenv('SDV_EVENTS_PATH', str(path))
    monkeypatch.setenv('SDV_EVENTS_STDOUT', '0')
    emit('training_epoch', _mirror_path=mirror, epoch=3,
         metrics={'val_loss': 0.125, 'invalid': float('nan')})
    event = json.loads(path.read_text())
    assert event['metrics'] == {'val_loss': 0.125, 'invalid': None}
    assert event['epoch'] == 3
    assert path.read_text() == mirror.read_text()


def test_collection_cleanup_selects_only_owned_actors(monkeypatch):
    from sdv.collection.session import CollectionSession
    import sdv.runtime.actors
    cleaned = []
    monkeypatch.setattr(sdv.runtime.actors, 'destroy_owned_actors',
                        lambda client, actors, job, seconds: cleaned.extend(actor.id for actor in actors))
    requested = []
    restored = []
    world = SimpleNamespace(get_settings=lambda: 'original settings', get_weather=lambda: 'original weather',
                            get_actors=lambda ids: requested.extend(ids) or [SimpleNamespace(id=i, type_id='vehicle.test') for i in ids],
                            apply_settings=lambda settings: restored.append(settings),
                            set_weather=lambda weather: restored.append(weather))
    traffic_manager = SimpleNamespace(set_synchronous_mode=lambda value: restored.append(value))
    client = SimpleNamespace(set_timeout=lambda value: None)
    session = CollectionSession(client, traffic_manager)
    session.begin_world(world)
    session.own([7, 9])
    session.close()
    session.close()
    assert set(requested) == {7, 9} and set(cleaned) == {7, 9}
    assert restored[:2] == ['original settings', 'original weather']


def test_raw_dataset_preflight_does_not_require_preview_fixture(tmp_path, monkeypatch):
    from sdv.config.validation import training_blockers
    config = {'sensor_use': {'front_cam': True}, 'model_conf': {'BEV': True}}
    for split, variable in [('train', 'SDV_TRAIN_DATA'), ('val', 'SDV_VAL_DATA')]:
        root = tmp_path / split
        monkeypatch.setenv(variable, str(root))
        for modality in ('front_cam_rgb', 'das', 'lane', 'vehicle_mask', 'walker_mask'):
            path = root / 'scene1' / modality / 'frame.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.delenv('SDV_TEST_TENSOR', raising=False)
    assert training_blockers(config) == []
    (tmp_path / 'train/scene1/lane/frame.png').unlink()
    assert any('BEV lane 표본' in item for item in training_blockers(config))


def test_collection_scene_respects_each_camera_modalities(tmp_path, monkeypatch):
    import yaml
    import os
    import sdv.config.validation
    from sdv.collection.bev_adapter import configured_settings, annotate_scene_map
    from sdv.collection.driving_runtime import driving_settings
    config = {'sensors': {'front_cam': {'seg': False, 'depth': False, 'od': False},
                          'right_cam': {'seg': True, 'depth': True, 'od': True}}}
    monkeypatch.setattr(sdv.config.validation, 'read_sensor_config', lambda: config)
    make_scene_dirs = function_from_file('sdv/collection/workflow.py', 'make_scene_dirs',
                                         {'Path': Path, 'List': List, 'OUTPUT_ROOT': tmp_path,
                                          'yaml': yaml, 'json': json, 'RUN_TAG': 'test', 'FPS': 10, 'os': os,
                                          'configured_settings': configured_settings,
                                          'driving_settings': driving_settings,
                                          'annotate_scene_map': annotate_scene_map})
    scene = make_scene_dirs('Town01', 'scene_a', ['front_cam', 'right_cam'], False, False, False, False)
    assert not (scene / 'front_cam_bbs').exists()
    assert all((scene / f'right_cam_{suffix}').is_dir() for suffix in ('rgb', 'bbs', 'seg', 'depth'))
    assert yaml.safe_load((scene / 'sensor_config.yaml').read_text()) == config


def test_disabling_last_checkpoint_keeps_best_checkpoint():
    import os
    source = (ROOT / 'sdv/training/experiment.py').read_text()
    start = source.index('    # Preserve the checkpoint interface')
    end = source.index('    measured = ', start)
    import textwrap
    saved = []
    namespace = {'save_dirs': '/result', 'train_config': {'save_checkboxes': {'Last_Epoch': False}},
                 'model': SimpleNamespace(state_dict=lambda: {'weight': 1}),
                 'torch': SimpleNamespace(save=lambda data, path: saved.append(path)),
                 'os': SimpleNamespace(replace=lambda src, dst: None),
                 'avg_val_loss': 0.3, 'best_val_loss': 0.5}
    exec(textwrap.dedent(source[start:end]), namespace)
    assert saved == ['/result/best_loss_model.pth.tmp']
    assert namespace['last_path'] is None
    assert namespace['checkpoint'] == '/result/best_loss_model.pth'
