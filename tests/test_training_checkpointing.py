"""Actual CPU optimizer/RNG continuity, plus dependency-free integrity checks."""
import json
import os
from pathlib import Path
import random

import pytest

from sdv.training import checkpointing


def test_dataset_signature_uses_content_not_only_size_or_time(tmp_path):
    frame = tmp_path / 'scene' / 'front_cam_rgb' / '0001.png'
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b'first')
    before = checkpointing.dataset_signature(tmp_path)
    stat = frame.stat()
    frame.write_bytes(b'other')
    os.utime(str(frame), ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert checkpointing.dataset_signature(tmp_path) != before


def test_signature_detects_calibration_and_config_change(tmp_path):
    train, val = tmp_path / 'train', tmp_path / 'val'
    train.mkdir()
    val.mkdir()
    (train / 'frame.png').write_bytes(b'train')
    (val / 'frame.png').write_bytes(b'val')
    sensor = tmp_path / 'sensor.yaml'
    sensor.write_text('fov: 90')
    before = checkpointing.experiment_signature({'LR': 0.001}, sensor, train, val)
    assert before['data_pipeline_schema'] == checkpointing.DATA_PIPELINE_SCHEMA
    assert checkpointing.experiment_signature({'LR': 0.002}, sensor, train, val) != before
    sensor.write_text('fov: 91')
    assert checkpointing.experiment_signature({'LR': 0.001}, sensor, train, val) != before


def test_pause_control_protocol(tmp_path):
    path = tmp_path / 'control.json'
    assert not checkpointing.pause_requested(path)
    for value, expected in [({'pause_requested': True}, True), ({'action': 'pause'}, True),
                            ({'pause_requested': False}, False), ({'action': 'stop'}, False),
                            ({'pause_requested': 'true'}, False), ([], False)]:
        path.write_text(json.dumps(value))
        assert checkpointing.pause_requested(path) is expected
    path.write_text('{broken')
    with pytest.raises(ValueError, match='제어 파일'):
        checkpointing.pause_requested(path)


@pytest.mark.parametrize('optimizer_name', ['Adam', 'AdamW', 'SGD'])
def test_actual_cpu_resume_matches_uninterrupted_training(tmp_path, optimizer_name):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    torch.set_num_threads(1)

    def create():
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.25), torch.nn.Linear(5, 1))
        options = {'lr': 0.01}
        if optimizer_name == 'SGD':
            options['momentum'] = 0.9
        opt = getattr(torch.optim, optimizer_name)(model.parameters(), **options)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7)
        return model, opt, sched

    def seed():
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)

    def epoch(model, opt, sched):
        loss = None
        for _ in range(3):
            x = torch.randn(4, 3) + random.random() + float(np.random.rand())
            target = torch.randn(4, 1)
            opt.zero_grad()
            loss = (model(x) - target).square().mean()
            loss.backward()
            opt.step()
        sched.step()
        return float(loss.detach())

    seed()
    full_model, full_opt, full_sched = create()
    expected_losses = [epoch(full_model, full_opt, full_sched) for _ in range(4)]
    expected_rng = (random.random(), float(np.random.rand()), torch.randn(3))

    seed()
    paused_model, paused_opt, paused_sched = create()
    first_loss = epoch(paused_model, paused_opt, paused_sched)
    path = tmp_path / 'resume_checkpoint.pth'
    checkpointing.save_resume(path, paused_model, paused_opt, paused_sched, completed_epoch=1,
        global_step=3, best_val_loss=first_loss, histories={'loss': [first_loss]},
        epoch_records=[{'epoch': 1, 'train_loss': first_loss}], signature={'immutable': 'value', 'data_pipeline_schema': checkpointing.DATA_PIPELINE_SCHEMA},
        output=tmp_path, config={'Epochs': 4})
    # A new process creates its model before restoring the saved RNG.
    resumed_model, resumed_opt, resumed_sched = create()
    state = checkpointing.load_resume(path, {'immutable': 'value', 'data_pipeline_schema': checkpointing.DATA_PIPELINE_SCHEMA}, tmp_path,
        resumed_model, resumed_opt, resumed_sched)
    assert state['completed_epoch'] == 1 and state['next_epoch'] == 2 and state['global_step'] == 3
    checkpointing.restore_rng(state['rng'])
    actual_losses = [first_loss] + [epoch(resumed_model, resumed_opt, resumed_sched) for _ in range(3)]
    assert actual_losses == expected_losses
    for expected, actual in zip(full_model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)
    assert full_sched.state_dict() == resumed_sched.state_dict()
    actual_rng = (random.random(), float(np.random.rand()), torch.randn(3))
    assert actual_rng[:2] == expected_rng[:2]
    assert torch.equal(actual_rng[2], expected_rng[2])


def test_resume_rejects_changed_settings_before_model_mutation(tmp_path):
    torch = pytest.importorskip('torch')
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    path = tmp_path / 'resume_checkpoint.pth'
    checkpointing.save_resume(path, model, opt, sched, 1, 4, 0.5, {'loss': [0.5]},
        [{'epoch': 1}], {'data': 'old', 'data_pipeline_schema': checkpointing.DATA_PIPELINE_SCHEMA}, tmp_path, {})
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='변경'):
        checkpointing.load_resume(path, {'data': 'new', 'data_pipeline_schema': checkpointing.DATA_PIPELINE_SCHEMA}, tmp_path, model, opt, sched)
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match='저장 경로'):
        checkpointing.load_resume(path, {'data': 'old', 'data_pipeline_schema': checkpointing.DATA_PIPELINE_SCHEMA}, tmp_path / 'other', model, opt, sched)


def test_atomic_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'resume_checkpoint.pth'
    path.write_bytes(b'previous complete checkpoint')
    def fail(value, output):
        output.write(b'partial')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        checkpointing._atomic_torch_save({}, path)
    assert path.read_bytes() == b'previous complete checkpoint'
    assert list(tmp_path.iterdir()) == [path]


def test_preview_preserves_training_rng_even_after_error():
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    before = checkpointing.capture_rng()
    expected = (random.random(), float(np.random.rand()), torch.randn(3))
    checkpointing.restore_rng(before)
    def preview():
        random.random()
        np.random.rand()
        torch.randn(10)
        raise RuntimeError('preview failed')
    with pytest.raises(RuntimeError, match='preview failed'):
        checkpointing.preserve_rng(preview)
    actual = (random.random(), float(np.random.rand()), torch.randn(3))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])


def test_real_collater_preserves_depth_mask_for_original_silog_loss():
    torch = pytest.importorskip('torch')
    import ast
    from sdv.training.modeling.loss import silog_loss
    source = Path(__file__).resolve().parents[1] / 'sdv/training/experiment.py'
    tree = ast.parse(source.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'collater')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    image = torch.zeros(1, 3, 2, 2)
    matrix = torch.eye(3).unsqueeze(0)
    translation = torch.zeros(1, 3)
    depth = torch.tensor([[[[2., 0.], [4., 8.]]]])
    mask = depth > 0
    sample = (image, image, None, None, None, None, None, None,
              matrix, translation, matrix, matrix, matrix, translation, None, depth, mask, [])
    batch = namespace['collater']([sample, sample])
    assert batch[16].dtype == torch.bool
    assert torch.equal(batch[16], torch.stack([mask, mask]))
    prediction = (batch[15] + 1).requires_grad_()
    loss = silog_loss(0.85)(prediction, batch[15], batch[16])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[~batch[16]] == 0).all()


@pytest.mark.parametrize('saved_version', [None, 1, 3])
def test_resume_rejects_old_label_pipeline_before_model_mutation(tmp_path, saved_version):
    torch = pytest.importorskip('torch')
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    path = tmp_path / 'resume_checkpoint.pth'
    old = {'same_config_data_and_sensor': 'unchanged'}
    if saved_version is not None:
        old['data_pipeline_schema'] = saved_version
    checkpointing.save_resume(path, model, opt, sched, 1, 4, 0.5, {'loss': [0.5]},
        [{'epoch': 1}], old, tmp_path, {})
    current = dict(old, data_pipeline_schema=checkpointing.DATA_PIPELINE_SCHEMA)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='이전 SDV 코드'):
        checkpointing.load_resume(path, current, tmp_path, model, opt, sched)
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())
    assert not opt.state
