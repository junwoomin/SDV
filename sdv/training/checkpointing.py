"""Epoch-boundary checkpoints for resumable research jobs.

Model-only legacy checkpoints are still written by the training engine. This
separate checkpoint contains everything required to continue the same experiment.
Only a trusted, locally generated checkpoint should be passed to ``load_resume``.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

SCHEMA_VERSION = 1
# Labels changed from color-linear resize / ambiguous depth scale to explicit
# IDs, nearest sampling, native meter encoding and frame-aligned modalities.
# Older paused experiments must not silently continue with different labels.
DATA_PIPELINE_SCHEMA = 2
BEV_GT_CONTRACT_VERSION = 'schema3-yx-same-frame-world-pose-v1'
GLOBAL_RASTER_CONTRACT_VERSION = 'schema4-global-png-affine-yx-v1'
PAUSE_EXIT_CODE = 75


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_signature(root):
    """Hash file identities and actual content, rather than relying on mtimes."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('재개할 데이터셋 경로가 없습니다: ' + str(root))
    digest = hashlib.sha256()
    count = 0
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        record = [path.relative_to(root).as_posix(), path.stat().st_size, _hash_file(path)]
        digest.update(json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\n')
        count += 1
    return {'root': str(root), 'files': count, 'sha256': digest.hexdigest()}


def experiment_signature(config, sensor_path, train_root, val_root, bev_schema=None):
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    sensor_path = Path(sensor_path).expanduser().resolve()
    signature = {'data_pipeline_schema': DATA_PIPELINE_SCHEMA,
            'config_sha256': hashlib.sha256(encoded).hexdigest(),
            'sensor_sha256': _hash_file(sensor_path),
            'train': dataset_signature(train_root), 'val': dataset_signature(val_root)}
    if bev_schema == 3:
        # Older legacy experiments retain their exact signature. New GT uses
        # a different spatial/calibration contract even if model heads match.
        signature['bev_ground_truth_contract'] = BEV_GT_CONTRACT_VERSION
    elif bev_schema == 4:
        signature['bev_ground_truth_contract'] = GLOBAL_RASTER_CONTRACT_VERSION
    return signature


def pause_requested(path=None):
    selected = path or os.environ.get('SDV_TRAIN_CONTROL')
    if not selected:
        return False
    try:
        value = json.loads(Path(selected).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as error:
        raise ValueError('학습 일시정지 제어 파일을 읽을 수 없습니다: ' + str(error))
    return isinstance(value, dict) and (value.get('pause_requested') is True or value.get('action') == 'pause')


def capture_rng():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda'):
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('체크포인트의 CUDA 장치 구성과 현재 실행 환경이 다릅니다.')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def preserve_rng(function, *args, **kwargs):
    """Preview inference must not consume the next training batch's RNG state."""
    state = capture_rng()
    try:
        return function(*args, **kwargs)
    finally:
        restore_rng(state)


def _atomic_torch_save(value, path):
    import torch
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + destination.name + '-', dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'wb') as output:
            torch.save(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, str(destination))
        # A power interruption must not lose the directory's atomic rename.
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_resume(path, model, optimizer, scheduler, completed_epoch, global_step,
                best_val_loss, histories, epoch_records, signature, output, config):
    """Called after scheduler.step; next epoch uses exactly the saved LR/RNG."""
    payload = {'schema_version': SCHEMA_VERSION, 'created_at': time.time(),
               'completed_epoch': int(completed_epoch), 'next_epoch': int(completed_epoch) + 1,
               'global_step': int(global_step), 'best_val_loss': float(best_val_loss),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'rng': capture_rng(),
               'histories': histories, 'epoch_records': epoch_records,
               'signature': signature, 'output': str(Path(output).resolve()), 'config': config}
    _atomic_torch_save(payload, path)
    return payload


def load_resume(path, signature, output, model, optimizer, scheduler, device=None):
    """Reject different settings, calibration, data or output before loading state."""
    import torch
    try:
        payload = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:  # Torch releases supported by the Python 3.7 research env.
        payload = torch.load(str(path), map_location='cpu')
    if not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('재개 가능한 SDV 학습 체크포인트가 아닙니다.')
    saved_signature = payload.get('signature')
    saved_pipeline = saved_signature.get('data_pipeline_schema') if isinstance(saved_signature, dict) else None
    current_pipeline = signature.get('data_pipeline_schema') if isinstance(signature, dict) else None
    if saved_pipeline != DATA_PIPELINE_SCHEMA or current_pipeline != DATA_PIPELINE_SCHEMA:
        raise ValueError(
            '데이터 처리 버전(data_pipeline_schema)이 다른 체크포인트입니다. '
            'Segmentation 라벨 보간·매핑과 Depth 단위 처리 방식이 변경되어 이전 학습을 그대로 재개할 수 없습니다. '
            '해당 체크포인트를 만든 이전 SDV 코드로 재개하거나, 현재 코드에서 새 실험을 시작하세요. '
            '저장 버전: {}, 현재 버전: {}.'.format(saved_pipeline, DATA_PIPELINE_SCHEMA))
    if saved_signature != signature:
        raise ValueError('학습 설정, 센서 보정 또는 데이터셋 내용이 변경되어 이 실험을 재개할 수 없습니다.')
    if payload.get('output') != str(Path(output).resolve()):
        raise ValueError('재개 체크포인트와 결과 저장 경로가 다릅니다.')
    completed = payload.get('completed_epoch')
    if type(completed) is not int or completed < 1 or payload.get('next_epoch') != completed + 1:
        raise ValueError('재개 체크포인트의 Epoch 정보가 잘못되었습니다.')
    if len(payload.get('epoch_records', [])) != completed:
        raise ValueError('재개 체크포인트의 학습 지표 기록이 완전하지 않습니다.')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    if device is None:
        device = next(model.parameters()).device
    # Torch's optimizer loader usually moves tensors itself, but older releases
    # can retain CPU momentum tensors after a map_location='cpu' load.
    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value
    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state)
    return payload
