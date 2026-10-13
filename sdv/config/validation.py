"""Validation shared by the interface and lightweight regression checks."""
import math
import copy
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
import yaml
from sdv.paths import ROOT
from sdv.training.model_registry import BACKBONES, BEV_DECODERS, IMAGE_DECODERS, get_backbone
from sdv.data_contract import normalize_task_settings, BEV_LAYERS

CAMERAS = {'front_cam', 'front_cam_pv', 'left_cam', 'right_cam', 'rear_cam', 'rear_left_cam', 'rear_right_cam'}
TOWNS = {'Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD'}

def integer(value, label, low, high):
    try:
        if isinstance(value, bool): raise ValueError()
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f'{label}: 숫자를 입력하세요.') from None
    if not parsed.is_finite() or parsed != parsed.to_integral_value() or not low <= parsed <= high:
        raise ValueError(f'{label}: {low}~{high} 사이의 정수를 입력하세요.')
    return int(parsed)

def number(value, label, low, high):
    try: result = float(value)
    except (TypeError,ValueError): raise ValueError(f'{label}: 숫자를 입력하세요.')
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f'{label}: {low}~{high} 사이의 숫자를 입력하세요.')
    return result

def local_path(path):
    result = Path(path).resolve()
    if ROOT not in result.parents:
        raise ValueError('파일 경로는 SDV_V1 안에 있어야 합니다.')
    return result

def sensor_value(key, value):
    bounds = {'width': (64, 1920), 'height': (64, 1080), 'fov': (10, 160),
              'points_per_second': (100, 200000), 'rotation_frequency': (1, 20), 'channels': (1, 128),
              'range': (1, 150), 'x': (-10, 10), 'y': (-10, 10), 'z': (.1, 10),
              'pitch': (-180, 180), 'yaw': (-180, 180), 'roll': (-180, 180),
              'upper_fov': (-90, 90), 'lower_fov': (-90, 90)}
    low, high = bounds.get(key, (-10000, 10000))
    parse = integer if key in {'width', 'height', 'channels', 'points_per_second', 'rotation_frequency'} else number
    return parse(value, key, low, high)

def validate_sensor_config(config):
    if not isinstance(config, dict):
        raise ValueError('센서 설정 YAML의 형식을 확인하세요.')
    towns, sensors = config.get('towns'), config.get('sensors')
    if not isinstance(towns, list) or not towns or any(t not in TOWNS for t in towns):
        raise ValueError('타운을 하나 이상 선택하세요. 지원 타운: Town01~07, Town10HD')
    if not isinstance(sensors, dict) or not any(k in CAMERAS for k in sensors):
        raise ValueError('카메라를 하나 이상 선택하세요. 라이다만 수집하는 경로는 아직 지원하지 않습니다.')
    checked = copy.deepcopy(config)
    for name, fields in sensors.items():
        if name not in CAMERAS | {'lidar'} or not isinstance(fields, dict):
            raise ValueError('지원하는 센서와 설정을 선택하세요.')
        for key, value in fields.items():
            if key not in {'seg', 'depth', 'od'}:
                checked['sensors'][name][key] = sensor_value(key, value)
            elif not isinstance(value, bool):
                raise ValueError(f'{name}.{key}: true 또는 false를 지정하세요.')
        required = {'x', 'y', 'z', 'pitch', 'yaw', 'roll'}
        if name != 'lidar': required |= {'width', 'height', 'fov'}
        if required - fields.keys():
            raise ValueError(f'{name}: 필수 항목 누락 {", ".join(sorted(required - fields.keys()))}')
        for key in ('seg', 'depth', 'od'): checked['sensors'][name].setdefault(key, False)
    checked['data_num'] = integer(config.get('data_num'), '타운당 수집 수', 1, 1000)
    return checked

def read_sensor_config(path=None):
    from sdv.telemetry import configured_path
    selected = configured_path('SDV_SENSOR_CONFIG', 'data/sensor_config.yaml') if path is None else local_path(path)
    with selected.open(encoding='utf-8') as f:
        return validate_sensor_config(yaml.safe_load(f))

def atomic_yaml(path, value):
    destination = local_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(value, allow_unicode=True), encoding='utf-8')
    temporary.replace(destination)

def training_blockers(config=None):
    from sdv.telemetry import configured_path
    missing=[]
    roots = [('SDV_TRAIN_DATA', 'data/train', '학습'), ('SDV_VAL_DATA', 'data/val', '검증')]
    for env_name, default, label in roots:
        p=configured_path(env_name, default)
        if not p.exists() or (p.is_dir() and not any(p.rglob('*.png'))):
            missing.append(f'{label} 데이터 분할 {p}')
            continue
        # The legacy loader indexes one scene below the split root and uses the
        # front camera as its sample index. Detect mismatches before GPU import.
        front = sorted(p.rglob('front_cam_rgb/*.png'))
        if not front:
            missing.append(f'{label}: <split>/<scene>/front_cam_rgb/*.png 구조가 필요합니다.')
            continue
        if config:
            required = [f'{camera}_rgb' for camera, enabled in config.get('sensor_use', {}).items()
                        if enabled and camera != 'lidar']
            tasks = config.get('model_conf', {})
            if tasks.get('BEV'):
                layers=normalize_task_settings(config.get('task_settings'))['bev']['layers']
                for layer in layers:
                    folder=BEV_LAYERS[layer]['folder']
                    bad=next((sample for sample in front if not (sample.parent.parent/folder/sample.name).is_file()
                        and not (layer in {'das','lane'} and (sample.parent.parent/'bev'/sample.name).is_file())),None)
                    if bad:missing.append(f'{label}: BEV {layer} 표본이 없습니다 ({bad.stem}).')
            for camera, enabled in config.get('sensor_use', {}).items():
                if not enabled or camera == 'lidar': continue
                required += [f'{camera}_{suffix}' for task, suffix in [('Seg', 'seg'), ('Depth', 'depth'), ('od', 'bbs')]
                             if tasks.get(task)]
            if config.get('lidar'): required += ['lidar']
            for folder in required:
                extension = '.npy' if folder == 'lidar' or folder.endswith('_bbs') else '.png'
                missing_sample = next((sample for sample in front
                    if not (sample.parent.parent / folder / (sample.stem + extension)).is_file()
                    and not (folder.endswith('_seg') and (sample.parent.parent/(folder+'_ids')/sample.name).is_file())
                    and not (folder.endswith('_depth') and (sample.parent.parent/(folder+'_m')/(sample.stem+'.npy')).is_file())), None)
                if missing_sample:
                    missing.append(f'{label}: {folder} 표본이 없습니다 ({missing_sample.parent.parent.name}/{missing_sample.stem}).'
                                   + (' 수집 das/lane과 기존 학습 bev 형식은 서로 다릅니다.' if folder == 'bev' else ''))
    tensor = configured_path('SDV_TEST_TENSOR', 'img/test.pth')
    if os.environ.get('SDV_TEST_TENSOR') and not tensor.is_file():
        missing.append(f'선택한 GT 미리보기 텐서가 없습니다: {tensor}')
    # Some callers deliberately inspect only dataset modalities before choosing
    # a model. Preserve that contract; read_train_config still requires one.
    if config and 'model' in config:
        model=config.get('model',{})
        try: spec=get_backbone(model.get('encoder') if isinstance(model,dict) else None)
        except ValueError as error: missing.append(str(error))
        else:
            filename=spec['weights']['filename']
            if model.get('pretrained',True) and not (ROOT/'.runtime/cache/torch/hub/checkpoints'/filename).is_file():
                missing.append('사전학습 가중치 '+filename+' (웹 학습 화면의 준비 상태 검사에서 다운로드하세요.)')
    return missing

def read_train_config(path):
    with local_path(path).open(encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
    if not isinstance(cfg,dict):raise ValueError('학습 설정 YAML 형식을 확인하세요.')
    for key,label,low,high in [('Epochs','에포크',1,1000),('batch','배치',1,64),('width','이미지 너비',64,1920),('height','이미지 높이',64,1080)]:
        cfg[key]=integer(cfg.get(key),label,low,high)
    cfg['LR']=number(cfg.get('LR'),'학습률',.000001,1)
    if cfg['width'] % 32 or cfg['height'] % 32:
        raise ValueError('현재 인코더/디코더의 정렬을 위해 학습 이미지 너비와 높이를 32의 배수로 지정하세요.')
    opts=cfg.get('opt_checkboxes',{})
    if not isinstance(opts, dict) or any(not isinstance(v, bool) for v in opts.values()) or sum(bool(v) for v in opts.values())!=1 or any(k not in {'Adam','AdamW','SGD'} for k in opts):
        raise ValueError('옵티마이저를 하나 선택하세요.')
    if not isinstance(cfg.get('model'), dict) or not isinstance(cfg['model'].get('encoder'), str) or cfg['model']['encoder'] not in BACKBONES:
        raise ValueError('지원되는 이미지 인코더를 선택하세요.')
    if 'pretrained' in cfg['model'] and type(cfg['model']['pretrained']) is not bool:
        raise ValueError('model.pretrained는 true/false여야 합니다.')
    tasks = cfg.get('model_conf', {})
    if not isinstance(tasks, dict) or set(tasks) - {'BEV', 'Seg', 'Depth', 'od'} or any(not isinstance(v, bool) for v in tasks.values()):
        raise ValueError('지원 출력은 BEV, Seg, Depth, od입니다.')
    if not any(tasks.values()):raise ValueError('출력 태스크를 하나 이상 선택하세요.')
    for key in ('BEV', 'Seg', 'Depth', 'od'): tasks.setdefault(key, False)
    if tasks['od'] and (cfg['width']%128 or cfg['height']%128):
        raise ValueError('객체 탐지(OD)는 기존 Anchor 크기에 맞게 이미지 너비와 높이를 128의 배수로 지정하세요 (예: 128, 256, 384).')
    supported_cameras = CAMERAS - {'front_cam_pv'}
    selected = cfg.get('sensor_use', {})
    if not isinstance(selected, dict) or any(not isinstance(v, bool) for v in selected.values()) or not selected.get('front_cam'):
        raise ValueError('기존 데이터 로더의 표본 기준인 front_cam을 선택하세요.')
    if any(enabled and key not in supported_cameras | {'lidar'} for key, enabled in selected.items()):
        raise ValueError('front_cam_pv는 수집 미리보기용이며 기존 학습 로더에서 지원하지 않습니다.')
    for camera in supported_cameras | {'lidar'}: selected.setdefault(camera, False)
    cfg['lidar'] = bool(cfg.get('lidar', selected.get('lidar', False)))
    if cfg['lidar'] != bool(selected.get('lidar')):
        raise ValueError('lidar와 sensor_use.lidar 설정을 동일하게 지정하세요.')
    model = cfg['model']
    if tasks['BEV'] and model.get('bev_decoder') not in BEV_DECODERS:
        raise ValueError('BEV 태스크에는 BEV decoder가 필요합니다.')
    if any(tasks[key] for key in ('Seg', 'Depth', 'od')) and model.get('decoder') not in IMAGE_DECODERS:
        raise ValueError('이미지 태스크에는 '+', '.join(IMAGE_DECODERS)+' decoder가 필요합니다.')
    if not tasks['BEV']:model['bev_decoder']=None
    save_options = cfg.get('save_checkboxes', {})
    if not isinstance(save_options, dict) or any(not isinstance(value, bool) for value in save_options.values()):
        raise ValueError('Checkpoint 저장 옵션은 true 또는 false여야 합니다.')
    if any(enabled and key not in {'Last_Epoch', 'Best_Loss'} for key, enabled in save_options.items()):
        raise ValueError('지원하는 Checkpoint 저장 옵션은 Last_Epoch와 Best_Loss입니다. Best_iou는 아직 구현되지 않았습니다.')
    if 'losses' in cfg:
        from sdv.training.loss_config import normalize_losses
        cfg['losses']=normalize_losses(cfg['losses'])
    monitor=cfg.setdefault('monitor',{})
    if not isinstance(monitor,dict) or set(monitor)-{'train_preview_every','val_preview_every','preview_samples','train_preview_basis','preview_max_mib'}:
        raise ValueError('monitor는 예측 저장 주기 설정 객체여야 합니다.')
    monitor['train_preview_every']=integer(monitor.get('train_preview_every',100),'학습 예측 저장 주기',0,1000000)
    monitor['val_preview_every']=integer(monitor.get('val_preview_every',1),'검증 예측 저장 주기',1,1000)
    monitor['preview_samples']=integer(monitor.get('preview_samples',4),'고정 예측 표본 수',4,4)
    if 'train_preview_basis' in monitor:
        if monitor['train_preview_basis'] not in ('epoch','global'):raise ValueError('TRAIN 기록 기준은 epoch 또는 global이어야 합니다.')
    if 'preview_max_mib' in monitor:
        monitor['preview_max_mib']=integer(monitor['preview_max_mib'],'예측 저장 최대 용량 (MiB)',1,102400)
    if 'task_settings' in cfg:
        advanced=normalize_task_settings(cfg['task_settings'])
        if tasks['Seg'] and advanced['segmentation']['mode']=='carla' and not advanced['segmentation']['class_ids']:
            raise ValueError('Segmentation: select at least one observed class')
        if tasks['od'] and not advanced['detection']['class_ids']:
            raise ValueError('Object Detection: select at least one observed class')
        if tasks['BEV'] and not advanced['bev']['layers']:
            raise ValueError('BEV: select at least one trainable layer')
        cfg['task_settings']=advanced
    return cfg
