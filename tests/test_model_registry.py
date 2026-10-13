"""Web/research configuration parity and lightweight registry contracts."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from studio.config import Config
from studio.pretrained_service import WEIGHTS
from sdv.config import validation
from sdv.training.model_registry import (
    BACKBONES, BEV_DECODERS, IMAGE_DECODERS, get_backbone, model_catalog,
)


def test_registry_import_has_no_torch_or_research_dependency():
    result = subprocess.run([sys.executable, '-c',
        'import sys,json; from sdv.training.model_registry import model_catalog; '
        'assert not any(x in sys.modules for x in ("torch","torchvision","efficientnet_pytorch")); '
        'print(json.dumps(model_catalog()))'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0, result.stderr
    catalog = json.loads(result.stdout)
    assert len(catalog['encoders']) == 11
    assert catalog['defaults'] == {'encoder': 'ResNet', 'decoder': 'FPN', 'bev_decoder': None}
    assert catalog['bev_decoders'][0]['id'] is None


def test_registry_has_consistent_stage_and_weight_contracts():
    for name, spec in BACKBONES.items():
        assert spec['id'] == name
        assert len(spec['feature_indices']) == len(spec['feature_channels']) == 3
        assert spec['feature_strides'] == (32, 16, 8)
        assert all(channel > 0 for channel in spec['feature_channels'])
        weight = WEIGHTS[name]
        assert weight == spec['weights']
        assert weight['source_url'].endswith('/' + weight['filename'])
        assert '-' + weight['hash_prefix'] + '.pth' in weight['filename']
        assert len(weight['hash_prefix']) == 8
        assert '/' not in weight['filename']
    assert BACKBONES['ResNet']['canonical_id'] == BACKBONES['ResNet34']['canonical_id']
    assert BACKBONES['ResNet']['weights']['filename'] == BACKBONES['ResNet34']['weights']['filename']
    assert BACKBONES['EfficientNet']['provider'] != BACKBONES['EfficientNet_B0']['provider']
    assert WEIGHTS['EfficientNet']['filename'] != WEIGHTS['EfficientNet_B0']['filename']
    assert all(spec['minimum_torchvision'] is None for spec in BACKBONES.values() if spec['family'] == 'ResNet')
    assert BACKBONES['EfficientNet_B0']['minimum_torchvision'] == '0.11'
    assert BACKBONES['MobileNetV3_Small']['minimum_torchvision'] == '0.9'
    assert BACKBONES['ConvNeXt_Tiny']['minimum_torchvision'] == '0.12'
    entry = get_backbone('ResNet')
    entry['weights']['filename'] = 'changed'
    assert get_backbone('ResNet')['weights']['filename'] != 'changed'
    catalog = model_catalog()
    catalog['defaults']['encoder'] = 'changed'
    assert model_catalog()['defaults']['encoder'] == 'ResNet'


@pytest.mark.parametrize('name', list(BACKBONES))
def test_new_encoder_settings_are_accepted_by_web_and_research(tmp_path, monkeypatch, name):
    config = Config(tmp_path)
    value = config.training_defaults()
    value['model'].update(encoder=name, decoder='UNet')
    checked = config.save_training(value)
    assert config.training()['model']['encoder'] == name
    assert checked['model']['decoder'] == 'UNet'
    file = tmp_path / 'train.yaml'
    file.write_text(yaml.safe_dump(checked), encoding='utf-8')
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    assert validation.read_train_config(file)['model'] == checked['model']


@pytest.mark.parametrize('decoder', BEV_DECODERS)
def test_all_bev_decoders_require_bev_and_normalize_when_disabled(tmp_path, monkeypatch, decoder):
    config = Config(tmp_path)
    value = config.training_defaults()
    value['model'].update(bev_decoder=decoder)
    value['model_conf']['BEV'] = True
    value['sensor_use'].update(left_cam=True, right_cam=True)
    assert config.validate_training(value)['model']['bev_decoder'] == decoder
    value['model_conf']['BEV'] = False
    assert config.validate_training(value)['model']['bev_decoder'] is None
    file = tmp_path / 'train.yaml'
    file.write_text(yaml.safe_dump(value), encoding='utf-8')
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    assert validation.read_train_config(file)['model']['bev_decoder'] is None


@pytest.mark.parametrize('name', ('Unknown', 'resnet18', '../ResNet', None, {}, 123))
def test_unsupported_encoder_never_falls_back_to_resnet(tmp_path, name):
    with pytest.raises(ValueError):
        get_backbone(name)
    value = Config(tmp_path).training_defaults()
    value['model']['encoder'] = name
    with pytest.raises(ValueError):
        Config(tmp_path).validate_training(value)


def test_training_blockers_use_selected_weights_and_allow_explicit_random_init(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    config = Config(tmp_path)
    value = config.training_defaults()
    value['model']['encoder'] = 'ConvNeXt_Tiny'
    blockers = validation.training_blockers(value)
    assert any(WEIGHTS['ConvNeXt_Tiny']['filename'] in item for item in blockers)
    assert not any(WEIGHTS['ResNet']['filename'] in item for item in blockers)
    value['model']['pretrained'] = False
    assert not any('사전학습 가중치' in item for item in validation.training_blockers(value))
    assert config.validate_training(value)['model']['pretrained'] is False
    value['model']['pretrained'] = 'false'
    with pytest.raises(ValueError, match='pretrained'):
        config.validate_training(value)
    value['model']['encoder'] = 'Unknown'
    assert any('지원되지 않는' in item for item in validation.training_blockers(value))


def test_data_only_blockers_do_not_require_model_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    value = {'sensor_use': {'front_cam': True}, 'model_conf': {'BEV': False}}
    assert not any('인코더' in item or '사전학습 가중치' in item
                   for item in validation.training_blockers(value))
    for model in ({}, {'encoder': 'Unknown'}, None):
        value['model'] = model
        assert any('지원되지 않는 이미지 인코더' in item
                   for item in validation.training_blockers(value))


@pytest.mark.parametrize('width,height', [(128,128), (256,256), (384,128)])
def test_detection_accepts_existing_anchor_aligned_dimensions(tmp_path, monkeypatch, width, height):
    config = Config(tmp_path)
    value = config.training_defaults()
    value['model_conf']['od'] = True
    value.update(width=width, height=height)
    assert config.validate_training(value)['width'] == width
    file = tmp_path / 'train.yaml'
    file.write_text(yaml.safe_dump(value), encoding='utf-8')
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    assert validation.read_train_config(file)['height'] == height


@pytest.mark.parametrize('width,height', [(96,128), (128,96), (96,96), (256,64)])
def test_detection_rejects_misaligned_dimensions_before_model_start(tmp_path, monkeypatch, width, height):
    config = Config(tmp_path)
    value = config.training_defaults()
    value.update(width=width, height=height)
    # The same 32-aligned dimensions remain valid for segmentation/depth.
    assert config.validate_training(value)['width'] == width
    value['model_conf']['od'] = True
    with pytest.raises(ValueError, match='128'):
        config.validate_training(value)
    file = tmp_path / 'train.yaml'
    file.write_text(yaml.safe_dump(value), encoding='utf-8')
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='128'):
        validation.read_train_config(file)
