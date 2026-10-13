"""Real CPU forward/backward contracts. No checkpoint download or CARLA required.

Run in a research Python containing matching CPU torch/torchvision packages:
    python -m pytest tests/test_model_contract.py -q
The lightweight web-only test environments explicitly skip this module.
"""
import builtins
import gc
import importlib.util
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip('torch', reason='Real model contract tests require PyTorch')
pytest.importorskip('torchvision', reason='Real model contract tests require torchvision')
pytest.importorskip('efficientnet_pytorch', reason='Legacy EfficientNet compatibility requires efficientnet-pytorch')

from sdv.training.model_registry import BACKBONES, BEV_DECODERS, IMAGE_DECODERS
from sdv.training.modeling.decoder import FPN, BiFPN, UNetDecoder
from sdv.training.modeling.encoder import build_encoder
from sdv.training.modeling.model import models


DECODERS = {'FPN': FPN, 'Bi-FPN': BiFPN, 'UNet': UNetDecoder}


@pytest.fixture(autouse=True)
def cpu_without_weight_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('pretrained=False must not download any model weights')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(73)
    yield
    torch.set_num_threads(old_threads)
    gc.collect()


def finite_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    assert gradients, 'No parameter gradients reached this module'
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


@pytest.mark.parametrize('encoder_name', list(BACKBONES))
@pytest.mark.parametrize('decoder_name', IMAGE_DECODERS)
def test_every_encoder_image_decoder_forward_backward(encoder_name, decoder_name):
    """All registered choices meet the same feature/lifted/OD-level contract."""
    encoder = build_encoder(encoder_name, D=4, C=8, pretrained=False).train()
    decoder = DECODERS[decoder_name](encoder.feature_channels, feat_c=16).train()
    image = torch.randn(2, 3, 64, 96, requires_grad=True)
    lifted, features = encoder(image)
    assert lifted.shape == (2, 8, 4, 4, 6)
    assert len(features) == 3
    assert tuple(encoder.feature_channels) == tuple(BACKBONES[encoder_name]['feature_channels'])
    for feature, channels, stride in zip(features, encoder.feature_channels, (32, 16, 8)):
        assert feature.shape == (2, channels, 64 // stride, 96 // stride)
        assert torch.isfinite(feature).all()
    pyramid = decoder(features)
    assert len(pyramid) == 5
    for level, stride in zip(pyramid, (8, 16, 32, 64, 128)):
        assert level.shape == (2, 16, math.ceil(64 / stride), math.ceil(96 / stride))
        assert torch.isfinite(level).all()
    (lifted.square().mean() + sum(level.square().mean() for level in pyramid)).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad)
    finite_gradients(encoder)
    finite_gradients(decoder)


def config(encoder='ResNet18', decoder='FPN', bev_decoder=None, od=False, size=128):
    return {
        'width': size, 'height': size, 'lidar': False,
        'sensor_use': {'front_cam': True, 'left_cam': bev_decoder is not None,
                       'right_cam': bev_decoder is not None, 'lidar': False},
        'model': {'encoder': encoder, 'decoder': decoder, 'bev_decoder': bev_decoder},
        'model_conf': {'BEV': bev_decoder is not None, 'Seg': bev_decoder is None,
                       'Depth': bev_decoder is None, 'od': od},
    }


def batch_inputs(batch=2, cameras=1, size=128):
    images = torch.randn(batch, cameras, 3, size, size, requires_grad=True)
    identity = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, cameras, 1, 1)
    translations = torch.zeros(batch, cameras, 3)
    translations[:, :, 0] = torch.arange(cameras) - (cameras - 1) / 2
    intrinsics = identity.clone()
    intrinsics[:, :, 0, 0] = intrinsics[:, :, 1, 1] = size / 2
    intrinsics[:, :, 0, 2] = intrinsics[:, :, 1, 2] = size / 2
    # Exact positional contract: x, trans, rots, intrins, post_trans, post_rots.
    return (images, translations, identity, intrinsics,
            torch.zeros_like(translations), identity.clone(), None, None)


@pytest.mark.parametrize('decoder_name', IMAGE_DECODERS)
def test_full_model_seg_depth_od_shapes_anchors_and_backward(decoder_name):
    """Five OD levels match real anchor count, alongside full-resolution heads."""
    model = models(config(decoder=decoder_name, od=True), pretrained=False).train()
    inputs = batch_inputs()
    output = model(*inputs)
    assert len(output) == 8
    assert all(item is None for item in output[:4])
    segmentation, depth, classification, regression = output[4:]
    assert segmentation.shape == (2, 1, 6, 128, 128)
    assert depth.shape == (2, 1, 1, 128, 128)
    assert ((depth >= 0) & (depth <= 100)).all()
    anchors = model.anchors(inputs[0].reshape(2, 3, 128, 128))
    expected_anchors = 9 * sum((128 // stride) ** 2 for stride in (8, 16, 32, 64, 128))
    assert anchors.shape == (1, expected_anchors, 4)
    assert classification.shape == (2, expected_anchors, 2)
    assert regression.shape == (2, expected_anchors, 4)
    assert ((classification >= 0) & (classification <= 1)).all()
    assert all(torch.isfinite(item).all() for item in output if item is not None)
    sum(item.square().mean() for item in output if item is not None).backward()
    finite_gradients(model.camencode)
    finite_gradients(model.decoder)
    finite_gradients(model.seg_output)
    finite_gradients(model.depth_output)
    finite_gradients(model.classifier)
    finite_gradients(model.regressor)
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()


@pytest.mark.parametrize('bev_decoder', BEV_DECODERS)
@pytest.mark.parametrize('image_decoder', [None, 'FPN'])
def test_full_model_bev_four_maps_and_geometry_backward(bev_decoder, image_decoder):
    """Keep the original BEV coordinate bounds and independent 200×200 maps."""
    model = models(config(decoder=image_decoder, bev_decoder=bev_decoder, size=64), pretrained=False).eval()
    if image_decoder is None:
        assert model.decoder is None
    inputs = batch_inputs(batch=1, cameras=3, size=64)
    assert model.nx.tolist() == [200, 200, 1]
    assert torch.allclose(model.dx, torch.tensor([.2, .2, 20.]))
    geometry = model.get_geometry(inputs[2], inputs[1], inputs[3], inputs[5], inputs[4])
    assert geometry.shape == (1, 3, 48, 4, 4, 3)
    assert torch.isfinite(geometry).all()
    output = model(*inputs)
    assert len(output) == 8
    assert all(item is None for item in output[4:])
    for logits in output[:4]:
        assert logits.shape == (1, 1, 200, 200)
        assert torch.isfinite(logits).all()
    assert len({logits.data_ptr() for logits in output[:4]}) == 4
    sum(logits.square().mean() for logits in output[:4]).backward()
    finite_gradients(model.camencode)
    finite_gradients(model.bevencode)
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()
    assert torch.count_nonzero(inputs[0].grad)


def test_resnet_alias_keeps_existing_parameter_key_and_shape_contract():
    original_name = build_encoder('ResNet', D=4, C=8, pretrained=False)
    explicit_name = build_encoder('ResNet34', D=4, C=8, pretrained=False)
    original_state, explicit_state = original_name.state_dict(), explicit_name.state_dict()
    assert original_state.keys() == explicit_state.keys()
    assert all(original_state[key].shape == explicit_state[key].shape for key in original_state)
    explicit_name.load_state_dict(original_state, strict=True)


@pytest.mark.parametrize('name', ['ResNet', 'EfficientNet'])
def test_preserved_original_encoder_checkpoint_and_lifted_numerics(name, monkeypatch):
    """Optional original-source comparison; no reference source is synthesized."""
    source = Path(os.environ.get('SDV_LEGACY_MODELING_ROOT',
                  str(Path(__file__).resolve().parents[2] / 'sources/sdv/training/modeling')))
    if not (source / 'encoder.py').is_file():
        pytest.skip('Set SDV_LEGACY_MODELING_ROOT to preserved original SDV modeling sources')
    spec = importlib.util.spec_from_file_location('qa_legacy_encoder_' + name, str(source / 'encoder.py'))
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    if name == 'ResNet':
        constructor = legacy.models.resnet34
        monkeypatch.setattr(legacy.models, 'resnet34', lambda **kwargs: constructor(weights=None))
        original = legacy.Encoder_Resnet(4, 8).eval()
    else:
        monkeypatch.setattr(legacy.EfficientNet, 'from_pretrained',
                            lambda model: legacy.EfficientNet.from_name(model))
        original = legacy.Encoder_EfficientNet(4, 8).eval()
    updated = build_encoder(name, D=4, C=8, pretrained=False).eval()
    original_state, updated_state = original.state_dict(), updated.state_dict()
    assert original_state.keys() == updated_state.keys()
    assert all(original_state[key].shape == updated_state[key].shape for key in original_state)
    updated.load_state_dict(original_state, strict=True)
    images = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        old_lifted, old_features = original(images)
        new_lifted, new_features = updated(images)
    assert torch.allclose(old_lifted, new_lifted, atol=1e-6, rtol=1e-5)
    expected = list(reversed(old_features[1:])) if name == 'ResNet' else old_features
    assert len(new_features) == len(expected) == 3
    assert all(torch.allclose(left, right, atol=1e-6, rtol=1e-5)
               for left, right in zip(expected, new_features))


def test_torchvision_paths_do_not_require_legacy_efficientnet_package(monkeypatch):
    original_import = builtins.__import__
    def block_legacy(name, *args, **kwargs):
        if name == 'efficientnet_pytorch' or name.startswith('efficientnet_pytorch.'):
            raise ImportError('QA deliberately blocked legacy EfficientNet package')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', block_legacy)
    build_encoder('MobileNetV3_Small', 4, 8, pretrained=False)
    models(config(decoder=None, bev_decoder='ResNet', size=64), pretrained=False)
    models(config(decoder=None, bev_decoder='UNet', size=64), pretrained=False)
    with pytest.raises(ImportError, match='legacy EfficientNet encoder requires'):
        build_encoder('EfficientNet', 4, 8, pretrained=False)
