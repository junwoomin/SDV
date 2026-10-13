"""Shared model contracts without importing PyTorch or research dependencies.

Feature levels are ordered deepest first: stride 32, 16, then 8. The explicit
ImageNet V1 weights are stable across torchvision versions; never use DEFAULT.
Sources are upstream torchvision model definitions, also linked in each entry.
"""
from copy import deepcopy


def _backbone(name, label, constructor, channels, indices, minimum, enum, filename,
              family, canonical=None, provider='torchvision', source_url=None):
    return {
        'id': name, 'label': label, 'canonical_id': canonical or constructor,
        'provider': provider, 'family': family, 'constructor': constructor,
        'feature_channels': tuple(channels), 'feature_indices': tuple(indices),
        'feature_strides': (32, 16, 8), 'minimum_torchvision': minimum,
        'weights_enum': enum, 'weights_member': 'IMAGENET1K_V1' if enum else None,
        'weights': {
            'model': label, 'filename': filename,
            'hash_prefix': filename.rsplit('-', 1)[1].split('.')[0],
            'source_url': source_url or 'https://download.pytorch.org/models/' + filename,
        },
    }


_RESNET_LEVELS = ('layer4', 'layer3', 'layer2')
BACKBONES = {
    'ResNet': _backbone('ResNet', 'ResNet-34 (기존)', 'resnet34', (512, 256, 128),
                       _RESNET_LEVELS, None, 'ResNet34_Weights', 'resnet34-b627a593.pth', 'ResNet'),
    'ResNet18': _backbone('ResNet18', 'ResNet-18', 'resnet18', (512, 256, 128),
                         _RESNET_LEVELS, None, 'ResNet18_Weights', 'resnet18-f37072fd.pth', 'ResNet'),
    'ResNet34': _backbone('ResNet34', 'ResNet-34', 'resnet34', (512, 256, 128),
                         _RESNET_LEVELS, None, 'ResNet34_Weights', 'resnet34-b627a593.pth', 'ResNet'),
    'ResNet50': _backbone('ResNet50', 'ResNet-50', 'resnet50', (2048, 1024, 512),
                         _RESNET_LEVELS, None, 'ResNet50_Weights', 'resnet50-0676ba61.pth', 'ResNet'),
    'ResNet101': _backbone('ResNet101', 'ResNet-101', 'resnet101', (2048, 1024, 512),
                          _RESNET_LEVELS, None, 'ResNet101_Weights', 'resnet101-63fe2227.pth', 'ResNet'),
    'EfficientNet': _backbone(
        'EfficientNet', 'EfficientNet-B0 (기존)', 'efficientnet-b0', (320, 112, 40),
        ('reduction_5', 'reduction_4', 'reduction_3'), None, None,
        'efficientnet-b0-355c32eb.pth', 'EfficientNet', canonical='efficientnet_pytorch_b0',
        provider='efficientnet_pytorch',
        source_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth'),
    'EfficientNet_B0': _backbone('EfficientNet_B0', 'EfficientNet-B0', 'efficientnet_b0',
                                (320, 112, 40), (7, 5, 3), '0.11', 'EfficientNet_B0_Weights',
                                'efficientnet_b0_rwightman-7f5810bc.pth', 'EfficientNet'),
    'EfficientNet_B1': _backbone('EfficientNet_B1', 'EfficientNet-B1', 'efficientnet_b1',
                                (320, 112, 40), (7, 5, 3), '0.11', 'EfficientNet_B1_Weights',
                                'efficientnet_b1_rwightman-bac287d4.pth', 'EfficientNet'),
    'MobileNetV3_Small': _backbone('MobileNetV3_Small', 'MobileNetV3 Small', 'mobilenet_v3_small',
                                 (96, 48, 24), (11, 8, 3), '0.9', 'MobileNet_V3_Small_Weights',
                                 'mobilenet_v3_small-047dcff4.pth', 'MobileNetV3'),
    'MobileNetV3_Large': _backbone('MobileNetV3_Large', 'MobileNetV3 Large', 'mobilenet_v3_large',
                                 (160, 112, 40), (15, 12, 6), '0.9', 'MobileNet_V3_Large_Weights',
                                 'mobilenet_v3_large-8738ca79.pth', 'MobileNetV3'),
    'ConvNeXt_Tiny': _backbone('ConvNeXt_Tiny', 'ConvNeXt Tiny', 'convnext_tiny',
                              (768, 384, 192), (7, 5, 3), '0.12', 'ConvNeXt_Tiny_Weights',
                              'convnext_tiny-983f1562.pth', 'ConvNeXt'),
}
BACKBONES['ResNet']['alias_of'] = 'ResNet34'

IMAGE_DECODERS = ('FPN', 'Bi-FPN', 'UNet')
BEV_DECODERS = ('ResNet', 'EfficientNet', 'UNet')
DEFAULT_MODEL = {'encoder': 'ResNet', 'decoder': 'FPN', 'bev_decoder': None}

# Primary definitions, retained as source provenance for registry maintenance.
SOURCES = {
    'ResNet': 'https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html',
    'EfficientNet': 'https://docs.pytorch.org/vision/main/_modules/torchvision/models/efficientnet.html',
    'MobileNetV3': 'https://docs.pytorch.org/vision/main/_modules/torchvision/models/mobilenetv3.html',
    'ConvNeXt': 'https://docs.pytorch.org/vision/main/_modules/torchvision/models/convnext.html',
    'efficientnet_pytorch': 'https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py',
}


def get_backbone(name):
    if not isinstance(name, str) or name not in BACKBONES:
        raise ValueError('지원되지 않는 이미지 인코더입니다: {}. 지원 모델: {}'.format(
            name, ', '.join(BACKBONES)))
    return deepcopy(BACKBONES[name])


def model_catalog():
    """JSON-serializable choices; runtime availability is added by the API."""
    return {
        'encoders': [get_backbone(name) for name in BACKBONES],
        'decoders': [{'id': name, 'label': name} for name in IMAGE_DECODERS],
        'bev_decoders': [{'id': None, 'label': '사용 안 함'}] +
                        [{'id': name, 'label': name} for name in BEV_DECODERS],
        'defaults': deepcopy(DEFAULT_MODEL),
    }
