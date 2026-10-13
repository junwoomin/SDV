"""Image backbones with one explicit research feature contract.

Every encoder returns the existing lifted depth tensor at stride 16 plus three
features ordered C5/C4/C3 (strides 32/16/8). Legacy ResNet34 and EfficientNet
retain their parameter/module names so existing state dictionaries still load.
"""
import inspect
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from sdv.training.model_registry import get_backbone


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x2, x1], dim=1))


def _version(value):
    numbers = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', str(value))
    return tuple(int(part or 0) for part in numbers.groups()) if numbers else None


def _torchvision_backbone(spec, pretrained):
    constructor = getattr(models, spec['constructor'], None)
    installed = getattr(torchvision, '__version__', 'unknown')
    minimum = spec.get('minimum_torchvision')
    installed_version, minimum_version = _version(installed), _version(minimum)
    if constructor is None or (installed_version and minimum_version and installed_version < minimum_version):
        raise ValueError('{} requires torchvision {} or newer; the research environment has {}. Choose a supported backbone or update that environment explicitly.'.format(
            spec['id'], minimum or 'with ' + spec['constructor'], installed))
    # Always instantiate without implicit URL selection. Older torchvision
    # releases use different checkpoint filenames for the same architecture;
    # explicitly load the registry file that the web preflight prepared.
    parameters = inspect.signature(constructor).parameters
    network = constructor(weights=None) if 'weights' in parameters else constructor(pretrained=False)
    if pretrained:
        state = torch.hub.load_state_dict_from_url(spec['weights']['source_url'],
                                                  map_location='cpu', progress=False, check_hash=True)
        network.load_state_dict(state)
    return network


class _DepthEncoder(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.D, self.C = D, C

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        value, features = self.get_depth(x)
        value = self.depthnet(value)
        depth = self.get_depth_dist(value[:, :self.D])
        lifted = depth.unsqueeze(1) * value[:, self.D:self.D + self.C].unsqueeze(2)
        return depth, lifted, features

    def forward(self, x):
        _, lifted, features = self.get_depth_feat(x)
        return lifted, features


class Encoder_EfficientNet(_DepthEncoder):
    """Legacy efficientnet-pytorch B0; trunk/up1/depthnet keys are unchanged."""
    def __init__(self, D, C, pretrained=True):
        super().__init__(D, C)
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as error:
            raise ImportError('The legacy EfficientNet encoder requires efficientnet-pytorch in the research Python environment. Torchvision backbones do not require that package.') from error
        self.trunk = (EfficientNet.from_pretrained('efficientnet-b0') if pretrained
                      else EfficientNet.from_name('efficientnet-b0'))
        self.feature_channels = (320, 112, 40)
        self.feature_strides = (32, 16, 8)
        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_eff_depth(self, x):
        endpoints = {}
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        previous = x
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if previous.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = previous
            previous = x
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        features = tuple(endpoints['reduction_{}'.format(level)] for level in (5, 4, 3))
        return self.up1(features[0], features[1]), features

    def get_depth(self, x):
        return self.get_eff_depth(x)


class Encoder_Resnet(_DepthEncoder):
    """ResNet family; the default preserves the original ResNet34 state keys."""
    def __init__(self, D, C, backbone='ResNet', pretrained=True):
        super().__init__(D, C)
        spec = get_backbone(backbone)
        if not spec['constructor'].startswith('resnet'):
            raise ValueError('Encoder_Resnet requires a registered ResNet backbone.')
        self.base_model = _torchvision_backbone(spec, pretrained)
        self.feat_names = ['layer1', 'layer2', 'layer3', 'layer4']
        self.feature_channels = tuple(spec['feature_channels'])
        self.feature_strides = (32, 16, 8)
        self.feat_out_channels = ([64, 128, 256, 512] if self.feature_channels[0] == 512
                                  else [256, 512, 1024, 2048])
        self.up = Up(self.feature_channels[0] + self.feature_channels[1], 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth(self, x):
        outputs = {}
        for name, module in self.base_model._modules.items():
            if name in ('fc', 'avgpool'):
                continue
            x = module(x)
            if name in self.feat_names:
                outputs[name] = x
        # FPN/BiFPN require three levels deep-to-shallow. The old four-level
        # shallow-to-deep list failed to unpack before any training step.
        features = tuple(outputs[name] for name in ('layer4', 'layer3', 'layer2'))
        return self.up(features[0], features[1]), features


class Encoder_Torchvision(_DepthEncoder):
    """Registered torchvision stages with the same depth-lifting head."""
    def __init__(self, D, C, backbone, pretrained=True):
        super().__init__(D, C)
        spec = get_backbone(backbone)
        network = _torchvision_backbone(spec, pretrained)
        self.feature_indices = tuple(spec['feature_indices'])
        self.feature_channels = tuple(spec['feature_channels'])
        self.feature_strides = (32, 16, 8)
        if not self.feature_indices or not hasattr(network, 'features'):
            raise ValueError('{} has no supported torchvision feature-stage adapter.'.format(backbone))
        self.trunk = nn.Sequential(*list(network.features.children())[:max(self.feature_indices) + 1])
        self.up = Up(self.feature_channels[0] + self.feature_channels[1], 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth(self, x):
        outputs = {}
        for index, module in enumerate(self.trunk):
            x = module(x)
            if index in self.feature_indices:
                outputs[index] = x
        features = tuple(outputs[index] for index in self.feature_indices)
        return self.up(features[0], features[1]), features


def build_encoder(name, D, C, pretrained=True):
    spec = get_backbone(name)
    if spec['provider'] == 'efficientnet_pytorch':
        return Encoder_EfficientNet(D, C, pretrained=pretrained)
    if spec['constructor'].startswith('resnet'):
        return Encoder_Resnet(D, C, backbone=name, pretrained=pretrained)
    return Encoder_Torchvision(D, C, backbone=name, pretrained=pretrained)
