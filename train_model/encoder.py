import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torchvision.models as models


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Encoder_EfficientNet(nn.Module):
    def __init__(self, D, C):
        super(Encoder_EfficientNet, self).__init__()
        self.D = D
        self.C = C
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, 512)

        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x,feats = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x,feats

    def get_eff_depth(self, x):
        endpoints = dict()

        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])

        return x,[endpoints['reduction_5'],endpoints['reduction_4'],endpoints['reduction_3']]#320 112 40

    def forward(self, x):
        depth, x ,feats = self.get_depth_feat(x)

        return x,feats
    

class Encoder_Resnet(nn.Module):
    def __init__(self, D, C):
        super(Encoder_Resnet, self).__init__()
        self.base_model = models.resnet34(pretrained=True)
        self.feat_names = [ 'layer1', 'layer2', 'layer3', 'layer4']
        self.feat_out_channels = [64, 128, 256, 512]
        self.up = Up(512+256, 512)
        self.D = D
        self.C = C
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)
    
    def get_depth(self, x):
        features = [x]
        skip_feat = []
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                if feature.size(1) in self.feat_out_channels:
                    skip_feat.append(feature)

        x=self.up(skip_feat[3],skip_feat[2])

        return x ,skip_feat
    
    def get_depth_feat(self, x):
        x,feats = self.get_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x ,feats

    def forward(self, x):
        depth, x ,feats= self.get_depth_feat(x)

        return x,feats











