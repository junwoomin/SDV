import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import yaml
from .utils import * 
from .encoder import *
from .decoder import *
from .pointpillar import PointPillarEncoder


class models(nn.Module):
    def __init__(self,cfg):
        super(models, self).__init__()

        self.cfg = cfg

        sensor_dict = cfg['sensor_use']
        self.count = sum(v for k, v in sensor_dict.items() if k != 'lidar' and v is True)


        X_BOUND = [-20.0, 20.0, 0.2]
        Y_BOUND = [-20.0, 20.0, 0.2]
        Z_BOUND = [-10.0, 10.0, 20.0]
        dx, bx, nx = gen_dx_bx(X_BOUND, Y_BOUND, Z_BOUND)

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 128
        output_C=128
        pp_C = 0
        
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        if cfg['model']['encoder'] == 'EfficientNet':
            self.camencode = Encoder_EfficientNet(self.D, self.camC)
            inC=[320, 112, 40]
        else:
            self.camencode = Encoder_Resnet(self.D, self.camC)
            inC=[512,256,128]


        self.use_quickcumsum = True
        if cfg['lidar']:
            pp_C=128
            self.pp = PointPillarEncoder(pp_C, X_BOUND, Y_BOUND, Z_BOUND)

        if cfg['model']['bev_decoder'] == 'EfficientNet':
            self.bevencode = Bev_Decoder_EfficientNet(inC=self.camC+pp_C)#320 112 40

        elif cfg['model']['bev_decoder'] == 'ResNet':
            self.bevencode = Bev_decoder_Resnet(inC=self.camC+pp_C)#512 256 128


        if cfg['model']['decoder'] == 'FPN':
            self.decoder = FPN(inC,feat_c=output_C)
        elif cfg['model']['decoder'] == 'Bi-FPN':
            self.decoder = BiFPN(inC,feat_c=output_C)

        if cfg['model_conf']['od']:
            num_classes=2

            aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
            num_scales= len([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

            num_anchors = len(aspect_ratios) * num_scales
            self.regressor = Regressor(in_channels=output_C, num_anchors=num_anchors,
                                   num_layers=4,
                                   pyramid_levels=5,
                                   onnx_export=False)

            self.classifier = Classifier(in_channels=output_C, num_anchors=num_anchors,
                                        num_classes=num_classes,
                                        num_layers=4,
                                        pyramid_levels=5,
                                        onnx_export=True)

            # self.regressor=RegressionModel(output_C)
            #self.classifier=ClassificationModel(num_features_in=128,num_classes=num_classes)

            self.anchors = Anchors(anchor_scale=1.25,onnx_export=False)


            
        if cfg['model_conf']['Seg']:
            self.seg_output= nn.Sequential(
            nn.Conv2d(output_C, 64, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, kernel_size=1, padding=0),
                )
        if cfg['model_conf']['Depth']:
            self.depth_output = nn.Sequential(
            nn.Conv2d(output_C, 64, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            )

    def create_frustum(self):

        ogfH, ogfW = int(self.cfg['height']), int(self.cfg['width'])

        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*[2.0, 50.0, 1.0], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)


        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):

        B, N, _ = trans.shape

        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        final = final.permute(0, 1, 3,2)
        
        return final

  
    def forward(self, x, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask):


        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW).cuda()
        x ,feats = self.camencode(x)
        if self.cfg['model']['decoder']:
            out_x = self.decoder(feats)

        if self.cfg['model_conf']['Seg']:
            seg=self.seg_output(out_x[0])
            seg=seg.view(B, N, 6 , imH, imW)
        else:
            seg=None
        if self.cfg['model_conf']['Depth']:
            depth=torch.sigmoid(self.depth_output(out_x[0]))*100.0
            
            depth=depth.view(B, N, 1 , imH, imW)
        else:
            depth=None
        if self.cfg['model_conf']['od']:
            regression = self.regressor(out_x)
            classification = self.classifier(out_x)
            
            #classification = torch.cat([self.classifier(feature) for feature in out_x], dim=1)
        else:
            classification=None
            regression=None
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        
        x = x.permute(0, 1, 3, 4, 5, 2)

        if self.cfg['model']['bev_decoder']:
            geom = self.get_geometry(rots.cuda(), trans.cuda(), intrins.cuda(), post_rots.cuda(), post_trans.cuda())
            x = self.voxel_pooling(geom, x)




        if self.cfg['lidar'] and self.cfg['model']['bev_decoder']:
            lidar_feature = self.pp(lidar_data.cuda(), lidar_mask.cuda())
            topdown_feature = torch.cat([topdown_feature, lidar_feature], dim=1)
        if self.cfg['model']['bev_decoder']:
            DAS,Lane,Vehicle,walker=self.bevencode(topdown_feature)
        else:
            DAS=None
            Lane=None
            Vehicle=None
            walker=None


        return DAS,Lane,Vehicle,walker,seg,depth,classification, regression



def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p

class Hardswish(nn.Module):  
    @staticmethod
    def forward(x):

        return x * F.hardtanh(x + 3, 0., 6.) / 6.


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class YOLOPBackboneEncoder(nn.Module):
    def __init__(self):
        super(YOLOPBackboneEncoder, self).__init__()

        self.layer0 = Focus(3, 32, 3)                       # 0
        self.layer1 = Conv(32, 64, 3, 2)                    # 1
        self.layer2 = BottleneckCSP(64, 64, 1)              # 2
        self.layer3 = Conv(64, 128, 3, 2)                   # 3
        self.layer4 = BottleneckCSP(128, 128, 3)            # 4

        self.layer5 = Conv(128, 256, 3, 2)                  # 5
        self.layer6 = BottleneckCSP(256, 256, 3)            # 6

        self.layer7 = Conv(256, 512, 3, 2)                  # 7
        self.layer8 = SPP(512, 512, [5, 9, 13])             # 8

        self.layer9 = BottleneckCSP(512, 512, 1, False)     # 9
        self.layer10 = Conv(512, 256, 1, 1)                 # 10

        self.upsample11 = nn.Upsample(scale_factor=2, mode='nearest')  # 11
        self.layer13 = BottleneckCSP(512, 256, 1, False)    # 13
        
        self.layer14 = Conv(256, 128, 1, 1)                 # 14
        self.upsample15 = nn.Upsample(scale_factor=2, mode='nearest')  # 15
        self.layer17 = BottleneckCSP(256, 128, 1, False)    # 17

        # 18~23: Downsample + Concat + BottleneckCSP
        self.layer18 = Conv(128, 128, 3, 2)                 # 18
        self.layer20 = BottleneckCSP(256, 256, 1, False)    # 20
        self.layer21 = Conv(256, 256, 3, 2)                 # 21
        self.layer23 = BottleneckCSP(512, 512, 1, False)    # 23

    def forward(self, x):
        x0 = self.layer0(x)     # 0
        x1 = self.layer1(x0)    # 1
        x2 = self.layer2(x1)    # 2
        x3 = self.layer3(x2)    # 3
        x4 = self.layer4(x3)    # 4

        x5 = self.layer5(x4)    # 5
        x6 = self.layer6(x5)    # 6

        x7 = self.layer7(x6)    # 7
        x8 = self.layer8(x7)    # 8
        x9 = self.layer9(x8)    # 9
        x10 = self.layer10(x9)  # 10

        x11 = self.upsample11(x10)            # 11
        x12 = torch.cat([x11, x6], dim=1)     # 12 (Concat with x6)
        x13 = self.layer13(x12)               # 13
        x14 = self.layer14(x13)               # 14

        x15 = self.upsample15(x14)            # 15
        x16 = torch.cat([x15, x4], dim=1)     # 16 (Concat with x4)
        x17 = self.layer17(x16)               # 17

        x18 = self.layer18(x17)               # 18
        x19 = torch.cat([x18, x14], dim=1)    # 19 (Concat with x14)
        x20 = self.layer20(x19)               # 20

        x21 = self.layer21(x20)               # 21
        x22 = torch.cat([x21, x10], dim=1)    # 22 (Concat with x10)
        x23 = self.layer23(x22)               # 23

        return x16,x17, x20, x23  

class Detect(nn.Module):
    stride=[ 8., 16., 32.]

    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()

        self.nc = nc  
        self.no = nc + 5 
        self.nl = len(anchors)  
        self.na = len(anchors[0]) // 2  

        self.grid = [torch.zeros(1)] * self.nl  
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.anchors= a.cuda()
        self.anchor_grid= a.clone().view(self.nl, 1, -1, 1, 1, 2).cuda()

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch) 

    def forward(self, x):
        z = [] 
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  
            bs, _, ny, nx = x[i].shape  
            x[i]=x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            z.append(y.view(bs, -1, self.no))
        return (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class od_models(nn.Module):
    def __init__(self,cfg):
        super(od_models, self).__init__()
        self.nc = 2

        self.gr = 1.0
        self.encoder= YOLOPBackboneEncoder()
        self.detect=Detect(self.nc, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512])
        self.seg_head = nn.Sequential(
            Conv(256, 128, 3, 1),                         # 25
            nn.Upsample(scale_factor=2, mode='nearest'), # 26
            BottleneckCSP(128, 64, 1, False),             # 27
            Conv(64, 32, 3, 1),                           # 28
            nn.Upsample(scale_factor=2, mode='nearest'), # 29
            Conv(32, 16, 3, 1),                           # 30
            BottleneckCSP(16, 8, 1, False),               # 31
            nn.Upsample(scale_factor=2, mode='nearest'), # 32
            Conv(8, 6, 3, 1),                             # 33
        )

        # Lane line segmentation head
        self.depth_head = nn.Sequential(
            Conv(256, 128, 3, 1),                         # 34
            nn.Upsample(scale_factor=2, mode='nearest'), # 35
            BottleneckCSP(128, 64, 1, False),             # 36
            Conv(64, 32, 3, 1),                           # 37
            nn.Upsample(scale_factor=2, mode='nearest'), # 38
            Conv(32, 16, 3, 1),                           # 39
            BottleneckCSP(16, 8, 1, False),               # 40
            nn.Upsample(scale_factor=2, mode='nearest'), # 41
            Conv(8, 1, 3, 1),                             # 42
        )


    def forward(self, x, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W).cuda()

        x16,x17, x20, x23 = self.encoder(x)     # P3, P4, P5
        det_out = self.detect([x17, x20, x23])  # detection result

        
        seg_head = self.seg_head(x16)  
        depth = self.depth_head(x16)  


        return det_out, seg_head, torch.sigmoid(depth)*100.0






