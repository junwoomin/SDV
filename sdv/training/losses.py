"""Task losses over the existing eight-output model contract."""
import torch
from torch import nn
from torch.nn import functional as F
from .loss_config import normalize_losses
from .modeling.loss import silog_loss, FocalLoss


class SegmentationLoss(nn.Module):
    def __init__(self,options):super().__init__();self.options=options
    def forward(self,pred,target):
        if pred.shape!=target.shape:raise ValueError('Seg loss shape mismatch')
        o=self.options;valid=torch.isfinite(target).all(dim=-3)&(target.sum(dim=-3)>0)
        if not bool(valid.any()):return pred.sum()*0
        if o['name']=='dice_bce':
            selected=valid.unsqueeze(-3).expand_as(target);p,t=pred[selected],target[selected]
            bce=F.binary_cross_entropy_with_logits(p,t);prob=p.sigmoid()
            dice=1-(2*(prob*t).sum()+o['smooth'])/(prob.sum()+t.sum()+o['smooth'])
            return o['weight']*(o['bce_weight']*bce+o['dice_weight']*dice)
        logits=pred.movedim(-3,-1)[valid];labels=target.argmax(dim=-3)[valid]
        ce=F.cross_entropy(logits,labels,label_smoothing=o['label_smoothing'])
        if o['name']=='cross_entropy':return o['weight']*ce
        probs=logits.softmax(dim=-1);truth=target.movedim(-3,-1)[valid]
        dice=1-((2*(probs*truth).sum(dim=0)+o['smooth'])/(probs.sum(dim=0)+truth.sum(dim=0)+o['smooth'])).mean()
        return o['weight']*(o['ce_weight']*ce+o['dice_weight']*dice)


class DepthLoss(nn.Module):
    def __init__(self,options):super().__init__();self.options=options;self.silog=silog_loss(options['variance_focus'])
    def forward(self,pred,target,mask):
        o=self.options;valid=mask.bool()&torch.isfinite(target)&(target>0)
        if not bool(valid.any()):return pred.sum()*0
        if o['name']=='silog':value=self.silog(pred,target,valid)
        elif o['name']=='l1':value=F.l1_loss(pred[valid],target[valid])
        else:value=F.smooth_l1_loss(pred[valid],target[valid],beta=o['beta_m'])
        return value*o['weight']


class BinaryBEVLoss(nn.Module):
    def __init__(self,options):
        super().__init__();self.options=options;self.register_buffer('pos_weight',torch.tensor([options['pos_weight']]))
    def forward(self,pred,target):
        if pred.shape!=target.shape:raise ValueError('BEV loss shape mismatch')
        valid=torch.isfinite(target);o=self.options
        if not bool(valid.any()):return pred.sum()*0
        p,t=pred[valid],target[valid]
        value=F.binary_cross_entropy_with_logits(p,t,pos_weight=self.pos_weight)*o['bce_weight']
        if o['name']=='dice_bce':
            prob=p.sigmoid();value+=o['dice_weight']*(1-(2*(prob*t).sum()+o['smooth'])/(prob.sum()+t.sum()+o['smooth']))
        return value*o['weight']


def build_losses(value=None):
    cfg=normalize_losses(value);od=cfg['od']
    return nn.ModuleDict({'Seg':SegmentationLoss(cfg['Seg']),'Depth':DepthLoss(cfg['Depth']),
        'BEV':BinaryBEVLoss(cfg['BEV']),
        'od':FocalLoss(alpha=od['alpha'],gamma=od['gamma'],beta=od['beta'],
            classification='focal' if od['name']=='focal_smooth_l1' else 'bce',
            classification_weight=od['classification_weight']*od['weight'],regression_weight=od['regression_weight']*od['weight'])})
