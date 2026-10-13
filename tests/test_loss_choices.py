import copy
import pytest
import torch
from sdv.training.loss_config import normalize_losses, loss_defaults
from sdv.training.losses import build_losses
from sdv.training.modeling.loss import Dice_BEC_Loss, SimpleLoss, silog_loss


def test_default_objectives_preserve_existing_formulas():
    losses=build_losses(None)
    pred=torch.randn(2,1,3,4,4,requires_grad=True);gt=torch.zeros_like(pred);gt[:,:,0,:2]=1
    assert torch.allclose(losses['Seg'](pred,gt),Dice_BEC_Loss()(pred,gt))
    depth=torch.full((2,1,1,4,4),4.,requires_grad=True);target=torch.full_like(depth,5.);mask=torch.ones_like(depth,dtype=torch.bool)
    assert torch.allclose(losses['Depth'](depth,target,mask),silog_loss(.85)(depth,target,mask))
    bev=torch.randn(2,1,4,4);target=torch.ones_like(bev);target[...,0]=float('nan')
    assert torch.allclose(losses['BEV'](bev,target),SimpleLoss(2.13)(bev,target))


@pytest.mark.parametrize('name',['dice_bce','cross_entropy','dice_ce'])
def test_seg_ignore_mask_and_backward(name):
    cfg=loss_defaults();cfg['Seg']['name']=name;loss=build_losses(cfg)['Seg']
    pred=torch.randn(2,1,3,4,4,requires_grad=True);gt=torch.zeros_like(pred);gt[:,:,1,:2]=1
    value=loss(pred,gt);assert torch.isfinite(value);value.backward()
    assert pred.grad[:,:,:,2:].count_nonzero()==0
    assert pred.grad[:,:,:,:2].count_nonzero()>0


@pytest.mark.parametrize('name',['silog','l1','smooth_l1'])
def test_depth_invalid_mask_and_empty_mask(name):
    cfg=loss_defaults();cfg['Depth']['name']=name;loss=build_losses(cfg)['Depth']
    pred=torch.ones(1,1,1,2,2,requires_grad=True);gt=torch.tensor([[[[[2.,float('nan')],[0.,-1.]]]]]);mask=torch.ones_like(gt,dtype=torch.bool)
    value=loss(pred,gt,mask);assert torch.isfinite(value);value.backward()
    assert pred.grad.count_nonzero()==1
    assert loss(pred,gt,torch.zeros_like(mask)).item()==0


def test_unsupported_contract_and_bad_parameters_rejected():
    with pytest.raises(ValueError):normalize_losses({'BEV':{'name':'cross_entropy'}})
    with pytest.raises(ValueError):normalize_losses({'Depth':{'beta_m':0}})
    with pytest.raises(ValueError):normalize_losses({'Seg':{'label_smoothing':1.2}})
    with pytest.raises(ValueError):normalize_losses({'od':{'unknown':2}})


@pytest.mark.parametrize('name',['focal_smooth_l1','bce_smooth_l1'])
def test_od_positive_anchor_and_empty_background_gradient(name):
    cfg=loss_defaults();cfg['od']['name']=name;cfg['od']['beta']=.5
    loss=build_losses(cfg)['od']
    pred=torch.full((1,2,2),.4,requires_grad=True);reg=torch.ones(1,2,4,requires_grad=True)*.3
    anchors=torch.tensor([[[0.,0.,10.,10.],[30.,30.,40.,40.]]]);target=torch.tensor([[[0.,0.,10.,10.,1.]]])
    cls,box,*_=loss(pred,reg,anchors,target);assert torch.isfinite(cls+box);(cls+box).backward()
    assert pred.grad.count_nonzero()>0 and box.item()>0
    background=loss(pred,reg,anchors,torch.zeros(1,0,5))
    assert background[0]>0 and background[1]==0
