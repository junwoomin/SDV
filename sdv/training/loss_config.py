"""Serializable loss contracts; safe to import in the web environment."""
import copy
import math

CHOICES={'Seg':['dice_bce','cross_entropy','dice_ce'], 'Depth':['silog','l1','smooth_l1'],
         'BEV':['bce','dice_bce'], 'od':['focal_smooth_l1','bce_smooth_l1']}
LABELS={'dice_bce':'Dice + BCE','cross_entropy':'Cross Entropy','dice_ce':'Dice + Cross Entropy',
        'silog':'SILog','l1':'L1 / MAE','smooth_l1':'Smooth L1 / Huber',
        'bce':'BCE with logits','focal_smooth_l1':'Focal + Smooth L1','bce_smooth_l1':'BCE + Smooth L1'}


def loss_defaults():
    return {'Seg':{'name':'dice_bce','weight':1.,'dice_weight':1.,'bce_weight':1.,'ce_weight':1.,'smooth':1.,'label_smoothing':0.},
            'Depth':{'name':'silog','weight':1.,'variance_focus':.85,'beta_m':1.},
            'BEV':{'name':'bce','weight':1.,'pos_weight':2.13,'dice_weight':1.,'bce_weight':1.,'smooth':1.},
            'od':{'name':'focal_smooth_l1','weight':1.,'alpha':.25,'gamma':2.,'beta':1/9,'classification_weight':1.,'regression_weight':1.}}


def normalize_losses(value=None):
    result=loss_defaults()
    if value is None:return result
    if not isinstance(value,dict) or set(value)-set(result):raise ValueError('지원하지 않는 loss 태스크입니다.')
    bounds={'weight':(1e-6,1000),'dice_weight':(0,100),'bce_weight':(0,100),'ce_weight':(0,100),
            'smooth':(1e-6,1000),'label_smoothing':(0,1),'variance_focus':(0,1),
            'beta_m':(1e-6,1000),'pos_weight':(1e-6,1000),'alpha':(0,1),'gamma':(0,10),
            'beta':(1e-6,100),'classification_weight':(1e-6,1000),'regression_weight':(1e-6,1000)}
    for task,patch in value.items():
        if not isinstance(patch,dict) or set(patch)-set(result[task]):raise ValueError(task+': 지원하지 않는 loss 파라미터입니다.')
        result[task].update(copy.deepcopy(patch))
    for task,options in result.items():
        if options['name'] not in CHOICES[task]:raise ValueError(task+': 출력/라벨 계약과 호환되지 않는 loss입니다.')
        for key,value in options.items():
            if key=='name':continue
            lo,hi=bounds[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError('{} loss {}: {}~{} 유한한 숫자가 필요합니다.'.format(task,key,lo,hi))
            options[key]=float(value)
        if options['name']=='dice_bce' and options['dice_weight']+options['bce_weight']==0:raise ValueError(task+': Dice/BCE 가중치를 모두 0으로 설정할 수 없습니다.')
        if options['name']=='dice_ce' and options['dice_weight']+options['ce_weight']==0:raise ValueError('Seg: Dice/CE 가중치를 모두 0으로 설정할 수 없습니다.')
        if options['name']=='bce' and options['bce_weight']==0:raise ValueError('BEV: BCE 가중치가 0입니다.')
    return result


def loss_catalog():
    return {'defaults':loss_defaults(),'choices':{k:[{'id':v,'label':LABELS[v]} for v in vs] for k,vs in CHOICES.items()},
            'contracts':{'Seg':'단일 클래스 one-hot / 제외 픽셀 마스킹, logits',
                         'Depth':'미터 단위, 유한한 양수 GT + validity mask',
                         'BEV':'서로 겹칠 수 있는 binary logits, NaN validity mask',
                         'od':'sigmoid anchor 분류 + encoded box regression; anchor assignment 유지'}}
