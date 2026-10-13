"""Validated, atomic, project-local configuration; no research imports at startup."""
from __future__ import annotations
import copy
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import yaml
from sdv.training.model_registry import BACKBONES, BEV_DECODERS, IMAGE_DECODERS
from sdv.data_contract import normalize_task_settings
from sdv.training.loss_config import loss_defaults, normalize_losses
from sdv.bev.schema import bev_settings_defaults, normalize_bev_settings

CAMERAS = ('front_cam', 'left_cam', 'right_cam', 'rear_cam', 'rear_left_cam', 'rear_right_cam')
TOWNS = ['Town01','Town02','Town03','Town04','Town05','Town06','Town07','Town10HD']

def integer(value, label, low, high):
    if isinstance(value, bool): raise ValueError(f'{label}: 정수를 입력하세요.')
    try:
        n = float(value)
        if not math.isfinite(n) or n != int(n) or not low <= n <= high: raise ValueError()
    except (ValueError,TypeError,OverflowError): raise ValueError(f'{label}: {low}~{high} 정수가 필요합니다.') from None
    return int(n)

def number(value, label, low, high):
    try:
        n=float(value)
        if isinstance(value,bool) or not math.isfinite(n) or not low<=n<=high: raise ValueError()
    except (ValueError,TypeError,OverflowError): raise ValueError(f'{label}: {low}~{high} 숫자가 필요합니다.') from None
    return n

def driving_defaults():
    return {'agent': 'roach', 'checkpoint': 'roach/log/ckpt_11833344.pth',
            'device': 'cpu', 'deterministic': True, 'policy_ppm': 5,
            'control_hz': 30, 'seed': 2026}


def validate_driving(value, fps=None):
    if not isinstance(value, dict):
        raise ValueError('Driving 설정은 JSON 객체여야 합니다.')
    unknown = set(value) - set(driving_defaults())
    if unknown:
        raise ValueError('지원하지 않는 Driving 설정: ' + ', '.join(sorted(unknown)))
    driving = {**driving_defaults(), **value}
    if driving['agent'] not in ('roach', 'existing', 'autopilot', 'manual'):
        raise ValueError('Driving Agent는 roach, existing, autopilot, manual 중 하나여야 합니다.')
    if driving['device'] not in ('cpu', 'cuda'):
        raise ValueError('Roach Device는 cpu 또는 cuda여야 합니다.')
    if type(driving['deterministic']) is not bool:
        raise ValueError('Deterministic Inference는 true/false여야 합니다.')
    driving['policy_ppm'] = integer(driving['policy_ppm'], 'Roach Policy PPM', 5, 5)
    driving['control_hz'] = integer(driving['control_hz'], 'Control Frequency', 1, 60)
    driving['seed'] = integer(driving['seed'], 'Traffic seed', 0, 2147483647)
    checkpoint = driving['checkpoint']
    if not isinstance(checkpoint, str) or not checkpoint.strip() or len(checkpoint) > 1024:
        raise ValueError('Roach Checkpoint 경로를 입력하세요.')
    driving['checkpoint'] = checkpoint.strip()
    if fps is not None and (driving['control_hz'] > fps or fps % driving['control_hz']):
        raise ValueError('Control Frequency는 시뮬레이션 FPS 이하이며 FPS를 정수로 나누어야 합니다.')
    if driving['agent'] == 'roach' and fps is not None and fps != 30:
        raise ValueError('기존 Roach History를 재현하려면 시뮬레이션 FPS를 30으로 설정하세요.')
    return driving


def atomic_json(path:Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)

class Config:
    def __init__(self,root=None):
        self.root=Path(root or os.environ.get('SDV_STUDIO_ROOT') or Path(__file__).resolve().parents[1]).resolve()
        self.state=self.root/'.runtime/studio';self.state.mkdir(parents=True,exist_ok=True)
        self._lock=threading.RLock();self.import_warnings=[]
        self.defaults={'project_name':'SDV Research','carla_install_path':'','carla_render_profile':'offscreen','carla_host':'127.0.0.1','carla_port':2000,'carla_tm_port':8000,'data_root':'data/collections','train_data':'data/train','val_data':'data/val','test_tensor':'img/test.pth','sensor_config':'data/sensor_config.yaml','results_root':'results','logs_root':'.runtime/studio/logs','python_executable':os.environ.get('SDV_RESEARCH_PYTHON',sys.executable),'vehicle':'vehicle.tesla.model3','weather':'cycle','npc_vehicles':4,'npc_walkers':2,'fps':30,'preview_fps':24,'driving':driving_defaults(),'bev_generator':bev_settings_defaults()}
    def path(self,value,allow_root=False):
        if not isinstance(value,(str,Path)) or not str(value).strip():raise ValueError('파일 경로가 비어 있습니다.')
        p=Path(value).expanduser();p=(p if p.is_absolute() else self.root/p).resolve()
        if (p==self.root and not allow_root) or (p!=self.root and self.root not in p.parents):raise ValueError('데이터·결과·로그 경로는 프로젝트 폴더 내부여야 합니다.')
        return p
    def read(self,name,default):
        p=self.state/(name+'.json')
        if not p.exists():return copy.deepcopy(default)
        try:
            result=json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(result,dict):raise ValueError('object required')
            return result
        except (ValueError,OSError) as e:raise ValueError(f'{p.name} 설정을 읽지 못했습니다: {e}') from e
    def settings(self):
        saved=self.read('settings',{})
        value={**copy.deepcopy(self.defaults),**saved}
        value['driving']={**driving_defaults(),**saved.get('driving',{})}
        if 'driving' not in saved:value['fps']=30
        bev=value.get('bev_generator')
        if isinstance(bev,dict) and bev.get('mode')=='height_aware' and 'global_ppm' not in bev:
            # Migrate the retired web default, not immutable running-job snapshots.
            # Static labels now come from the supplied whole-town PNG assets.
            bev=copy.deepcopy(bev)
            bev.update(mode='global_raster',layer_policy='hblr_ground_height')
            value['bev_generator']=normalize_bev_settings(bev)
        fps=value.get('preview_fps')
        # The old UI accepted fractional 0.1–2 FPS. Preserve saved integers;
        # map only now-unsupported legacy fractions to the new default.
        if isinstance(fps,(int,float)) and not isinstance(fps,bool) and .1<=fps<=2 and fps!=int(fps):
            value['preview_fps']=24
        return value
    def validate_settings(self,value,base=None):
        if not isinstance(value,dict):raise ValueError('설정은 JSON 객체여야 합니다.')
        unknown=set(value)-set(self.defaults)
        if unknown:raise ValueError('지원하지 않는 설정: '+', '.join(sorted(unknown)))
        s={**(self.settings() if base is None else base),**value}
        for key in ('data_root','train_data','val_data','test_tensor','results_root','logs_root','sensor_config'):self.path(s[key])
        from .carla_diagnostics import PROFILES
        if s['carla_render_profile'] not in PROFILES:raise ValueError('지원하지 않는 CARLA 렌더링 프로파일입니다.')
        if not isinstance(s['carla_install_path'],str):raise ValueError('CARLA 설치 경로는 문자열이어야 합니다.')
        if s['carla_install_path']:s['carla_install_path']=str(Path(s['carla_install_path']).expanduser().resolve())
        s['project_name']=str(s['project_name']).strip()[:100]
        if not s['project_name']:raise ValueError('프로젝트 이름을 입력하세요.')
        host=str(s['carla_host']).strip()
        if not host or len(host)>253 or not re.fullmatch(r'[a-zA-Z0-9.:%_-]+',host):raise ValueError('CARLA 서버 주소를 확인하세요. URL 대신 호스트 이름이나 IP를 입력하세요.')
        s['carla_host']=host
        for key in ('carla_port','carla_tm_port'):s[key]=integer(s[key],key,1,65535)
        if s['carla_port']==s['carla_tm_port']:raise ValueError('CARLA 포트와 Traffic Manager 포트는 달라야 합니다.')
        s['npc_vehicles']=integer(s['npc_vehicles'],'NPC 차량 수',0,200)
        s['npc_walkers']=integer(s['npc_walkers'],'보행자 수',0,200)
        s['fps']=integer(s['fps'],'시뮬레이션 FPS',1,60)
        s['preview_fps']=integer(s['preview_fps'],'프리뷰 FPS',1,60)
        s['driving']=validate_driving(s.get('driving',{}),s['fps'])
        self.path(s['driving']['checkpoint'])
        s['bev_generator']=normalize_bev_settings(s.get('bev_generator'))
        self.path(s['bev_generator']['map_root'])
        for key in ('vehicle','weather'):
            if not isinstance(s[key],str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}',s[key]):raise ValueError(f'{key} 값을 확인하세요.')
        executable=str(s['python_executable']).strip()
        candidate=Path(executable).expanduser() if '/' in executable else Path(shutil.which(executable) or executable)
        if not candidate.is_file() or not os.access(candidate,os.X_OK):raise ValueError('실행 가능한 Python 경로를 입력하세요.')
        s['python_executable']=os.path.abspath(candidate)
        return s
    def save_settings(self,value):
        with self._lock:
            s=self.validate_settings(value)
            atomic_json(self.state/'settings.json',s)
            return s
    def sensor_defaults(self):
        cam={'width':400,'height':300,'fov':100,'x':1.3,'y':0,'z':2.3,'pitch':0,'roll':0,'yaw':0,'seg':True,'depth':True,'od':True}
        return {'towns':['Town01'],'data_num':100,'sensors':{'front_cam':cam}}
    def _legacy(self,name,path,default,validate):
        if (self.state/(name+'.json')).exists():return self.read(name,default)
        source=self.root/path
        if source.is_file():
            try:return validate(yaml.safe_load(source.read_text(encoding='utf-8')))
            except (ValueError,TypeError,yaml.YAMLError) as exc:
                warning=f'기존 {path} 설정은 수정하지 않았습니다. 웹에서 확인이 필요합니다: {exc}'
                if warning not in self.import_warnings:self.import_warnings.append(warning)
        return copy.deepcopy(default)
    def sensors(self):return self._legacy('sensors','data/sensor_config.yaml',self.sensor_defaults(),self.validate_sensors)
    def validate_sensors(self,value):
        if not isinstance(value,dict):raise ValueError('센서 설정은 JSON 객체여야 합니다.')
        s=copy.deepcopy(value)
        if set(s)-{'towns','data_num','sensors'}:raise ValueError('센서 설정의 최상위 키는 towns, data_num, sensors입니다.')
        towns=s.get('towns')
        if not isinstance(towns,list) or not towns or any(t not in TOWNS for t in towns):raise ValueError('지원 타운을 하나 이상 선택하세요: '+', '.join(TOWNS))
        s['towns']=list(dict.fromkeys(towns));s['data_num']=integer(s.get('data_num'),'타운당 샘플 수',1,1000)
        sensors=s.get('sensors')
        if not isinstance(sensors,dict) or not any(c in sensors for c in (*CAMERAS,'front_cam_pv')):raise ValueError('카메라를 하나 이상 선택하세요.')
        bounds={'width':(64,1920),'height':(64,1080),'fov':(10,160),'x':(-10,10),'y':(-10,10),'z':(.1,10),'pitch':(-180,180),'yaw':(-360,360),'roll':(-180,180),'points_per_second':(100,200000),'rotation_frequency':(1,20),'channels':(1,128),'range':(1,150),'upper_fov':(-90,90),'lower_fov':(-90,90)}
        for name,fields in sensors.items():
            if name not in (*CAMERAS,'front_cam_pv','lidar') or not isinstance(fields,dict):raise ValueError('지원하지 않는 센서입니다: '+str(name))
            required={'x','y','z','pitch','yaw','roll'} | ({'width','height','fov','seg','depth','od'} if name!='lidar' else set())
            if required-set(fields):raise ValueError(f'{name}: 필수 항목 누락 '+', '.join(sorted(required-set(fields))))
            for key,v in fields.items():
                if key in {'seg','depth','od'}:
                    if type(v) is not bool:raise ValueError(f'{name}.{key}: true/false가 필요합니다.')
                elif key in bounds:
                    parse=integer if key in {'width','height','points_per_second','channels','rotation_frequency'} else number
                    fields[key]=parse(v,name+'.'+key,*bounds[key])
                else:raise ValueError('지원하지 않는 센서 설정: '+key)
            # Legacy CARLA validation accepts -180..180; normalize equivalent yaw.
            fields['yaw']=(fields['yaw']+180)%360-180
            if name=='lidar' and fields.get('upper_fov',10)<=fields.get('lower_fov',-30):
                raise ValueError('lidar.upper_fov는 lower_fov보다 커야 합니다.')
        return s
    def save_sensors(self,value):
        s=self.validate_sensors(value);atomic_json(self.state/'sensors.json',s);return s
    def training_defaults(self):
        return {'Epochs':20,'batch':2,'LR':.0001,'width':256,'height':256,'model':{'encoder':'ResNet','decoder':'FPN','bev_decoder':None},'model_conf':{'BEV':False,'Seg':True,'Depth':True,'od':False},'sensor_use':{c:c=='front_cam' for c in (*CAMERAS,'lidar')},'lidar':False,'opt_checkboxes':{'Adam':True,'AdamW':False,'SGD':False},'save_checkboxes':{'Last_Epoch':True,'Best_Loss':True},'save_other_checkboxes':{},'monitor':{'train_preview_every':100,'val_preview_every':1,'preview_samples':4,'train_preview_basis':'epoch','preview_max_mib':1024},'losses':loss_defaults()}
    def training(self):
        value=self._legacy('training','data/train_config.yaml',self.training_defaults(),self.validate_training)
        if isinstance(value.get('model'),dict) and value.get('model_conf',{}).get('BEV') is False:
            value['model']['bev_decoder']=None
        value.setdefault('monitor',copy.deepcopy(self.training_defaults()['monitor']))
        value.setdefault('losses',copy.deepcopy(self.training_defaults()['losses']))
        return value
    def validate_training(self,value):
        if not isinstance(value,dict):raise ValueError('학습 설정은 JSON 객체여야 합니다.')
        if set(value)-(set(self.training_defaults())|{'task_settings'}):raise ValueError('지원하지 않는 학습 설정 키가 있습니다.')
        c=copy.deepcopy(value)
        from sdv.training.loss_config import normalize_losses
        c['losses']=normalize_losses(c.get('losses'))
        monitor=c.setdefault('monitor',copy.deepcopy(self.training_defaults()['monitor']))
        if not isinstance(monitor,dict) or set(monitor)-{'train_preview_every','val_preview_every','preview_samples','train_preview_basis','preview_max_mib'}:raise ValueError('monitor는 예측 저장 주기 설정 객체여야 합니다.')
        monitor['train_preview_every']=integer(monitor.get('train_preview_every',100),'학습 예측 저장 주기 (Step, 0은 비활성화)',0,1000000)
        monitor['val_preview_every']=integer(monitor.get('val_preview_every',1),'검증 예측 저장 주기 (Epoch)',1,1000)
        monitor['preview_samples']=integer(monitor.get('preview_samples',4),'고정 예측 표본 수',4,4)
        if 'train_preview_basis' in monitor:
            if monitor['train_preview_basis'] not in ('epoch','global'):raise ValueError('TRAIN 기록 기준은 epoch 또는 global이어야 합니다.')
        if 'preview_max_mib' in monitor:
            monitor['preview_max_mib']=integer(monitor['preview_max_mib'],'예측 저장 최대 용량 (MiB)',1,102400)
        for key,low,high in [('Epochs',1,1000),('batch',1,64),('width',64,1920),('height',64,1080)]:c[key]=integer(c.get(key),key,low,high)
        c['LR']=number(c.get('LR'),'학습률',.000001,1)
        if c['width']%32 or c['height']%32:raise ValueError('학습 이미지 너비와 높이는 32의 배수여야 합니다.')
        model=c.get('model',{})
        if not isinstance(model,dict) or not isinstance(model.get('encoder'),str) or model['encoder'] not in BACKBONES:raise ValueError('지원되는 이미지 인코더를 선택하세요: '+', '.join(BACKBONES))
        if model.get('decoder') not in IMAGE_DECODERS:raise ValueError('지원되는 이미지 디코더를 선택하세요: '+', '.join(IMAGE_DECODERS))
        if 'pretrained' in model and type(model['pretrained']) is not bool:raise ValueError('model.pretrained는 true/false여야 합니다.')
        for key,allowed in [('model_conf',{'BEV','Seg','Depth','od'}),('sensor_use',set(CAMERAS)|{'lidar'}),('opt_checkboxes',{'Adam','AdamW','SGD'})]:
            obj=c.get(key)
            if not isinstance(obj,dict) or set(obj)-allowed or any(type(v) is not bool for v in obj.values()):raise ValueError(f'{key}는 지원 키와 true/false로 설정하세요.')
            for k in allowed:obj.setdefault(k,False)
        if c['model_conf']['od'] and (c['width']%128 or c['height']%128):raise ValueError('객체 탐지(OD)는 기존 Anchor 크기에 맞게 이미지 너비와 높이를 128의 배수로 지정하세요 (예: 128, 256, 384).')
        if c['model_conf']['BEV']:
            if model.get('bev_decoder') not in BEV_DECODERS:raise ValueError('BEV 태스크에는 BEV 디코더가 필요합니다: '+', '.join(BEV_DECODERS))
        else:model['bev_decoder']=None
        if not any(c['model_conf'].values()):raise ValueError('출력 태스크를 하나 이상 선택하세요.')
        # Absence is intentional: historical jobs retain their six-channel
        # checkpoint contract. Only explicit new drafts use discovered classes.
        if 'task_settings' in c:
            advanced=normalize_task_settings(c['task_settings'])
            if c['model_conf']['Seg'] and advanced['segmentation']['mode']=='carla' and not advanced['segmentation']['class_ids']:
                raise ValueError('Segmentation: 실제 데이터의 클래스를 하나 이상 선택하세요.')
            if c['model_conf']['od'] and not advanced['detection']['class_ids']:
                raise ValueError('Object Detection: 실제 Annotation에 있는 클래스를 하나 이상 선택하세요.')
            if c['model_conf']['BEV'] and not advanced['bev']['layers']:
                raise ValueError('BEV: 학습할 레이어를 하나 이상 선택하세요.')
            c['task_settings']=advanced
        if sum(c['opt_checkboxes'].values())!=1:raise ValueError('옵티마이저를 하나 선택하세요.')
        if not c['sensor_use'].get('front_cam'):raise ValueError('현재 학습 데이터 로더는 front_cam을 필수로 사용합니다.')
        if type(c.get('lidar')) is not bool:raise ValueError('lidar는 true/false여야 합니다.')
        if c['lidar']!=c['sensor_use']['lidar']:raise ValueError('lidar와 sensor_use.lidar 값이 같아야 합니다.')
        for key in ('save_checkboxes','save_other_checkboxes'):
            if not isinstance(c.get(key),dict) or any(type(v) is not bool for v in c[key].values()):raise ValueError(f'{key}는 true/false 설정이어야 합니다.')
        if any(v for k,v in c['save_checkboxes'].items() if k not in {'Last_Epoch','Best_Loss'}):raise ValueError('현재 체크포인트 정책은 Last_Epoch와 Best_Loss만 지원합니다.')
        if any(c['save_other_checkboxes'].values()):raise ValueError('추가 모델 형식 내보내기는 원본 연구 코드에서 지원하지 않습니다.')
        return c
    def save_training(self,value):
        c=self.validate_training(value);atomic_json(self.state/'training.json',c);return c
    def yaml(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(value,allow_unicode=True,sort_keys=False),encoding='utf-8')
