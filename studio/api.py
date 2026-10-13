"""Local HTTP API. No mock data, arbitrary command execution, or heavy imports."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import hmac
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

import psutil
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import Config, TOWNS, atomic_json, validate_driving, number, integer
from .jobs import JobManager, now
from .catalog import Catalog
from .compat import to_thread
from .carla_service import CarlaService
from .sensor_preview_service import SensorPreviewService
from .pretrained_service import PretrainedService
from .training_options import training_options
from .predictions import prediction_index, prediction_frame
from .sequences import SequenceCatalog
from .task_catalog import task_catalog, selection_blockers
from sdv.training.model_registry import BACKBONES, IMAGE_DECODERS, BEV_DECODERS, get_backbone, model_catalog
from sdv.training.metric_registry import metric_registry

DEPENDENCY_SCRIPT=r'''
import importlib.util,json
names=['carla','numpy','yaml','cv2','torch','torchvision','pandas','matplotlib','scipy','shapely','pyquaternion','efficientnet_pytorch','PIL','pygame','networkx','torch_scatter']
r={'dependencies':{}}
for name in names:
 try:r['dependencies'][name]={'available':importlib.util.find_spec(name) is not None}
 except Exception as e:r['dependencies'][name]={'available':False,'error':str(e)}
r['cuda_available']=False
if r['dependencies']['torch']['available']:
 try:
  import torch
  r['cuda_available']=torch.cuda.is_available();r['torch_version']=torch.__version__
  r['cuda_devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
 except Exception as e:r['dependencies']['torch']['error']=str(e)
if r['dependencies']['torchvision']['available']:
 try:
  import torchvision
  r['dependencies']['torchvision']['version']=torchvision.__version__
  from sdv.training.model_registry import BACKBONES
  r['model_builders']={spec['constructor']:callable(getattr(torchvision.models,spec['constructor'],None)) for spec in BACKBONES.values() if spec['provider']=='torchvision'}
 except Exception as e:r['dependencies']['torchvision']['error']=str(e)
print(json.dumps(r))
'''
CARLA_SCRIPT=r'''
import json,sys
try:
 import carla
 c=carla.Client(sys.argv[1],int(sys.argv[2]));c.set_timeout(3.0);w=c.get_world()
 print(json.dumps({'connected':True,'map':w.get_map().name,'version':c.get_server_version(),'maps':c.get_available_maps(),'vehicles':[b.id for b in w.get_blueprint_library().filter('vehicle.*')]}))
except Exception as e:
 print(json.dumps({'connected':False,'error':str(e)}));sys.exit(1)
'''

class Runtime:
    def __init__(self,config,jobs):
        self.config=config;self.jobs=jobs;self._deps=None;self._deps_key=None;self._deps_at=0;self._lock=threading.Lock();self.carla={'connected':False,'error':'CARLA에 연결하지 않았습니다.','checked_at':None};self._resource_cache=(0,{});self._carla_enabled=False;self._carla_probing=False;self._carla_generation=0;self._manual_lock=threading.Lock()
    def capabilities(self,refresh=False,settings_override=None):
        s=settings_override if settings_override is not None else self.config.settings();key=s['python_executable']
        with self._lock:
            if not refresh and self._deps and key==self._deps_key and time.monotonic()-self._deps_at<30:return self._deps
            try:
                proc=subprocess.run([key,'-c',DEPENDENCY_SCRIPT],cwd=self.config.root,capture_output=True,text=True,timeout=15)
                lines=proc.stdout.strip().splitlines();result=json.loads(lines[-1]) if lines else {}
                if not isinstance(result,dict) or 'dependencies' not in result:raise ValueError(proc.stderr[-1500:] or 'Python dependency probe failed')
            except (OSError,subprocess.TimeoutExpired,ValueError) as exc:result={'dependencies':{},'cuda_available':False,'error':'연구 Python 확인 실패: '+str(exc)}
            result.update(encoders=list(BACKBONES),decoders=list(IMAGE_DECODERS),bev_decoders=list(BEV_DECODERS),towns=TOWNS,training_supported=True,reinforcement_supported=False,warnings=list(self.config.import_warnings))
            self._deps=result;self._deps_key=key;self._deps_at=time.monotonic();return result
    def model_catalog(self,capabilities=None):
        deps=capabilities if capabilities is not None else self.capabilities()
        available=deps.get('dependencies',{});builders=deps.get('model_builders',{})
        catalog=model_catalog()
        for spec in catalog['encoders']:
            dependency='torchvision' if spec['provider']=='torchvision' else 'efficientnet_pytorch'
            item=available.get(dependency,{})
            reason=None;reason_en=None
            if not item.get('available') or item.get('error'):
                reason='연구 Python의 {} 설치 상태를 확인하세요.'.format(dependency)
                reason_en='Check {} in the research Python environment.'.format(dependency)
            elif spec['provider']=='torchvision' and spec['constructor'] in builders and not builders[spec['constructor']]:
                minimum=spec['minimum_torchvision']
                requirement='torchvision {} 이상'.format(minimum) if minimum else 'torchvision의 {} 모델'.format(spec['constructor'])
                requirement_en='torchvision {} or later'.format(minimum) if minimum else 'the torchvision {} builder'.format(spec['constructor'])
                reason='{}에는 {}이 필요합니다. 현재 버전: {}'.format(spec['label'],requirement,item.get('version','?'))
                reason_en='{} requires {}. Installed: {}'.format(spec['label'],requirement_en,item.get('version','?'))
            spec.update(available=reason is None,reason=reason,reason_en=reason_en)
        legacy=available.get('efficientnet_pytorch',{})
        for spec in catalog['bev_decoders']:
            missing=spec['id']=='EfficientNet' and (not legacy.get('available') or legacy.get('error'))
            spec.update(available=not missing,
                        reason='이 BEV 디코더에는 연구 Python의 efficientnet-pytorch가 필요합니다.' if missing else None,
                        reason_en='This BEV decoder requires efficientnet-pytorch in the research Python environment.' if missing else None)
        return catalog
    def probe_carla(self):
        generation=self._carla_generation
        self._carla_enabled=True
        self._carla_probing=True
        s=self.config.settings()
        try:
            proc=subprocess.run([s['python_executable'],'-c',CARLA_SCRIPT,s['carla_host'],str(s['carla_port'])],cwd=self.config.root,capture_output=True,text=True,timeout=7)
            lines=proc.stdout.strip().splitlines();result=json.loads(lines[-1]) if lines else {'connected':False,'error':proc.stderr[-1500:] or 'CARLA 응답이 없습니다.'}
        except (OSError,subprocess.TimeoutExpired,ValueError) as exc:result={'connected':False,'error':str(exc)}
        result.update(host=s['carla_host'],port=s['carla_port'],checked_at=now(),checked_unix=time.time())
        if not result.get('connected'):result['hint']='CARLA 서버 실행 상태, 주소/포트, 연구 Python의 carla 패키지 버전을 확인하세요.'
        if generation==self._carla_generation:self.carla=result
        self._carla_probing=False
        return result
    def carla_state(self):
        result=dict(self.carla)
        age=time.time()-result.get('checked_unix',0)
        if self._carla_enabled and age>10 and not self._carla_probing:
            self._carla_probing=True
            threading.Thread(target=self.probe_carla,daemon=True).start()
        if age>20 and self._carla_enabled:result.update(connected=False,stale=True,error='CARLA 상태를 다시 확인하는 중입니다.')
        result['checking']=self._carla_probing
        s=self.config.settings();result.setdefault('host',s['carla_host']);result.setdefault('port',s['carla_port']);return result
    def resources(self):
        if time.monotonic()-self._resource_cache[0]<2:return self._resource_cache[1]
        memory=psutil.virtual_memory();disk=psutil.disk_usage(self.config.root)
        result={'cpu_percent':psutil.cpu_percent(interval=None),'memory_percent':memory.percent,'memory_used_gb':round(memory.used/1024**3,2),'memory_total_gb':round(memory.total/1024**3,2),'disk_free_gb':round(disk.free/1024**3,2),'gpus':[],'gpu_error':None}
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
            if p.returncode:result['gpu_error']='NVIDIA GPU 정보를 읽을 수 없습니다.'
            else:
                for line in p.stdout.strip().splitlines():
                    name,use,used,total=[x.strip() for x in line.split(',')]
                    result['gpus'].append({'name':name,'utilization':float(use),'memory_used_mb':float(used),'memory_total_mb':float(total)})
        except (OSError,subprocess.TimeoutExpired,ValueError):result['gpu_error']='nvidia-smi 또는 NVIDIA GPU를 사용할 수 없습니다.'
        self._resource_cache=(time.monotonic(),result);return result
    def preflight(self,kind,override=None,prepare_weights=False,settings_override=None,allow_busy=False):
        blockers=[];s=settings_override if settings_override is not None else self.config.settings()
        deps=self.capabilities(settings_override=s) if settings_override is not None else self.capabilities();available=deps.get('dependencies',{})
        pretrained=None
        if kind=='training':
            draft=override if isinstance(override,dict) else self.config.training()
            encoder=draft.get('model',{}).get('encoder') if isinstance(draft.get('model'),dict) else None
            preparation=getattr(self,'pretrained',None)
            if preparation and encoder in BACKBONES and draft.get('model',{}).get('pretrained',True):
                pretrained=preparation.ensure(encoder) if prepare_weights else preparation.status(encoder)
        local=getattr(self,'local_carla',None)
        if local and local.busy():blockers.append('CARLA 설치·시작 작업이 진행 중입니다. 준비가 끝난 뒤 실행하세요.')
        preview=getattr(self,'sensor_preview',None)
        if preview and preview.busy():blockers.append('센서 미리보기를 중지한 뒤 연구 작업을 시작하세요.')
        if not allow_busy and self.jobs.active():blockers.append('이미 실행 중인 작업이 있습니다. 먼저 완료하거나 중지하세요.')
        if deps.get('error'):blockers.append(deps['error'])
        try:
            if kind in ('collection','sensor_preview'):
                c=self.config.validate_sensors(override if override is not None else self.config.sensors())
                required=['carla','numpy','PIL'] if kind=='sensor_preview' else ['carla','numpy','yaml','cv2','scipy','shapely','torch','pygame','PIL','networkx']
                status=self.carla_state()
                if not status.get('connected'):blockers.append('CARLA 연결 확인을 먼저 실행하세요. '+status.get('error',''))
                elif kind=='collection' and status.get('maps'):
                    maps=[x.split('/')[-1] for x in status['maps']]
                    for town in c['towns']:
                        if town not in maps:blockers.append('CARLA 서버에 타운이 없습니다: '+town)
                if status.get('vehicles') and s['vehicle'] not in status['vehicles']:blockers.append('CARLA 서버에서 차량 blueprint를 찾지 못했습니다: '+s['vehicle'])
                if kind=='collection':
                    driving=validate_driving(s.get('driving',{}),s['fps'])
                    if driving['agent']=='roach':
                        checkpoint=self.config.path(driving['checkpoint'])
                        if not checkpoint.is_file():
                            blockers.append('Roach Checkpoint가 없습니다: {}. 원본 ckpt_11833344.pth를 배치하세요. 임의 가중치 주행은 허용하지 않습니다.'.format(checkpoint))
                        if driving['device']=='cuda' and not deps.get('cuda_available'):
                            blockers.append('Roach CUDA Device를 사용할 수 없습니다. CPU를 선택하거나 연구 Python의 CUDA 환경을 확인하세요.')
                bev=s.get('bev_generator',{})
                if kind=='collection' and bev.get('mode')=='global_raster':
                    from sdv.bev.global_map import load_global_map
                    map_root=self.config.path(bev['map_root'])
                    for town in c['towns']:
                        try:load_global_map(map_root,town,bev['global_ppm'])
                        except (OSError,ValueError,KeyError) as exc:
                            blockers.append('Global BEV PNG 준비 실패: {} / {} PPM. 지도 폴더 {} 및 catalog.json을 확인하세요. {}'.format(town,bev['global_ppm'],map_root,exc))
            elif kind=='training':
                c=self.config.validate_training(override if override is not None else self.config.training())
                required=['torch','torchvision','numpy','yaml','cv2','pandas','matplotlib','scipy','pyquaternion','PIL']
                backbone=get_backbone(c['model']['encoder'])
                if backbone['provider']=='efficientnet_pytorch' or c['model'].get('bev_decoder')=='EfficientNet':required.append('efficientnet_pytorch')
                builders=deps.get('model_builders',{})
                if backbone['provider']=='torchvision' and backbone['constructor'] in builders and not builders[backbone['constructor']]:
                    unsupported=next(row for row in self.model_catalog(deps)['encoders'] if row['id']==backbone['id'])
                    blockers.append(unsupported['reason'])
                if c.get('lidar'):required.append('torch_scatter')
                if not deps.get('cuda_available'):blockers.append('원본 학습 코드는 CUDA GPU가 필요합니다. 연구 Python의 PyTorch/CUDA 환경을 확인하세요.')
                for key,label in [('train_data','학습'),('val_data','검증')]:
                    p=self.config.path(s[key])
                    if not p.is_dir() or next(p.rglob('*.png'),None) is None:blockers.append(f'{label} 데이터 PNG가 없습니다: {s[key]}')
                if self.config.path(s['train_data'])==self.config.path(s['val_data']):blockers.append('학습과 검증 데이터 경로는 달라야 합니다.')
                # Reuse the actual dataset compatibility validator when supplied.
                try:
                    from .dataset_tools import validate_training_inputs
                except ImportError:pass
                else:
                    blockers.extend(validate_training_inputs(self.config.root,s,c))
                    blockers.extend(selection_blockers(self.config.root,s,c))
            else:return {'ready':False,'blockers':['지원 작업: collection, training'],'kind':kind}
            for name in required:
                item=available.get(name,{})
                if not item.get('available') or item.get('error'):blockers.append('연구 Python 종속성이 없거나 불완전합니다: '+name)
        except (ValueError,TypeError,KeyError) as exc:blockers.append(str(exc))
        if pretrained and not pretrained.get('ready'):
            if pretrained.get('status')=='Running':blockers.append('사전학습 가중치를 준비 중입니다. 다운로드와 검증이 끝나면 다시 확인합니다: '+pretrained['filename'])
            elif pretrained.get('error'):blockers.append('사전학습 가중치 준비 실패: '+pretrained['error'])
            else:blockers.append('준비 상태 검사를 누르면 필요한 사전학습 가중치를 자동으로 다운로드합니다: '+pretrained['filename'])
        result={'ready':not blockers,'blockers':list(dict.fromkeys(blockers)),'kind':kind}
        if pretrained:result['pretrained']=pretrained
        return result

def create_app(root=None):
    config=Config(root);jobs=JobManager(config);runtime=Runtime(config,jobs);catalog=Catalog(config.root,config.settings)
    sequences=SequenceCatalog(catalog)
    local_carla=CarlaService(config);runtime.local_carla=local_carla;runtime._local_task_id=None;runtime._local_client_result=None
    sensor_preview=SensorPreviewService(config);runtime.sensor_preview=sensor_preview
    pretrained=PretrainedService(config);runtime.pretrained=pretrained
    def queue_resources(job):
        blockers=[]
        if local_carla.busy():blockers.append('CARLA 설치·시작 작업이 진행 중입니다. 준비가 끝난 뒤 실행하세요.')
        if sensor_preview.busy():blockers.append('센서 미리보기를 중지한 뒤 연구 작업을 시작하세요.')
        return {'ready':not blockers,'blockers':blockers}
    jobs.set_training_gate(lambda job:runtime.preflight('training',job['config'],settings_override=job['settings']),resource_gate=queue_resources)
    @asynccontextmanager
    async def lifespan(app):
        yield
        await to_thread(sensor_preview.shutdown)
        await to_thread(jobs.shutdown)
        await to_thread(local_carla.shutdown)
        await to_thread(pretrained.shutdown)
    app=FastAPI(title='SDV Studio',version='1.0.0',lifespan=lifespan)
    app.state.config=config;app.state.jobs=jobs;app.state.runtime=runtime;app.state.catalog=catalog;app.state.local_carla=local_carla
    app.state.sensor_preview=sensor_preview
    app.state.pretrained=pretrained
    app.state.sequences=sequences
    @app.middleware('http')
    async def local_access(request,call_next):
        token=os.environ.get('SDV_AUTH_TOKEN','')
        try:host=urlsplit('http://'+request.headers.get('host','')).hostname
        except ValueError:host=None
        allowed=os.environ.get('SDV_ALLOWED_HOSTS','127.0.0.1,localhost,::1,testserver').split(',')
        if host not in allowed and not token:return JSONResponse({'detail':'허용되지 않은 Host입니다.'},status_code=403)
        if request.url.path.startswith('/api/') and token:
            supplied=request.headers.get('authorization','')
            if supplied.startswith('Bearer '):supplied=supplied[len('Bearer '):]
            if not hmac.compare_digest(supplied,token):return JSONResponse({'detail':'인증 토큰이 필요합니다.'},status_code=401)
        if request.method in {'POST','PUT','PATCH','DELETE'}:
            if request.headers.get('x-sdv-request')!='1':return JSONResponse({'detail':'X-SDV-Request 헤더가 필요합니다.'},status_code=403)
            origin=request.headers.get('origin');fetch_site=request.headers.get('sec-fetch-site','')
            if fetch_site=='cross-site':return JSONResponse({'detail':'다른 사이트의 변경 요청은 허용하지 않습니다.'},status_code=403)
            if origin:
                source=urlsplit(origin)
                if source.netloc!=request.headers.get('host') or source.scheme not in {'http','https'}:return JSONResponse({'detail':'동일한 SDV 사이트에서 요청하세요.'},status_code=403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='same-origin';response.headers['X-Frame-Options']='DENY'
        if request.url.path.startswith('/api/') or request.url.path=='/' or (request.url.path.startswith('/static/') and request.url.path.endswith(('.js','.css','.html'))):
            # A restarted installation must serve its matching UI and module
            # graph; cached old FPS constraints are misleading after upgrades.
            response.headers['Cache-Control']='no-store'
        return response
    @app.exception_handler(ValueError)
    async def invalid(request,exc):return JSONResponse({'detail':str(exc)},status_code=422)
    @app.exception_handler(KeyError)
    async def missing(request,exc):return JSONResponse({'detail':'요청한 항목을 찾을 수 없습니다.'},status_code=404)
    @app.exception_handler(FileNotFoundError)
    async def no_file(request,exc):return JSONResponse({'detail':'아직 저장된 파일이 없습니다.'},status_code=404)
    def summaries():return jobs.summaries()
    def status():
        local=local_carla.status();engine=local.get('engine',{});task=local.get('task',{})
        if task.get('operation')=='prepare' and task.get('status') in {'Completed','Failed','Stopped'}:
            result_key=(task.get('id'),task.get('status'))
            with runtime._lock:
                if result_key!=runtime._local_client_result:
                    # Client installation can succeed even when later engine startup fails.
                    runtime._local_client_result=result_key;runtime._deps=None;runtime._deps_at=0
        if engine.get('ready') and task.get('id')!=runtime._local_task_id:
            runtime._local_task_id=task.get('id');runtime._carla_enabled=True
            if not runtime._carla_probing:
                runtime._carla_probing=True
                threading.Thread(target=runtime.probe_carla,daemon=True).start()
        if runtime._local_task_id and not engine.get('running') and runtime.carla.get('connected'):
            runtime._carla_generation+=1;runtime._carla_enabled=False
            runtime.carla.update(connected=False,error=local.get('task',{}).get('message') or '로컬 CARLA 엔진이 종료되었습니다. 엔진 로그를 확인하세요.')
            runtime._local_task_id=None
        all_jobs=summaries()
        collection=next((j for j in all_jobs if j['kind']=='collection' and j['status'] in ('Running','Pausing')),None)
        if collection and engine.get('state')=='Crashed':
            jobs.stop(collection['id'])
        elif collection and engine.get('owned'):
            from .carla_service import update_state
            stage=(collection.get('last_event') or {}).get('stage') or ('Running' if collection.get('progress_current',0)>0 else 'Loading Map')
            update_state(local_carla.directory,local['task']['id'],engine={'state':'Loading Map' if stage=='Loading Map' else 'Running','last_stage':stage})
        if not collection and engine.get('state')=='Running' and engine.get('running') and engine.get('owned'):
            from .carla_service import update_state
            update_state(local_carla.directory,local['task']['id'],engine={'state':'Ready','last_stage':'Collection completed / idle'})
            engine['state']='Ready'
        return {'carla':runtime.carla_state(),'carla_local':local,'sensor_preview':sensor_preview.status(),'pretrained':pretrained.status(),'resources':runtime.resources(),'jobs':all_jobs,'active_job':next((j for j in all_jobs if j['status'] in {'Running','Pausing'}),None)}
    @app.get('/api/bootstrap')
    def bootstrap():
        sensors=config.sensors();training=config.training()
        capabilities=runtime.capabilities()
        return {**status(),'settings':config.settings(),'sensors':sensors,'training':training,'training_options':training_options(config.root,config.settings(),training.get('sensor_use'),training.get('task_settings')),'capabilities':capabilities,'model_catalog':runtime.model_catalog(capabilities),'metric_registry':metric_registry(training),'experiments':catalog.experiments(),'datasets':catalog.datasets(),'warnings':list(config.import_warnings)}
    @app.get('/api/training/datasets')
    def recognized_datasets():
        from .dataset_choices import dataset_choices
        return dataset_choices(config.root,config.settings())
    @app.get('/api/training/losses')
    def training_losses():
        from sdv.training.loss_config import loss_catalog
        return loss_catalog()
    @app.get('/api/training/metrics')
    def training_metrics():return metric_registry(config.training())
    @app.get('/api/training/models')
    def training_models():return runtime.model_catalog()
    @app.get('/api/status')
    def get_status():return status()
    @app.get('/api/settings')
    def settings():return config.settings()
    @app.get('/api/driving/checkpoints')
    def driving_checkpoints():
        selected=config.settings()['driving']['checkpoint']
        candidates={selected}
        for directory in ('roach/log','checkpoints/roach'):
            folder=config.path(directory)
            if folder.is_dir():
                for path in folder.glob('*.pth'):
                    if len(candidates)>=250:break
                    if config.root in path.resolve().parents:candidates.add(str(path.relative_to(config.root)))
        return {'checkpoints':[{'path':name,'exists':config.path(name).is_file(),
                'bytes':config.path(name).stat().st_size if config.path(name).is_file() else None}
                for name in sorted(candidates)],'selected':selected}
    @app.post('/api/driving/inspect')
    def inspect_driving(value:dict=Body(...)):
        if set(value)-{'checkpoint','device','deterministic'}:
            raise ValueError('Checkpoint 검사에는 checkpoint, device, deterministic만 지정하세요.')
        driving=validate_driving({**config.settings()['driving'],**value})
        checkpoint=config.path(driving['checkpoint'])
        if not checkpoint.is_file():
            raise HTTPException(409,'Roach Checkpoint가 없습니다: {}. 원본 가중치를 배치하세요. 임의 가중치로 대체하지 않습니다.'.format(checkpoint))
        if jobs.active():raise HTTPException(409,'실행 중인 작업을 중지한 뒤 별도 모델 로딩 검사를 실행하세요.')
        script=('import json,sys; from sdv.driving.policy import inspect_checkpoint; '
                'print(json.dumps(inspect_checkpoint(sys.argv[1],device=sys.argv[2],'
                'deterministic=sys.argv[3]=="true",observation_profile="lrs_loop")))')
        try:
            proc=subprocess.run([config.settings()['python_executable'],'-c',script,str(checkpoint),
                driving['device'],str(driving['deterministic']).lower()],cwd=config.root,
                capture_output=True,text=True,timeout=60)
            if proc.returncode:
                raise ValueError((proc.stderr or proc.stdout)[-2000:] or 'Roach 모델 로딩에 실패했습니다.')
            result=json.loads(proc.stdout.strip().splitlines()[-1])
            return {**result,'checkpoint':str(checkpoint),'loaded_for_validation':True,
                    'worker_model_loaded':False,'message':'실제 모델 로딩 검사를 통과했습니다. 수집 Worker는 시작 시 다시 로딩합니다.'}
        except (OSError,ValueError,IndexError,subprocess.TimeoutExpired) as exc:
            raise HTTPException(409,'Roach 모델 확인 실패: '+str(exc)) from exc
    @app.get('/api/bev/maps')
    def bev_map_catalog(root:str=None):
        from sdv.bev.global_map import catalog as global_map_catalog
        settings=config.settings()
        requested=root if root is not None else settings['bev_generator']['map_root']
        result=global_map_catalog(config.path(requested))
        result['selected']={'towns':config.sensors().get('towns',[]),
                            'global_ppm':settings['bev_generator']['global_ppm']}
        return result
    @app.put('/api/settings')
    def save_settings(value:dict=Body(...)):
        previous=config.settings()
        changed={k for k,v in value.items() if previous.get(k)!=v}
        active=jobs.active()
        if changed<= {'train_data','val_data','sensor_config','test_tensor'} and (not active or active['kind']=='training'):
            # These defaults never alter a job's immutable snapshots. They can
            # be saved even while the dispatcher holds its brief launch lock.
            result=config.save_settings(value);catalog.invalidate()
            return result
        with jobs.resource_guard():
            previous=config.settings()
            active=jobs.active()
            changed={k for k,v in value.items() if previous.get(k)!=v}
            if active and (active['kind']!='training' or changed-{'train_data','val_data','sensor_config','test_tensor'}):
                raise HTTPException(409,'실행 중에는 다음 학습의 데이터·센서 경로만 바꿀 수 있습니다. 시스템 설정은 작업을 완료하거나 중지한 뒤 저장하세요.')
            connection_changed=any(k in value and str(value[k])!=str(previous.get(k)) for k in ('carla_host','carla_port','carla_install_path'))
            if 'python_executable' in value:
                def executable_path(raw):
                    path=os.path.expanduser(str(raw).strip())
                    return os.path.abspath(path if '/' in path else shutil.which(path) or path)
                connection_changed=connection_changed or executable_path(value['python_executable'])!=executable_path(previous['python_executable'])
            local=local_carla.status()
            if connection_changed and sensor_preview.busy():
                raise HTTPException(409,'센서 미리보기를 중지한 뒤 서버 주소·포트·연구 Python을 변경하세요.')
            if connection_changed and (local_carla.busy() or local.get('engine',{}).get('running')):
                raise HTTPException(409,'로컬 CARLA 설치·엔진을 중지한 뒤 서버 주소·포트·연구 Python을 변경하세요.')
            result=config.save_settings(value);catalog.invalidate()
        if any(previous.get(k)!=result.get(k) for k in ('carla_host','carla_port','python_executable')):
            runtime._carla_enabled=False;runtime._carla_generation+=1
            runtime.carla={'connected':False,'error':'설정이 바뀌었습니다. CARLA 연결을 다시 확인하세요.','checked_at':None}
        return result
    @app.get('/api/config/sensors')
    def sensors():return config.sensors()
    @app.put('/api/config/sensors')
    def save_sensors(value:dict=Body(...)):return config.save_sensors(value)
    @app.get('/api/config/training')
    def training():return config.training()
    @app.put('/api/config/training')
    def save_training(value:dict=Body(...)):return config.save_training(value)
    @app.post('/api/training/options')
    def training_input_options(value:dict=Body(...)):
        if set(value)-{'sensor_use','train_data','val_data','task_settings'}:raise HTTPException(400,'지원 옵션은 sensor_use, train_data, val_data, task_settings입니다.')
        selected=value.get('sensor_use')
        if selected is not None and (not isinstance(selected,dict) or any(type(v) is not bool for v in selected.values())):
            raise HTTPException(400,'sensor_use는 센서별 true/false 객체여야 합니다.')
        draft_settings=config.settings()
        for key in ('train_data','val_data'):
            if key in value:
                config.path(value[key])
                draft_settings[key]=value[key]
        return training_options(config.root,draft_settings,selected,value.get('task_settings'))
    @app.post('/api/training/task-catalog')
    def training_task_catalog(value:dict=Body(...)):
        if set(value)-{'sensor_use','train_data','val_data'}:raise ValueError('Supported fields: sensor_use, train_data, val_data')
        draft_settings=config.settings()
        for key in ('train_data','val_data'):
            if key in value:
                config.path(value[key]);draft_settings[key]=value[key]
        return task_catalog(config.root,draft_settings,value.get('sensor_use'))
    @app.post('/api/preflight')
    def preflight(value:dict=Body(...)):
        for key in ('prepare_weights','allow_busy'):
            if key in value and type(value[key]) is not bool:raise HTTPException(400,key+'는 true/false여야 합니다.')
        settings_override=training_settings(value['settings']) if 'settings' in value else None
        if value.get('allow_busy') and value.get('kind')!='training':raise HTTPException(400,'allow_busy는 학습 대기열 준비 검사에만 사용할 수 있습니다.')
        return runtime.preflight(value.get('kind'),value.get('config'),prepare_weights=value.get('prepare_weights',False),settings_override=settings_override,allow_busy=value.get('allow_busy',False))
    def training_settings(value):
        if not isinstance(value,dict) or set(value)-{'train_data','val_data','sensor_config','test_tensor'}:
            raise ValueError('다음 학습의 설정은 train_data, val_data, sensor_config, test_tensor 경로만 지정하세요.')
        return config.validate_settings(value)
    @app.get('/api/training/queue')
    def training_queue():return {'jobs':jobs.queue()}
    @app.post('/api/training/queue')
    def enqueue_training(value:dict=Body(...)):
        if set(value)-{'name','config','settings','prepare_weights'}:raise HTTPException(400,'지원하지 않는 대기열 설정 키가 있습니다.')
        if 'prepare_weights' in value and type(value['prepare_weights']) is not bool:raise HTTPException(400,'prepare_weights는 true/false여야 합니다.')
        draft=config.validate_training(value.get('config',config.training()))
        selected=training_settings(value.get('settings',{}))
        # Preparation is asynchronous. An unavailable job stays queued with its
        # real blockers, rather than pretending to be a running training task.
        runtime.preflight('training',draft,prepare_weights=value.get('prepare_weights',True),settings_override=selected,allow_busy=True)
        result=jobs.enqueue_training(value.get('name',''),draft,settings_override=selected)
        catalog.invalidate()
        return result
    @app.post('/api/training/queue/reorder')
    def reorder_training(value:dict=Body(...)):
        if set(value)!={'ids'}:raise HTTPException(400,'ids 목록으로 대기 순서를 지정하세요.')
        return {'jobs':jobs.reorder(value['ids'])}
    @app.get('/api/sensor-preview')
    @app.get('/api/sensor-preview/status')
    def preview_status():return sensor_preview.status()
    def preview_payload(value):
        view=value.get('view','perspective')
        if view not in ('top','side','perspective'):raise ValueError('미리보기 시점은 top, side, perspective입니다.')
        sensors=config.validate_sensors(value['sensors'] if 'sensors' in value else config.sensors())
        return sensors,view
    @app.post('/api/sensor-preview/start')
    def start_sensor_preview(value:dict=Body(...)):
        sensors,view=preview_payload(value)
        with jobs.resource_guard():
            if jobs.active():raise HTTPException(409,'연구 작업을 먼저 중지한 뒤 센서 미리보기를 시작하세요.')
            if local_carla.busy():raise HTTPException(409,'CARLA 설치·실행 준비가 끝난 뒤 센서 미리보기를 시작하세요.')
            if sensor_preview.busy():raise HTTPException(409,'센서 미리보기가 이미 실행 중입니다. 설정 적용을 사용하세요.')
            connected=runtime.probe_carla()
            if not connected.get('connected'):raise HTTPException(409,connected.get('error','CARLA 연결을 먼저 확인하세요.'))
            check=runtime.preflight('sensor_preview',sensors)
            if not check['ready']:raise HTTPException(409,'\n'.join(check['blockers']))
            engine=local_carla.status().get('engine',{})
            owns_clock=bool(engine.get('owned') and engine.get('running') and engine.get('ready'))
            try:return sensor_preview.start(sensors=sensors,view=view,owns_clock=owns_clock)
            except (ValueError,RuntimeError,OSError) as error:raise HTTPException(409,str(error)) from error
    @app.put('/api/sensor-preview')
    def apply_sensor_preview(value:dict=Body(...)):
        sensors,view=preview_payload(value)
        with jobs.resource_guard():
            if jobs.active():raise HTTPException(409,'연구 작업 중에는 센서 미리보기를 변경할 수 없습니다.')
            try:return sensor_preview.apply(sensors=sensors,view=view)
            except (ValueError,RuntimeError,OSError) as error:raise HTTPException(409,str(error)) from error
    @app.post('/api/sensor-preview/stop')
    def stop_sensor_preview():
        with jobs.resource_guard():return sensor_preview.stop()
    @app.get('/api/sensor-preview/logs')
    def sensor_preview_logs():return {'text':sensor_preview.logs()}
    @app.get('/api/sensor-preview/frame')
    def sensor_preview_frame(channel:str='front_cam_rgb',revision:int=None,session_id:str=None):
        if revision is not None or session_id is not None:
            current=sensor_preview.status()
            if (revision is not None and revision!=current.get('revision')) or (session_id is not None and session_id!=current.get('session_id')):
                raise FileNotFoundError('새 센서 구성의 프레임을 기다리는 중입니다.')
        return FileResponse(sensor_preview.frame(channel),media_type='image/jpeg',headers={'Cache-Control':'no-store'})
    @app.post('/api/carla/connect')
    def connect():
        if local_carla.busy():raise HTTPException(409,'CARLA 설치·시작이 진행 중입니다. 준비가 끝나면 자동으로 연결합니다.')
        return runtime.probe_carla()
    @app.get('/api/carla/local')
    def local_status():return local_carla.status()
    @app.post('/api/carla/local/start')
    def local_start(value:dict=Body(...)):
        operation=value.get('operation','prepare')
        if operation not in ('prepare','install'):raise ValueError('CARLA 작업은 prepare 또는 install이어야 합니다.')
        with jobs.resource_guard():
            if sensor_preview.busy():raise HTTPException(409,'센서 미리보기를 중지한 뒤 CARLA를 설치·실행하세요.')
            if jobs.active():raise HTTPException(409,'연구 작업을 완료하거나 중지한 뒤 CARLA를 설치·실행하세요.')
            try:result=local_carla.start(operation=operation)
            except (ValueError,RuntimeError,OSError) as exc:raise HTTPException(409,str(exc)) from exc
        return result
    @app.post('/api/carla/local/stop')
    def local_stop():
        with jobs.resource_guard():
            if sensor_preview.busy():raise HTTPException(409,'센서 미리보기를 중지한 뒤 CARLA 엔진을 종료하세요.')
            active=jobs.active()
            if active and active['kind']=='collection':raise HTTPException(409,'데이터 수집을 먼저 중지한 뒤 CARLA 엔진을 종료하세요.')
            result=local_carla.stop()
            runtime._carla_generation+=1;runtime._carla_enabled=False;runtime._local_task_id=None
            runtime.carla={'connected':False,'error':'로컬 CARLA 중지를 요청했습니다.','checked_at':now()}
            return result
    def diagnostic_report():
        from .carla_diagnostics import diagnostics
        report=diagnostics(config)
        local=local_carla.status()
        report['engine']=local.get('engine',{})
        report['task']=local.get('task',{})
        report['rpc_port']=config.settings()['carla_port']
        report['logs']=local_carla.logs()
        return report
    @app.post('/api/carla/diagnostics')
    def run_diagnostics():return diagnostic_report()
    @app.get('/api/carla/diagnostics/export')
    def export_diagnostics():
        return Response(json.dumps(diagnostic_report(),ensure_ascii=False,indent=2),media_type='application/json',
                        headers={'Content-Disposition':'attachment; filename="sdv-carla-diagnostics.json"'})
    @app.post('/api/carla/diagnostics/test')
    def test_engine():
        with jobs.resource_guard():
            if jobs.active() or sensor_preview.busy():raise HTTPException(409,'수집·학습·프리뷰를 먼저 중지하세요.')
            try:return local_carla.start(operation='test')
            except (ValueError,RuntimeError,OSError) as error:raise HTTPException(409,str(error)) from error
    @app.get('/api/carla/local/logs')
    def local_logs():return {'text':local_carla.logs()}
    @app.post('/api/carla/disconnect')
    def disconnect():
        with jobs.resource_guard():
            if sensor_preview.busy():raise HTTPException(409,'센서 미리보기를 중지한 뒤 연결을 해제하세요.')
            active=jobs.active()
            if active and active['kind']=='collection':raise HTTPException(409,'수집을 중지한 뒤 연결을 해제하세요.')
            runtime._carla_enabled=False;runtime._carla_generation+=1
            runtime.carla={'connected':False,'error':'사용자가 연결을 해제했습니다. 외부 CARLA 서버는 계속 실행됩니다.','checked_at':now()};return runtime.carla_state()
    @app.get('/api/jobs')
    def list_jobs():return {'jobs':summaries()}
    @app.post('/api/jobs')
    def start_job(value:dict=Body(...)):
        kind=value.get('kind');override=value.get('config')
        with jobs.resource_guard():
            if sensor_preview.busy():raise HTTPException(409,'센서 미리보기를 중지한 뒤 연구 작업을 시작하세요.')
            if local_carla.busy():raise HTTPException(409,'CARLA 설치·시작 작업이 끝난 뒤 연구 작업을 실행하세요.')
            if kind=='collection':runtime.probe_carla()
            check=runtime.preflight(kind,override)
            if not check['ready']:raise HTTPException(409,'\n'.join(check['blockers']))
            try:return jobs._start(kind,value.get('name',''),override)
            except ValueError as exc:raise HTTPException(409,str(exc)) from exc
    @app.get('/api/jobs/{jid}')
    def job(jid:str):
        jobs.reconcile();value=jobs.get(jid)
        return dict(value,metric_registry=metric_registry(value.get('config')))
    @app.post('/api/jobs/{jid}/manual-control')
    def manual_control(jid:str,value:dict=Body(...)):
        if set(value)-{'sequence','sent_unix','throttle','brake','steer'}:
            raise ValueError('지원하지 않는 Manual Control 항목입니다.')
        command={key:number(value.get(key),key,*bounds) for key,bounds in
                 (('throttle',(0,1)),('brake',(0,1)),('steer',(-1,1)))}
        if command['throttle'] and command['brake']:
            raise ValueError('Throttle과 Brake를 동시에 적용할 수 없습니다.')
        command['sequence']=integer(value.get('sequence'),'Manual sequence',0,9007199254740991)
        sent=number(value.get('sent_unix'),'Manual timestamp',0,1e12)
        received=time.time()
        if not -.2<=received-sent<=.5:
            raise HTTPException(409,'Manual Control 요청이 만료되었습니다. 서버와 브라우저의 시각/연결을 확인하세요.')
        with runtime._manual_lock:
            job=jobs.get(jid)
            if (job['kind']!='collection' or job['status']!='Running' or job.get('stop_requested')
                    or job.get('settings',{}).get('driving',{}).get('agent')!='manual'):
                raise HTTPException(409,'실행 중인 Manual Control 수집 작업에서만 차량을 제어할 수 있습니다.')
            path=config.path(job['run_dir'])/'manual_control.json'
            previous=json.loads(path.read_text()) if path.is_file() else {}
            if command['sequence']<=previous.get('sequence',-1):
                raise HTTPException(409,'이전 Manual Control 요청은 적용하지 않습니다.')
            command.update(job_id=jid,issued_unix=received,sent_unix=sent,hand_brake=False,reverse=False)
            atomic_json(path,command)
        return {'accepted':True,'command':command,'expires_unix':received+.5}
    @app.post('/api/jobs/{jid}/stop')
    def stop_job(jid:str):return jobs.stop(jid)
    @app.post('/api/jobs/{jid}/pause')
    def pause_job(jid:str):return jobs.pause(jid)
    @app.post('/api/jobs/{jid}/restart')
    def restart_job(jid:str):return jobs.restart(jid)
    @app.post('/api/jobs/{jid}/resume')
    def resume_job(jid:str):return jobs.resume(jid)
    @app.delete('/api/jobs/{jid}')
    def remove_job(jid:str):return jobs.remove(jid)
    @app.get('/api/jobs/{jid}/logs')
    def logs(jid:str):return {'text':jobs.logs(jid)}
    @app.get('/api/jobs/{jid}/scene')
    def collection_scene(jid:str):
        value=jobs.get(jid)
        if value['kind']!='collection':raise ValueError('Scene telemetry requires a collection job')
        event=value.get('last_committed_frame') or value.get('last_event') or {}
        if not event.get('info'):return {'scene':None,'frame_id':None,'timestamp':None,'reason':'No committed collection frame'}
        path=config.path(event['info'])
        output=Path(value['output_dir']).resolve()
        if output not in path.parents or path.parent.name!='info' or path.suffix!='.json':raise ValueError('Invalid collection metadata path')
        for did,(scene,_) in catalog._scene_paths().items():
            if scene==path.parent.parent:
                sequence=sequences.sequence(did)
                frame=next((f for f in sequence['frames'] if f['frame_key']==path.stem),None)
                if frame is not None:
                    cameras=sequence.get('cameras') or ['front_cam']
                    camera='front_cam' if 'front_cam' in cameras else cameras[0]
                    result=sequences.frame(did,frame['index'],camera)
                    return {'scene':result['scene'],'frame_id':result['frame_id'],'timestamp':result['timestamp'],
                            'dataset_id':did,'index':frame['index'],'frame_key':frame['frame_key'],'layers':sequence['layers'],
                            'bev_gt':result.get('bev_gt'),'bev_views':sequence.get('bev_views',[]),'camera':camera}
        return {'scene':None,'frame_id':event.get('carla_frame'),'timestamp':event.get('timestamp_s'),'reason':'Committed sequence indexing is pending'}
    @app.get('/api/jobs/{jid}/metrics')
    def metrics(jid:str):
        value=jobs.get(jid)
        return {'metrics':value.get('metrics',[]),'metric_registry':metric_registry(value.get('config'))}
    def job_run(jid):
        return config.path(jobs.get(jid)['run_dir'])
    @app.get('/api/jobs/{jid}/predictions')
    def job_predictions(jid:str,split:str=None,epoch:int=None,offset:int=0,limit:int=100,task:str=None,sample_id:str=None,global_step:int=None):
        return prediction_index(job_run(jid),split,epoch,offset,limit,task,sample_id,global_step)
    @app.get('/api/jobs/{jid}/predictions/frame')
    def job_prediction_frame(jid:str,path:str):return FileResponse(prediction_frame(job_run(jid),path))
    @app.get('/api/jobs/{jid}/preview')
    def preview(jid:str,channel:str='camera'):
        if channel not in {'camera','bev'}:raise ValueError('프리뷰 채널은 camera 또는 bev입니다.')
        j=jobs.get(jid);value=j.get('preview',{}).get(channel)
        if not value:raise FileNotFoundError()
        p=config.path(value)
        roots=[Path(j['run_dir']).resolve(),Path(j['output_dir']).resolve()]
        if not any(root in p.parents for root in roots) or p.suffix.lower() not in {'.jpg','.jpeg','.png'}:raise ValueError('허용된 작업 이미지가 아닙니다.')
        if not p.is_file():raise FileNotFoundError()
        return FileResponse(p)
    @app.get('/api/experiments')
    def experiments():return {'experiments':catalog.experiments()}
    @app.get('/api/experiments/{eid}')
    def experiment(eid:str):return catalog.experiment(eid)
    @app.get('/api/experiments/{eid}/predictions')
    def experiment_predictions(eid:str,split:str=None,epoch:int=None,offset:int=0,limit:int=100,task:str=None,sample_id:str=None,global_step:int=None):
        return prediction_index(catalog.run_path(eid),split,epoch,offset,limit,task,sample_id,global_step)
    @app.get('/api/experiments/{eid}/predictions/frame')
    def experiment_prediction_frame(eid:str,path:str):return FileResponse(prediction_frame(catalog.run_path(eid),path))
    @app.get('/api/experiments/{eid}/artifact')
    def artifact(eid:str,path:str):return FileResponse(catalog.artifact(eid,path),filename=Path(path).name)
    @app.get('/api/experiments/{eid}/export')
    def export(eid:str):
        p=catalog.export(eid);return FileResponse(p,filename=p.name,media_type='application/zip')
    @app.get('/api/datasets')
    def datasets():return {'datasets':catalog.datasets()}
    @app.get('/api/datasets/{did}/deletion')
    def dataset_deletion_information(did:str):
        from .dataset_delete import deletion_target,deletion_blockers
        jobs.reconcile()
        path=deletion_target(catalog,did)
        blockers=deletion_blockers(catalog,jobs,path)
        return {'id':did,'name':path.name,'path':str(path.relative_to(config.root)),
                'can_delete':not blockers,'blockers':blockers}
    @app.delete('/api/datasets/{did}')
    def delete_dataset(did:str,value:dict=Body(...)):
        from .dataset_delete import deletion_target,deletion_blockers,delete_scene
        if set(value)!={'confirmation_name'}:raise ValueError('삭제 확인용 Scene 이름이 필요합니다.')
        with jobs.resource_guard(),jobs._lock:
            jobs.reconcile()
            path=deletion_target(catalog,did)
            blockers=deletion_blockers(catalog,jobs,path)
            if blockers:raise HTTPException(409,' '.join(blockers))
            try:delete_scene(catalog,path,value['confirmation_name'])
            finally:sequences.invalidate()
        return {'deleted':True,'id':did,'name':path.name}
    @app.get('/api/datasets/{did}/preview')
    def dataset_preview(did:str,channel:str='front_cam_rgb'):return FileResponse(catalog.preview(did,channel))
    @app.get('/api/datasets/{did}/trajectory')
    def trajectory(did:str):return catalog.trajectory(did)
    @app.get('/api/datasets/{did}/sequence')
    def dataset_sequence(did:str):return sequences.sequence(did)
    @app.get('/api/datasets/{did}/frames/{index}')
    def dataset_frame_information(did:str,index:int,camera:str='front_cam'):return sequences.frame(did,index,camera)
    @app.get('/api/datasets/{did}/frame')
    def dataset_frame_image(did:str,index:int=0,channel:str='rgb',camera:str='front_cam',layers:str=None,
                            classes:str=None,show_boxes:bool=True,colors:str=None,depth_min:float=0,
                            depth_max:float=100,colormap:str='turbo'):
        selected_layers=None if layers is None else [s for s in layers.split(',') if s]
        selected_classes=None if classes is None else [int(s) for s in classes.split(',') if s]
        selected_colors=None if colors is None else json.loads(colors)
        data=sequences.render(did,index,channel,camera,selected_layers,selected_classes,show_boxes,
                              selected_colors,depth_min,depth_max,colormap)
        return Response(data,media_type='image/png')
    @app.get('/api/datasets/{did}/quality')
    def dataset_quality(did:str):
        from .dataset_quality import validate_sequence
        return validate_sequence(sequences,did)
    @app.post('/api/datasets/{did}/planning')
    def dataset_planning(did:str,value:dict=Body(...)):
        from .planning import export_sequence
        if set(value)-{'horizon_frames','stride','max_gap_s'}:raise ValueError('Unsupported planning export setting')
        result=export_sequence(sequences,did,horizon=value.get('horizon_frames',20),stride=value.get('stride',1),max_gap_s=value.get('max_gap_s'))
        return JSONResponse(result,headers={'Content-Disposition':'attachment; filename="sdv-sequence-{}.json"'.format(did)})
    @app.post('/api/datasets/prepare')
    def prepare_dataset(value:dict=Body(...)):
        with jobs.resource_guard():
            if jobs.active():raise HTTPException(409,'실행 중에는 데이터 분할을 준비할 수 없습니다.')
            c=config.validate_training(value.get('train_config') or config.training())
            result=catalog.prepare_dataset(value.get('name',''),value.get('train_scenes',[]),value.get('val_scenes',[]),c)
            if isinstance(result,dict) and result.get('settings'):config.save_settings(result['settings'])
            return result
    @app.get('/api/events')
    async def events(request:Request):
        async def stream():
            while not await request.is_disconnected():
                payload=await to_thread(status)
                yield 'event: status\ndata: '+json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n\n'
                await asyncio.sleep(2)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'X-Accel-Buffering':'no'})
    static=Path(__file__).resolve().parent/'static'
    if static.is_dir():app.mount('/static',StaticFiles(directory=static),name='static')
    @app.get('/',include_in_schema=False)
    def index():return FileResponse(static/'index.html')
    return app
