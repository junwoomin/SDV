"""Durable process supervisor and immutable FIFO training queue."""
from __future__ import annotations
import json
import math
import fcntl
from contextlib import contextmanager
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
from .config import atomic_json
from sdv.training.metric_registry import metric_definition


def now():return datetime.now(timezone.utc).isoformat()

def alive(pid,created):
    if not pid or not created:return False
    try:
        p=psutil.Process(int(pid))
        return abs(p.create_time()-float(created))<.1 and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except (psutil.Error,ValueError,TypeError):return False

class JobManager:
    def __init__(self,config):
        self.config=config;self.db_path=config.state/'jobs.sqlite3';self._lock=threading.RLock();self._owned={};self._adopted=set();self._closing=False;self._training_gate=None;self._training_resource_gate=None;self._dispatch_thread=None;self._dispatch_wake=threading.Event()
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,status TEXT NOT NULL,created_at TEXT NOT NULL,payload TEXT NOT NULL)')
            db.execute('DROP INDEX IF EXISTS one_running_job')
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_active_job ON jobs((1)) WHERE status IN ('Running','Pausing')")
            columns={r[1] for r in db.execute('PRAGMA table_info(jobs)')}
            for column in ('summary','curve'):
                if column not in columns:db.execute(f'ALTER TABLE jobs ADD COLUMN {column} TEXT')
            for row in db.execute('SELECT payload FROM jobs').fetchall():
                job=json.loads(row['payload'])
                if job['kind']=='training' and 'best_metrics' not in job:
                    for event in job.get('metrics',[]):
                        if (event.get('event') or event.get('type'))=='training_epoch':self._record_best(job,event)
                self._save(job,db)
        self.reconcile()
    def db(self):
        db=sqlite3.connect(self.db_path,timeout=15);db.row_factory=sqlite3.Row;db.execute('PRAGMA busy_timeout=15000');return db
    def _compact(self,job):
        keys=('id','kind','name','status','created_at','started_at','finished_at','progress_current','progress_total','progress_pct','message','stop_requested','return_code','output_dir','last_event','preview','pid','queue_position','queue_removed','blocked_reason','pause_requested','checkpoint','pause_event','last_completed_epoch','global_step','active_elapsed_seconds','segment_started_unix','resume_count','config','settings','best_metrics','last_driving_status','restart_of','restart_mode')
        summary={key:job.get(key) for key in keys}
        elapsed=job.get('active_elapsed_seconds')
        if elapsed is None:
            try:
                start=datetime.fromisoformat(job['started_at']).timestamp()
                end=datetime.fromisoformat(job['finished_at']).timestamp() if job.get('finished_at') else time.time()
                elapsed=max(0,end-start)
            except (TypeError,ValueError,KeyError):elapsed=0.0
        if job['status'] in ('Running','Pausing') and job.get('segment_started_unix'):
            elapsed+=max(0,time.time()-job['segment_started_unix'])
        summary['can_resume'] = bool(job['kind']=='training' and job['status'] in ('Paused','Stopped','Failed') and not job.get('queue_removed') and self._usable_checkpoint(job) and 0<job.get('last_completed_epoch',0)<job.get('config',{}).get('Epochs',0))
        summary['elapsed_seconds']=elapsed
        completed=job.get('last_completed_epoch',0)
        durations=[e['duration_seconds'] for e in job.get('metrics',[]) if (e.get('event') or e.get('type'))=='training_epoch' and isinstance(e.get('duration_seconds'),(int,float)) and math.isfinite(e['duration_seconds']) and e['duration_seconds']>0]
        if durations and job['status'] in ('Running','Pausing'):
            remaining=max(0,job.get('progress_total',0)-completed)*sum(durations)/len(durations)
            latest=next((e for e in reversed(job.get('metrics',[])) if (e.get('event') or e.get('type'))=='training_epoch' and isinstance(e.get('timestamp'),(int,float))),None)
            if latest is not None and math.isfinite(latest['timestamp']):
                boundary=max(latest['timestamp'],job.get('segment_started_unix') or latest['timestamp'])
                remaining=max(0,remaining-max(0,time.time()-boundary))
            summary['eta_remaining_seconds']=remaining
            summary['estimated_finish_at']=datetime.fromtimestamp(time.time()+remaining,timezone.utc).isoformat()
        else:summary.update(eta_remaining_seconds=None,estimated_finish_at=None)
        epochs=[e for e in job.get('metrics',[]) if (e.get('event') or e.get('type'))=='training_epoch']
        source=epochs or job.get('metrics',[])
        curve=source[-200:]
        summary['metrics_truncated']=len(source)>len(curve)
        return summary,curve
    def _save(self,job,db=None):
        if db is not None:
            summary,curve=self._compact(job)
            db.execute('UPDATE jobs SET status=?,payload=?,summary=?,curve=? WHERE id=?',(job['status'],json.dumps(job,ensure_ascii=False,allow_nan=False),json.dumps(summary,ensure_ascii=False,allow_nan=False),json.dumps(curve,ensure_ascii=False,allow_nan=False),job['id']));return
        with self.db() as connection:self._save(job,connection)
    def summaries(self):
        self.reconcile()
        with self.db() as db:
            rows=db.execute("SELECT summary FROM jobs WHERE id IN (SELECT id FROM jobs ORDER BY created_at DESC LIMIT 200) OR status IN ('Running','Pausing') ORDER BY created_at DESC").fetchall()
            result=[json.loads(row['summary']) for row in rows]
            latest=next((j['id'] for j in result if j['kind']=='training' and j['status'] in ('Running','Pausing')),None) or next((j['id'] for j in result if j['kind']=='training' and j['status']!='Queued'),None)
            for job in result:job['metrics']=[]
            if latest:
                row=db.execute('SELECT curve FROM jobs WHERE id=?',(latest,)).fetchone()
                next(j for j in result if j['id']==latest)['metrics']=json.loads(row['curve'] or '[]')
        return result
    def get(self,jid):
        with self.db() as db:row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
        if not row:raise KeyError(jid)
        job=json.loads(row['payload']);summary,_=self._compact(job)
        for key in ('elapsed_seconds','eta_remaining_seconds','estimated_finish_at','can_resume'):job[key]=summary[key]
        return job
    def list(self):
        self.reconcile()
        with self.db() as db:rows=db.execute('SELECT payload FROM jobs ORDER BY created_at DESC LIMIT 200').fetchall()
        return [json.loads(r['payload']) for r in rows]
    def active(self):
        self.reconcile()
        with self.db() as db:row=db.execute("SELECT id FROM jobs WHERE status IN ('Running','Pausing') LIMIT 1").fetchone()
        return self.get(row['id']) if row else None
    def reconcile(self):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT payload FROM jobs WHERE status IN ('Running','Pausing')"):
                job=json.loads(row['payload'])
                if job['id'] in self._owned:continue
                if not job.get('pid'):
                    starter=job.get('supervisor_pid');started=job.get('supervisor_created')
                    if alive(starter,started) and time.time()-job.get('created_unix',0)<15:continue
                elif alive(job.get('pid'),job.get('pid_created')):
                    if not self._closing and job['id'] not in self._adopted:
                        self._adopted.add(job['id']);threading.Thread(target=self._adopt,args=(job['id'],),daemon=True).start()
                    continue
                if self._descendants_alive(job):
                    self._signal(job,signal.SIGKILL)
                    job['message']='종료된 작업의 남은 프로세스를 정리하는 중입니다.'
                    self._save(job,db);continue
                receipt=Path(job.get('receipt_path','/nonexistent'))
                if receipt.is_file():
                    try:
                        record=json.loads(receipt.read_text());rc=record['return_code']
                        finished=datetime.fromtimestamp(record.get('finished_unix',time.time()),timezone.utc).isoformat()
                        self._events(job,0)
                        self._complete_exit(job,rc,finished)
                        self._save(job,db);continue
                    except (ValueError,OSError,KeyError):pass
                job.update(status='Stopped' if job.get('stop_requested') else 'Failed',finished_at=now(),message='작업 프로세스가 종료되었습니다. 로그를 확인하세요.',return_code=job.get('return_code',-1))
                self._save(job,db)
    def _adopt(self,jid):
        position=0
        try:
            while not self._closing:
                with self.db() as db:
                    db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
                    if not row:return
                    job=json.loads(row['payload'])
                    if job['status'] not in ('Running','Pausing'):return
                    position=self._events(job,position);self._capture_descendants(job);self._save(job,db)
                if not alive(job.get('pid'),job.get('pid_created')):
                    self.reconcile();return
                time.sleep(.5)
        finally:self._adopted.discard(jid)
    def _command(self,kind,snapshot_dir,settings):
        python=settings['python_executable']
        if kind=='collection':return [python,'-u',str(self.config.root/'carla_run.py')]
        if kind=='training':return [python,'-u',str(self.config.root/'train2.py'),'--config',str(snapshot_dir/'train_config.yaml')]
        raise ValueError('지원 작업: collection, training')
    @contextmanager
    def resource_guard(self):
        with (self.config.state/'resource.lock').open('a') as handle:
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('데이터 준비 또는 작업 시작이 진행 중입니다. 잠시 후 다시 시도하세요.') from None
            try:yield
            finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    def start(self,kind,name='',override=None):
        with self.resource_guard():return self._start(kind,name,override)
    def _start(self,kind,name='',override=None):
        self.reconcile()
        with self._lock:
            job=self._build_job(kind,name,override,'Running')
            try:
                with self.db() as db:
                    db.execute('BEGIN IMMEDIATE')
                    if db.execute("SELECT 1 FROM jobs WHERE status IN ('Running','Pausing')").fetchone():raise ValueError('이미 실행 중인 작업이 있습니다. 학습은 대기열에 추가하세요.')
                    db.execute('INSERT INTO jobs(id,status,created_at,payload) VALUES(?,?,?,?)',(job['id'],job['status'],job['created_at'],json.dumps(job,ensure_ascii=False)))
                    self._save(job,db)
            except Exception:
                import shutil
                shutil.rmtree(job['run_dir'],ignore_errors=True)
                raise
            return self._launch(job)
    def _build_job(self,kind,name,override,status,settings_override=None):
        if kind not in ('collection','training'):raise ValueError('지원 작업: collection, training')
        if self._closing:raise ValueError('서버가 종료 중입니다.')
        cfg=self.config;s=cfg.validate_settings(settings_override) if settings_override is not None else cfg.settings()
        if kind=='training' and cfg.path(s['sensor_config']).is_file():
            import yaml
            sensors=cfg.validate_sensors(yaml.safe_load(cfg.path(s['sensor_config']).read_text(encoding='utf-8')))
        else:sensors=cfg.validate_sensors(override if kind=='collection' and override is not None else cfg.sensors())
        training=cfg.validate_training(override if kind=='training' and override is not None else cfg.training())
        jid=uuid.uuid4().hex[:16];run_dir=cfg.path(s['results_root'])/('job_'+jid);run_dir.mkdir(parents=True,exist_ok=False)
        log_path=cfg.path(s['logs_root'])/(jid+'.log');log_path.parent.mkdir(parents=True,exist_ok=True)
        collection_dir=cfg.path(s['data_root'])/('collection_'+jid)
        snapshot=run_dir/'config';snapshot.mkdir()
        cfg.yaml(snapshot/'sensor_config.yaml',sensors);cfg.yaml(snapshot/'train_config.yaml',training);atomic_json(snapshot/'settings.json',s)
        command=self._command(kind,snapshot,s)
        total=sensors['data_num']*len(sensors['towns']) if kind=='collection' else training['Epochs']
        job={'id':jid,'kind':kind,'name':str(name or ('데이터 수집' if kind=='collection' else '인지 모델 학습'))[:100],'status':status,'created_at':now(),'created_unix':time.time(),'started_at':None if status=='Queued' else now(),'finished_at':None,'progress_current':0,'progress_total':total,'progress_pct':0.0,'message':'작업 프로세스를 시작합니다.','metrics':[],'pid':None,'pid_created':None,'supervisor_pid':os.getpid(),'supervisor_created':psutil.Process().create_time(),'stop_requested':False,'return_code':None,'descendants':[],'run_dir':str(run_dir),'output_dir':str(collection_dir if kind=='collection' else run_dir/'output'),'log_path':str(log_path),'events_path':str(run_dir/'events.jsonl'),'receipt_path':str(run_dir/'exit.json'),'settings':s,'config':sensors if kind=='collection' else training,'sensor_config':sensors,'command':command}
        atomic_json(snapshot/'experiment.json',{'id':jid,'kind':kind,'name':job['name'],'created_at':job['created_at']})
        job.update(queue_position=None,queue_removed=False,pause_requested=False,blocked_reason=[],active_elapsed_seconds=0.0,resume_count=0,segment_started_unix=None,control_path=str(run_dir/'training_control.json'),checkpoint=None,last_completed_epoch=0,global_step=0)
        return job
    def _launch(self,job):
        cfg=self.config;s=job['settings'];jid=job['id'];run_dir=Path(job['run_dir']);snapshot=run_dir/'config';collection_dir=Path(job['output_dir']);log_path=Path(job['log_path']);command=job['command']
        try:Path(job['receipt_path']).unlink()
        except FileNotFoundError:pass
        try:Path(job['control_path']).unlink()
        except FileNotFoundError:pass
        job.update(segment_started_unix=time.time(),started_at=job.get('started_at') or now(),finished_at=None,stop_requested=False,pause_requested=False,blocked_reason=[],return_code=None)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
            if row:job['stop_requested']=json.loads(row['payload']).get('stop_requested',False)
            self._save(job,db)
        env=os.environ.copy();env.update({'PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg','SDL_VIDEODRIVER':'dummy','SDV_SENSOR_CONFIG':str(snapshot/'sensor_config.yaml'),'SDV_COLLECTION_OUTPUT':str(collection_dir),'SDV_TRAIN_OUTPUT':str(run_dir/'output'),'SDV_TRAIN_DATA':str(cfg.path(s['train_data'])),'SDV_VAL_DATA':str(cfg.path(s['val_data'])),'SDV_EVENTS_PATH':job['events_path'],'SDV_CARLA_MODE':'external','SDV_CARLA_HOST':s['carla_host'],'SDV_CARLA_PORT':str(s['carla_port']),'SDV_CARLA_TM_PORT':str(s['carla_tm_port']),'SDV_TM_PORT':str(s['carla_tm_port']),'SDV_VEHICLE':s['vehicle'],'SDV_WEATHER':s['weather'],'SDV_NPC_VEHICLES':str(s['npc_vehicles']),'SDV_NPC_WALKERS':str(s['npc_walkers']),'SDV_FPS':str(s['fps']),'SDV_PREVIEW_FPS':str(s['preview_fps']),'SDV_BEV_GENERATOR':json.dumps(s.get('bev_generator',{})),'SDV_BEV_CACHE':str(cfg.root/'.runtime/bev')})
        from .carla_service import read_state, state_directory
        engine_state=read_state(state_directory(cfg)).get('engine',{})
        env['SDV_CARLA_RENDER_PROFILE']=engine_state.get('profile','external-unspecified') if engine_state.get('running') and engine_state.get('host')==s['carla_host'] and engine_state.get('port')==s['carla_port'] else 'external-unspecified'
        env['SDV_DRIVING_CONFIG']=json.dumps(s.get('driving',{}))
        env['SDV_MANUAL_CONTROL_FILE']=str(run_dir/'manual_control.json')
        env['SDV_JOB_ID']=jid
        try:(run_dir/'manual_control.json').unlink()
        except FileNotFoundError:pass
        env['SDV_BEV_MAP_ROOT']=str(cfg.path(s.get('bev_generator',{}).get('map_root','assets/bev_maps')))
        env['SDV_TRAIN_CONTROL']=job['control_path']
        if job.get('resume_from'):env['SDV_TRAIN_RESUME']=job['resume_from']
        else:env.pop('SDV_TRAIN_RESUME',None)
        if cfg.path(s['test_tensor']).is_file():env['SDV_TEST_TENSOR']=str(cfg.path(s['test_tensor']))
        else:env.pop('SDV_TEST_TENSOR',None)
        logf=None
        try:
            logf=log_path.open('ab',buffering=0)
            logf.write(('SDV Studio job '+jid+'\n'+' '.join(command)+'\n').encode())
            worker_command=[sys.executable,'-u',str(Path(__file__).with_name('worker.py')),'--receipt',job['receipt_path'],'--',*command]
            proc=subprocess.Popen(worker_command,cwd=cfg.root,env=env,stdin=subprocess.DEVNULL,stdout=logf,stderr=subprocess.STDOUT,start_new_session=True)
            try:created=psutil.Process(proc.pid).create_time()
            except psutil.NoSuchProcess:created=None
            job.update(pid=proc.pid,pid_created=created,message='프로세스가 실행 중입니다.')
            with self.db() as db:
                db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
                if row:job['stop_requested']=json.loads(row['payload']).get('stop_requested',False)
                self._save(job,db)
            self._owned[jid]=(proc,logf)
            threading.Thread(target=self._monitor,args=(jid,proc,logf),daemon=True,name='sdv-job-'+jid).start()
        except Exception as exc:
            if logf:logf.close()
            job.update(status='Failed',finished_at=now(),message='프로세스를 시작하지 못했습니다: '+str(exc),return_code=-1)
            self._save(job)
        return self.get(jid)
    def set_training_gate(self,gate,resource_gate=None):
        """Enable automatic dispatch only after Runtime wires its readiness checks."""
        self._training_gate=gate;self._training_resource_gate=resource_gate
        if self._dispatch_thread is None:
            self._dispatch_thread=threading.Thread(target=self._dispatcher,daemon=True,name='sdv-training-queue')
            self._dispatch_thread.start()
        self._dispatch_wake.set()
    def enqueue_training(self,name='',override=None,settings_override=None):
        # Configuration and all paths are copied before the queued row appears.
        with self._lock:
            job=self._build_job('training',name,override,'Queued',settings_override)
            try:
                with self.db() as db:
                    db.execute('BEGIN IMMEDIATE')
                    queued=[json.loads(r['payload']) for r in db.execute("SELECT payload FROM jobs WHERE status='Queued'")]
                    job['queue_position']=max([j.get('queue_position') or 0 for j in queued]+[0])+1
                    job['message']='학습 대기열에 추가되었습니다.'
                    db.execute('INSERT INTO jobs(id,status,created_at,payload) VALUES(?,?,?,?)',(job['id'],'Queued',job['created_at'],json.dumps(job,ensure_ascii=False)))
                    self._save(job,db)
            except Exception:
                import shutil
                shutil.rmtree(job['run_dir'],ignore_errors=True)
                raise
        self._dispatch_wake.set()
        return self.get(job['id'])
    def queue(self):
        self.reconcile()
        with self.db() as db:
            rows=db.execute("SELECT summary FROM jobs WHERE status IN ('Running','Pausing','Paused','Queued') OR id IN (SELECT id FROM jobs WHERE status NOT IN ('Running','Pausing','Paused','Queued') ORDER BY created_at DESC LIMIT 200) ORDER BY created_at").fetchall()
        jobs=[json.loads(r['summary']) for r in rows]
        jobs=[j for j in jobs if j['kind']=='training' and not j.get('queue_removed')]
        order={'Running':0,'Pausing':0,'Queued':1,'Paused':2,'Completed':3,'Failed':3,'Stopped':3}
        for job in jobs:
            # No historical epoch curves travel in periodic queue updates.
            # Full detail and all predictions remain at the selected job endpoint.
            job['metrics']=[]
            if job['status'] in ('Running','Pausing') and job.get('segment_started_unix'):
                job['elapsed_seconds']=(job.get('active_elapsed_seconds') or 0)+max(0,time.time()-job['segment_started_unix'])
                if job.get('estimated_finish_at'):
                    job['eta_remaining_seconds']=max(0,datetime.fromisoformat(job['estimated_finish_at']).timestamp()-time.time())
        return sorted(jobs,key=lambda j:(order.get(j['status'],4),j.get('queue_position') or 0,j['created_at']))
    def reorder(self,ids):
        if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids) or len(ids)!=len(set(ids)):
            raise ValueError('중복 없이 대기 작업 ID 목록을 입력하세요.')
        with self._lock,self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            queued=[json.loads(r['payload']) for r in db.execute("SELECT payload FROM jobs WHERE status='Queued'")]
            jobs={j['id']:j for j in queued if j['kind']=='training' and not j.get('queue_removed')}
            if set(ids)!=set(jobs):raise ValueError('현재 대기 중인 모든 학습 작업의 ID를 순서대로 지정하세요.')
            for position,jid in enumerate(ids,1):
                jobs[jid]['queue_position']=position;self._save(jobs[jid],db)
        self._dispatch_wake.set()
        return self.queue()
    def pause(self,jid):
        with self._lock,self.db() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
            if not row:raise KeyError(jid)
            job=json.loads(row['payload'])
            if job['kind']!='training':raise ValueError('학습 작업만 일시정지할 수 있습니다.')
            if job['status'] in ('Pausing','Paused'):return job
            if job['status']!='Running' or job.get('stop_requested'):raise ValueError('실행 중인 학습만 일시정지할 수 있습니다.')
            if not job.get('pid'):raise ValueError('학습 프로세스가 시작 중입니다. 잠시 후 일시정지를 요청하세요.')
            if not job.get('control_path'):raise ValueError('일시정지 기능이 추가되기 전에 시작된 작업입니다. 중지 후 새 학습으로 시작하세요.')
            atomic_json(Path(job['control_path']),{'action':'pause','pause_requested':True,'requested_at':now()})
            job.update(status='Pausing',pause_requested=True,message='현재 에폭을 완료하고 모델·옵티마이저 상태를 저장한 뒤 일시정지합니다.')
            self._save(job,db)
        return self.get(jid)
    def _usable_checkpoint(self,job,path=None):
        candidate=path or job.get('checkpoint')
        if not isinstance(candidate,str) or not candidate:return None
        try:
            p=Path(candidate).resolve();root=Path(job['output_dir']).resolve()
            if root not in p.parents or p.name!='resume_checkpoint.pth' or not p.is_file() or p.stat().st_size<=0:return None
            return str(p)
        except (OSError,ValueError):return None
    def resume(self,jid):
        with self._lock,self.db() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
            if not row:raise KeyError(jid)
            job=json.loads(row['payload'])
            if job['kind']!='training' or job['status'] not in ('Paused','Stopped','Failed') or job.get('queue_removed'):
                raise ValueError('일시정지·중지·실패한 학습의 완료 Epoch 체크포인트만 이어할 수 있습니다.')
            checkpoint=self._usable_checkpoint(job)
            if not checkpoint:raise ValueError('이어하기 체크포인트가 없거나 손상되었습니다. 새 학습 재시작을 사용하세요.')
            if job.get('last_completed_epoch',0)>=job['config']['Epochs']:raise ValueError('모든 Epoch가 완료되어 이어할 작업이 없습니다. 새 학습을 시작하세요.')
            queued=[json.loads(r['payload']) for r in db.execute("SELECT payload FROM jobs WHERE status='Queued'")]
            job.update(status='Queued',queue_position=max([j.get('queue_position') or 0 for j in queued]+[0])+1,resume_from=checkpoint,resume_count=job.get('resume_count',0)+1,pause_requested=False,pause_event=None,stop_requested=False,blocked_reason=[],message='저장된 마지막 완료 에폭 다음부터 이어 학습하도록 대기열에 추가했습니다.')
            self._save(job,db)
        self._dispatch_wake.set()
        return self.get(jid)
    def restart(self,jid):
        # A fresh run has a new identity/output and preserves the source history.
        # The same-source double click returns its existing pending child.
        with self.resource_guard(),self._lock:
            source=self.get(jid)
            if source['kind']!='training' or source['status'] not in ('Stopped','Failed','Completed','Paused'):
                raise ValueError('종료·일시정지한 학습만 Epoch 1부터 새로 시작할 수 있습니다.')
            with self.db() as db:
                rows=db.execute("SELECT payload FROM jobs WHERE status IN ('Queued','Running','Pausing')").fetchall()
                for row in rows:
                    existing=json.loads(row['payload'])
                    if existing.get('restart_of')==jid:return existing
            job=self._build_job('training',source['name']+' · 새 학습',source['config'],'Queued',settings_override=source['settings'])
            job.update(restart_of=jid,restart_mode='fresh',message='원본 설정으로 Epoch 1부터 새 학습 대기. 원본 결과·체크포인트는 보존됩니다.')
            with self.db() as db:
                db.execute('BEGIN IMMEDIATE')
                queued=[json.loads(r['payload']) for r in db.execute("SELECT payload FROM jobs WHERE status='Queued'")]
                job['queue_position']=max([j.get('queue_position') or 0 for j in queued]+[0])+1
                db.execute('INSERT INTO jobs(id,status,created_at,payload) VALUES(?,?,?,?)',(job['id'],'Queued',job['created_at'],json.dumps(job,ensure_ascii=False)))
                self._save(job,db)
            self._dispatch_wake.set()
            return self.get(job['id'])

    def remove(self,jid):
        with self._lock,self.db() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
            if not row:raise KeyError(jid)
            job=json.loads(row['payload'])
            if job['kind']!='training':raise ValueError('학습 대기열의 작업만 삭제할 수 있습니다.')
            if job['status'] in ('Running','Pausing'):raise ValueError('먼저 학습을 일시정지하거나 중지한 뒤 대기열에서 삭제하세요.')
            job['queue_removed']=True;job['queue_position']=None
            if job['status']=='Queued':job.update(status='Stopped',finished_at=now(),message='대기열에서 삭제되었습니다. 설정과 결과는 실험 기록에 보존됩니다.')
            self._save(job,db)
        self._dispatch_wake.set()
        return self.get(jid)
    def _dispatcher(self):
        while not self._closing:
            # Readiness can involve disk scans or interpreter probes. Failed
            # readiness is retried slowly, and explicit queue edits wake early.
            self._dispatch_wake.wait(4);self._dispatch_wake.clear()
            if self._closing:return
            try:self._dispatch_once()
            except (ValueError,sqlite3.Error,OSError):continue
    def _dispatch_once(self):
        if self._closing or self._training_gate is None:return None
        self.reconcile()
        with self.db() as db:
            if db.execute("SELECT 1 FROM jobs WHERE status IN ('Running','Pausing')").fetchone():return None
            waiting=[json.loads(r['payload']) for r in db.execute("SELECT payload FROM jobs WHERE status='Queued'")]
            waiting=[j for j in waiting if j['kind']=='training' and not j.get('queue_removed')]
            if not waiting:return None
            job=min(waiting,key=lambda j:(j.get('queue_position') or 0,j['created_at']))
        # Gate checks never hold the process resource lock, SQLite write lock,
        # or manager lock. Saving a draft and adding another model remain usable.
        try:
            result=self._training_gate(job)
            ready=result.get('ready',False) if isinstance(result,dict) else bool(result)
            blockers=result.get('blockers',[]) if isinstance(result,dict) else []
        except Exception as exc:ready=False;blockers=['준비 상태 검사 실패: '+str(exc)]
        blockers=[str(value)[:1000] for value in blockers][:20]
        if not ready:
            with self.db() as db:
                db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(job['id'],)).fetchone()
                if row:
                    latest=json.loads(row['payload'])
                    if latest['status']=='Queued' and not latest.get('queue_removed'):
                        latest.update(blocked_reason=blockers or ['실행 환경의 준비가 완료되지 않았습니다.'],message='대기열 첫 작업의 준비 조건이 충족되기를 기다립니다.')
                        self._save(latest,db)
            return None
        if self._closing:return None
        with self.resource_guard(),self._lock:
            if self._closing:return None
            if self._training_resource_gate is not None:
                try:
                    resources=self._training_resource_gate(job)
                    resources_ready=resources.get('ready',False) if isinstance(resources,dict) else bool(resources)
                    reasons=resources.get('blockers',[]) if isinstance(resources,dict) else []
                except Exception as exc:resources_ready=False;reasons=['실행 자원 검사 실패: '+str(exc)]
                if not resources_ready:
                    with self.db() as db:
                        db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(job['id'],)).fetchone()
                        if row:
                            current=json.loads(row['payload'])
                            if current['status']=='Queued' and not current.get('queue_removed'):
                                current.update(blocked_reason=[str(v)[:1000] for v in reasons][:20] or ['실행 자원이 사용 중입니다.'],message='다른 작업이 실행 자원을 해제하기를 기다립니다.')
                                self._save(current,db)
                    return None
            with self.db() as db:
                db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(job['id'],)).fetchone()
                if not row:return None
                job=json.loads(row['payload'])
                current=[json.loads(r['payload']) for r in db.execute("SELECT payload FROM jobs WHERE status='Queued'")]
                current=[j for j in current if j['kind']=='training' and not j.get('queue_removed')]
                if not current or min(current,key=lambda j:(j.get('queue_position') or 0,j['created_at']))['id']!=job['id']:return None
                if job['status']!='Queued' or job.get('queue_removed') or db.execute("SELECT 1 FROM jobs WHERE status IN ('Running','Pausing')").fetchone():return None
                job.update(status='Running',pid=None,pid_created=None,created_unix=time.time(),descendants=[],supervisor_pid=os.getpid(),supervisor_created=psutil.Process().create_time(),blocked_reason=[],message='대기열에서 학습을 시작합니다.')
                self._save(job,db)
            return self._launch(job)
    def _complete_exit(self,job,rc,finished=None):
        if job.get('segment_started_unix'):
            end=datetime.fromisoformat(finished).timestamp() if finished else time.time()
            job['active_elapsed_seconds']=job.get('active_elapsed_seconds',0.0)+max(0,end-job['segment_started_unix'])
            job['segment_started_unix']=None
        pause=job.get('pause_event')
        checkpoint=self._usable_checkpoint(job,pause.get('resume_checkpoint') if isinstance(pause,dict) else None)
        status=('Stopped' if job.get('stop_requested') else
                'Paused' if rc==75 and isinstance(pause,dict) and checkpoint else
                'Completed' if rc==0 else 'Failed')
        job.update(status=status,return_code=rc,finished_at=finished or now())
        if status=='Paused':job.update(checkpoint=checkpoint,message='에폭 {}까지 저장하고 일시정지했습니다.'.format(job.get('last_completed_epoch',job.get('progress_current',0))))
        elif status=='Completed':
            if job['kind']=='collection':job['message']='수집 프로세스가 완료되었습니다. 실제 저장 {} / {} 샘플.'.format(job['progress_current'],job['progress_total'])
            else:job.update(progress_pct=100.0,progress_current=job['progress_total'],message='작업이 완료되었습니다.')
        elif status=='Stopped':job['message']='사용자 또는 서버 종료 요청으로 작업을 중지했습니다.'
        else:
            lines=[x.strip() for x in self.logs(job['id'],3000).splitlines() if x.strip()]
            job['message']=('프로세스 실패: '+(lines[-1] if lines else 'exit {}'.format(rc)))[:1000]
        return job
    def _record_best(self,job,event):
        best=job.setdefault('best_metrics',{})
        metrics=event.get('metrics')
        if not isinstance(metrics,dict):return
        epoch=event.get('epoch')
        if type(epoch) is not int:return
        for key,value in metrics.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):continue
            definition=metric_definition(key,job.get('config'))
            if definition is None or definition.get('unit')=='pixels':continue
            direction=definition['direction'];previous=best.get(key)
            if previous is None or (value<previous['value'] if direction=='min' else value>previous['value']):
                best[key]={'value':float(value),'epoch':epoch,'direction':direction}
    def _events(self,job,position):
        position=job.get('events_position',position)
        p=Path(job['events_path'])
        if not p.exists():return position
        with p.open('rb') as f:
            f.seek(position)
            for _ in range(5000):
                start=f.tell();line=f.readline()
                if not line:break
                if not line.endswith(b'\n'):f.seek(start);break
                try:event=json.loads(line)
                except (ValueError,UnicodeError):continue
                if not isinstance(event,dict):continue
                typ=event.get('event') or event.get('type')
                job['last_event']=event
                if job['kind']=='collection' and typ=='driving_status':
                    job['last_driving_status']=event
                if job['kind']=='collection' and typ in ('collection_started','collection_town_started','collection_progress','collection_completed'):
                    # Collection events always contain GLOBAL committed counts.
                    # The percentage is derived from these integers, never from a
                    # map-local percentage or a successful process exit alone.
                    saved=event.get('saved');total=event.get('total')
                    if (type(saved) is int and type(total) is int and total>0
                            and job.get('progress_current',0)<=saved<=total):
                        job.update(progress_current=saved,progress_total=total,progress_pct=100.0*saved/total)
                    job['message']=str(event.get('message') or '전체 {} / {} 샘플 저장{}'.format(
                        job['progress_current'],job['progress_total'],
                        ' · {}'.format(event['town']) if event.get('town') else ''))[:1000]
                    if event.get('preview'):job['preview']=event['preview']
                    if typ=='collection_progress' and event.get('info'):
                        job['last_committed_frame']={k:event.get(k) for k in ('info','scene','sample','carla_frame','timestamp_s')}
                elif typ in ('training_step','training_validation','training_epoch') and job['kind']=='training':
                    value=event.get('progress')
                    if isinstance(value,(int,float)) and 0<=value<=100:job['progress_pct']=float(value)*100 if value<=1 else float(value)
                    job['progress_current']=event.get('epoch',job['progress_current']);job['progress_total']=event.get('epochs',job['progress_total'])
                    if event.get('metrics') or typ=='training_epoch':
                        recorded=job['metrics']+[event]
                        epochs=[e for e in recorded if (e.get('event') or e.get('type'))=='training_epoch']
                        steps=[e for e in recorded if (e.get('event') or e.get('type'))!='training_epoch'][-200:]
                        job['metrics']=epochs+steps
                    job['message']=str(event.get('message') or (f"{event.get('town','')} · {job['progress_current']} / {job['progress_total']}" if job['kind']=='collection' else f"Epoch {job['progress_current']} / {job['progress_total']}"))[:1000]
                    if type(event.get('global_step')) is int:job['global_step']=event['global_step']
                    if typ=='training_epoch':
                        job['last_completed_epoch']=event.get('epoch',job.get('last_completed_epoch',0))
                        self._record_best(job,event)
                    if event.get('preview'):job['preview']=event['preview']
                elif typ in ('training_checkpoint','training_paused') and job['kind']=='training':
                    checkpoint=self._usable_checkpoint(job,event.get('resume_checkpoint'))
                    if checkpoint:job['checkpoint']=checkpoint
                    if type(event.get('epoch')) is int:job['last_completed_epoch']=event['epoch']
                    if type(event.get('global_step')) is int:job['global_step']=event['global_step']
                    if typ=='training_paused':job['pause_event']=event
                    if event.get('message'):job['message']=str(event['message'])[:1000]
                elif typ in ('error','failed'):job['message']=str(event.get('message','작업 오류'))[:1000]
                elif event.get('message'):job['message']=str(event['message'])[:1000]
            job['events_position']=f.tell()
            return f.tell()
    def _capture_descendants(self,job):
        try:
            descendants=psutil.Process(job['pid']).children(recursive=True)
            known={x['pid']:x for x in job.get('descendants',[])}
            for child in descendants:
                try:known[child.pid]={'pid':child.pid,'created':child.create_time()}
                except psutil.Error:pass
            job['descendants']=list(known.values())
        except psutil.Error:pass
    def _descendants_alive(self,job):
        return any(alive(x['pid'],x['created']) for x in job.get('descendants',[]))
    def _signal(self,job,sig):
        if not alive(job.get('pid'),job.get('pid_created')) and not self._descendants_alive(job):return False
        try:os.killpg(int(job['pid']),sig);return True
        except (ProcessLookupError,PermissionError):return False
    def _monitor(self,jid,proc,logf):
        position=0;sent=None;level=0
        try:
            while True:
                with self._lock:
                    with self.db() as db:
                        db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone();job=json.loads(row['payload'])
                        position=self._events(job,position)
                        self._capture_descendants(job)
                        if job.get('stop_requested'):
                            if sent is None:self._signal(job,signal.SIGINT);sent=time.monotonic();level=1
                            elif level==1 and time.monotonic()-sent>=12:self._signal(job,signal.SIGTERM);level=2
                            elif level==2 and time.monotonic()-sent>=20:self._signal(job,signal.SIGKILL);level=3
                        rc=proc.poll()
                        if rc is not None and self._descendants_alive(job):
                            # An exited launcher is not proof that its children exited.
                            # Retain Running until verified owned descendants are gone.
                            if sent is None:self._signal(job,signal.SIGTERM);sent=time.monotonic();level=2
                            elif time.monotonic()-sent>=3:self._signal(job,signal.SIGKILL);level=3
                            job['message']='남아 있는 작업 프로세스를 정리하는 중입니다.'
                            rc=None
                        if rc is not None:
                            position=self._events(job,position)
                            self._complete_exit(job,rc)
                        self._save(job,db)
                if rc is not None:break
                time.sleep(.25)
        except Exception as exc:
            job=self.get(jid);self._signal(job,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:self._signal(job,signal.SIGKILL);proc.wait()
            job.update(status='Failed',finished_at=now(),message='작업 감독 오류: '+str(exc),return_code=proc.returncode);self._save(job)
        finally:
            logf.close();self._owned.pop(jid,None);self._dispatch_wake.set()
    def stop(self,jid):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
            if not row:raise KeyError(jid)
            job=json.loads(row['payload'])
            if job['status'] not in ('Running','Pausing'):return job
            job.update(stop_requested=True,message='중지 요청: 센서와 CARLA 액터를 정리하는 중입니다.');self._save(job,db)
        # Signals immediately, even when request reaches another server process.
        if jid not in self._owned:
            self._signal(job,signal.SIGINT)
            threading.Thread(target=self._external_stop,args=(jid,),daemon=True).start()
        return self.get(jid)
    def _external_stop(self,jid):
        for seconds,sig in [(12,signal.SIGTERM),(8,signal.SIGKILL)]:
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                job=self.get(jid)
                if job['status'] not in ('Running','Pausing') or (not alive(job.get('pid'),job.get('pid_created')) and not self._descendants_alive(job)):self.reconcile();return
                time.sleep(.25)
            self._signal(job,sig)
        self.reconcile()
    def logs(self,jid,max_bytes=65536):
        p=Path(self.get(jid)['log_path'])
        if not p.exists():return ''
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-max_bytes));return f.read(max_bytes).decode('utf-8',errors='replace')
    def shutdown(self):
        self._closing=True;self._dispatch_wake.set()
        targets=set(self._owned)|set(self._adopted)
        for jid in targets:self.stop(jid)
        deadline=time.monotonic()+24
        while time.monotonic()<deadline:
            self.reconcile()
            if not any(self.get(jid)['status'] in ('Running','Pausing') for jid in targets):break
            time.sleep(.1)
        for jid in targets:
            job=self.get(jid)
            if job['status'] in ('Running','Pausing'):self._signal(job,signal.SIGKILL)
        self.reconcile()
