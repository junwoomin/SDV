import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from studio.config import Config, atomic_json
from studio.carla_worker import engine_command, RPC_SCRIPT
from studio.carla_service import CarlaService, state_directory
from studio.api import create_app


def engine_fixture(tmp_path):
    cfg = Config(tmp_path)
    root = tmp_path/'carla'
    binary = root/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\nexit 0\n');binary.chmod(0o700)
    (root/'VERSION').write_text('0.9.15')
    maps = root/'CarlaUE4/Content/Carla/Maps';maps.mkdir(parents=True)
    (maps/'Town01.umap').touch()
    return cfg


def test_rendering_profiles_and_permission_checks(tmp_path, monkeypatch):
    cfg = engine_fixture(tmp_path)
    settings = cfg.settings()
    state = {'settings': settings, 'town': 'Town01'}
    command = engine_command(cfg, state)
    assert '-RenderOffScreen' in command
    assert '-quality-level=Low' not in command
    monkeypatch.delenv('DISPLAY', raising=False)
    settings['carla_render_profile'] = 'low'
    with pytest.raises(RuntimeError, match='DISPLAY'):engine_command(cfg,state)
    settings['carla_render_profile'] = 'offscreen'
    Path(command[0]).chmod(0o600)
    with pytest.raises(PermissionError):engine_command(cfg,state)


def test_rpc_readiness_requires_snapshot_and_matching_version(tmp_path):
    source = """class Client:
 def __init__(self,*args):pass
 def set_timeout(self,*args):pass
 def get_server_version(self):return '0.9.15'
 def get_client_version(self):return '0.9.15'
 def get_world(self):return World()
class World:
 def get_map(self):return type('Map',(),{'name':'Town01'})()
 def get_snapshot(self):raise RuntimeError('snapshot unavailable')
"""
    module = tmp_path/'carla.py';module.write_text(source)
    command = [sys.executable,'-B','-c',RPC_SCRIPT,'127.0.0.1','2000']
    failed = subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert not json.loads(failed.stdout)['connected']
    source=source.replace("raise RuntimeError('snapshot unavailable')", "return type('Snapshot',(),{'frame':42,'timestamp':type('Time',(),{'elapsed_seconds':1.4})()})()")
    module.write_text(source)
    passed = subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert json.loads(passed.stdout)['frame'] == 42
    module.write_text(source.replace("get_server_version(self):return '0.9.15'", "get_server_version(self):return '0.9.14'"))
    mismatch = subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert json.loads(mismatch.stdout)['fatal']


def test_repeated_segmentation_fault_stops_restart(tmp_path):
    cfg=Config(tmp_path);directory=state_directory(cfg)
    atomic_json(directory/'state.json',{'task':{'id':'old','status':'Failed'},'engine':{'running':False},
        'settings':cfg.settings(),'crash_streak':2,'worker':{}})
    service=CarlaService(cfg)
    with pytest.raises(ValueError,match='반복'):service.start('prepare')


def test_diagnostics_and_export_preserve_error_information(tmp_path,monkeypatch):
    import studio.carla_diagnostics as diagnostic_module
    monkeypatch.setattr(diagnostic_module,'diagnostics',lambda *a,**k:{'carla_version':'0.9.15','gpu':[],
        'vulkan':{'available':False,'error':'controlled ICD error'},'ready':False})
    app=create_app(tmp_path)
    with TestClient(app,base_url='http://127.0.0.1:8765') as client:
        assert client.post('/api/carla/diagnostics').status_code==403
        response=client.post('/api/carla/diagnostics',headers={'X-SDV-Request':'1'})
        assert response.json()['vulkan']['error']=='controlled ICD error'
        exported=client.get('/api/carla/diagnostics/export')
        assert 'attachment' in exported.headers['content-disposition']
        assert exported.json()['vulkan']['error']=='controlled ICD error'
