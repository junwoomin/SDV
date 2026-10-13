#!/usr/bin/env python3
"""CPU-only smoke check. Does not start CARLA or load PyTorch."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from studio.api import create_app

def main():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='sdv-web-check-') as tmp:
        app=create_app(Path(tmp))
        with TestClient(app) as client:
            bootstrap=client.get('/api/bootstrap')
            assert bootstrap.status_code==200, bootstrap.text
            data=bootstrap.json()
            assert {'settings','sensors','training','carla','jobs'} <= data.keys(), data.keys()
            invalid=client.put('/api/settings',json={'carla_port':70000},headers={'X-SDV-Request':'1'})
            assert invalid.status_code>=400, invalid.text
            for route in ['/api/status','/api/jobs','/api/experiments','/api/datasets']:
                response=client.get(route)
                assert response.status_code==200,(route,response.text)
            print(json.dumps({'ok':True,'hardware_workloads_started':False,'checked':['bootstrap','status','jobs','experiments','datasets','invalid port']},ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
