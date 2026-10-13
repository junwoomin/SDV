"""Test-only CPU adaptation of the actual CUDA-only research entry point.

Only the hardware guard and .cuda transfer calls are adapted. Uses actual SDV
Dataset/ResNet18/FPN/Seg/Depth/loss/optimizer/metrics/PNG/checkpoint code on
clearly synthetic fixtures. This does not verify any CUDA behavior.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = None
PYTHON = Path(sys.executable)


def runner(config, output, mode, fixture):
    global FIXTURE
    FIXTURE = Path(fixture).resolve()
    import random
    import numpy as np
    import torch
    sys.path.insert(0, str(PROJECT))
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    torch.set_num_threads(1)
    # Explicit test-only adaptation; production source still requires CUDA.
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    os.environ.update(SDV_TRAIN_DATA=str(FIXTURE/'train'), SDV_VAL_DATA=str(FIXTURE/'val'),
                      SDV_SENSOR_CONFIG=str(FIXTURE/'sensors.yaml'), SDV_TRAIN_OUTPUT=str(output),
                      SDV_TEST_TENSOR=str(FIXTURE/'qa_gt.pth'),
                      SDV_EVENTS_PATH=str(output.parent/(output.name+'_events.jsonl')), SDV_EVENTS_STDOUT='0')
    if mode == 'pause':
        control = FIXTURE/'control.json'
        control.write_text(json.dumps({'pause_requested': True}))
        os.environ['SDV_TRAIN_CONTROL'] = str(control)
    if mode == 'resume':
        os.environ['SDV_TRAIN_RESUME'] = str(output/'resume_checkpoint.pth')
    sys.argv = ['train2.py', '--config', str(config)]
    source = (PROJECT/'sdv/training/experiment.py').read_text()
    guard = 'if not torch.cuda.is_available():\n    raise RuntimeError("현재 SDV 연구 모델은 CUDA GPU가 필요합니다. CUDA가 지원되는 Python 환경을 선택하세요.")\n'
    assert source.count(guard) == 1
    source = source.replace(guard, '# CUDA guard disabled only in isolated CPU QA.\n', 1)
    exec(compile(source, str(PROJECT/'sdv/training/experiment.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(PROJECT/'sdv/training/experiment.py')})


def setup():
    import numpy as np
    from PIL import Image
    import yaml
    import torch
    from studio.config import Config
    FIXTURE.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        for camera in ('front_cam_rgb', 'front_cam_seg', 'front_cam_depth'):
            (FIXTURE/split/'fixture_scene'/camera).mkdir(parents=True)
        yy, xx = np.indices((64,64))
        palette = np.array(((102,102,156),(153,153,153),(250,170,30),
                            (220,220,0),(152,251,152),(110,190,160)), dtype=np.uint8)
        for index in range(4):
            rgb = np.stack(((xx*3+index*11)%256, (yy*3+index*7)%256,
                            ((xx+yy)*2+index*5)%256),axis=-1).astype(np.uint8)
            seg = palette[(xx//12+index)%6]
            depth = np.full((64,64), 32+index*8, dtype=np.uint8)
            for folder, pixels in (('front_cam_rgb',rgb),('front_cam_seg',seg),('front_cam_depth',depth)):
                Image.fromarray(pixels).save(str(FIXTURE/split/'fixture_scene'/folder/('{:05d}.png'.format(index))))
    config = Config.training_defaults(None)
    config.update(Epochs=2,batch=2,width=64,height=64)
    config['model'].update(encoder='ResNet18',pretrained=False)
    config['monitor'].update(train_preview_every=1,val_preview_every=1)
    (FIXTURE/'config.yaml').write_text(yaml.safe_dump(config))
    sensor = {'towns':['Town01'],'data_num':4,'sensors':{'front_cam':{'width':64,'height':64,
        'fov':90,'x':1.3,'y':0,'z':2.3,'pitch':0,'yaw':0,'roll':0,'seg':True,'depth':True,'od':False}}}
    (FIXTURE/'sensors.yaml').write_text(yaml.safe_dump(sensor))
    # Isolate legacy GT preview from a researcher's optional img/test.pth.
    torch.save({name: torch.zeros(0,1,2,2) for name in
                ('das_GT','lane_GT','vehicle_GT','walker_GT')}, str(FIXTURE/'qa_gt.pth'))
    disabled = dict(config)
    disabled['Epochs'] = 1
    disabled['monitor'] = dict(config['monitor'],train_preview_every=0)
    (FIXTURE/'disabled.yaml').write_text(yaml.safe_dump(disabled))


def main(report_path, keep_output):
    global FIXTURE
    import numpy as np
    import pandas as pd
    import torch
    import shutil
    sys.path.insert(0, str(PROJECT))
    runtime = PROJECT/'.runtime'
    runtime.mkdir(exist_ok=True)
    FIXTURE = Path(tempfile.mkdtemp(prefix='qa-training-monitor-', dir=str(runtime)))
    setup()
    started = time.monotonic()
    runs=[]
    for mode, output, config, expected in (
        ('full',FIXTURE/'full',FIXTURE/'config.yaml',0),
        ('pause',FIXTURE/'resumed',FIXTURE/'config.yaml',75),
        ('resume',FIXTURE/'resumed',FIXTURE/'config.yaml',0),
        ('full',FIXTURE/'disabled',FIXTURE/'disabled.yaml',0)):
        logfile = FIXTURE/(mode+'_'+output.name+'.log')
        with logfile.open('wb') as log:
            result=subprocess.run([str(PYTHON),__file__,'--allow-test-cpu-adaptation','--worker',str(config),str(output),mode,str(FIXTURE)],
                cwd=str(PROJECT),stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if result.returncode != expected:
            print(logfile.read_text()[-8000:])
            raise AssertionError('{} return code {} expected {}'.format(mode,result.returncode,expected))
        runs.append({'mode':mode,'return_code':result.returncode})
    full = torch.load(str(FIXTURE/'full/resume_checkpoint.pth'),map_location='cpu',weights_only=False)
    resumed = torch.load(str(FIXTURE/'resumed/resume_checkpoint.pth'),map_location='cpu',weights_only=False)
    assert full['completed_epoch'] == resumed['completed_epoch'] == 2
    assert full['global_step'] == resumed['global_step'] == 4
    assert all(torch.equal(value,resumed['model'][key]) for key,value in full['model'].items())
    assert full['histories'] == resumed['histories']
    assert full['scheduler'] == resumed['scheduler']
    for name in ('full','resumed'):
        csv = pd.read_csv(FIXTURE/name/'train_metrics.csv')
        assert csv.epoch.tolist() == [1,2]
        assert {'val_seg_miou','val_depth_rmse','train_depth_abs_rel'} <= set(csv.columns)
    fullcsv=pd.read_csv(FIXTURE/'full/train_metrics.csv').drop(columns=['epoch_duration_sec'])
    resumedcsv=pd.read_csv(FIXTURE/'resumed/train_metrics.csv').drop(columns=['epoch_duration_sec'])
    np.testing.assert_array_equal(fullcsv.to_numpy(),resumedcsv.to_numpy())
    fullindex=json.loads((FIXTURE/'full/predictions/index.json').read_text())
    resumedindex=json.loads((FIXTURE/'resumed/predictions/index.json').read_text())
    assert fullindex['datasets']['val']['samples'] == resumedindex['datasets']['val']['samples']
    assert fullindex['datasets']['val']['count'] == 4
    assert len(resumedindex['entries']) == len(fullindex['entries']) == 48
    assert {entry['epoch'] for entry in resumedindex['entries'] if entry['split']=='val'} == {1,2}
    assert [e['id'] for e in fullindex['entries']] == [e['id'] for e in resumedindex['entries']]
    assert [e['sample_id'] for e in fullindex['entries']] == [e['sample_id'] for e in resumedindex['entries']]
    for entry in resumedindex['entries']:
        assert all((FIXTURE/'resumed'/path).is_file() for path in entry['images'].values())
    disabledindex=json.loads((FIXTURE/'disabled/predictions/index.json').read_text())
    assert {entry['split'] for entry in disabledindex['entries']} == {'val'}
    assert not list((FIXTURE/'disabled').rglob('save_samples*'))
    events=[json.loads(line) for line in (FIXTURE/'resumed_events.jsonl').read_text().splitlines()]
    assert [e['epoch'] for e in events if e['event']=='training_epoch'] == [1,2]
    assert [e['epoch'] for e in events if e['event']=='training_paused'] == [1]
    assert [e['epoch'] for e in events if e['event']=='training_resumed'] == [2]
    assert all(e['duration_seconds'] > 0 for e in events if e['event']=='training_epoch')
    report={'passed':True,'hardware':'CPU','cuda_verified':False,'carla_verified':False,
        'adaptation':'Explicit test-only Tensor/Module.cuda no-op and CUDA availability guard removed in memory; production unchanged',
        'fixture':'Generated synthetic 64x64 RGB/Seg/Depth, 4 actual files each train/val; not research accuracy results',
        'model':'actual SDV ResNet18 + FPN + Seg/Depth','epochs':2,'runs':runs,
        'global_steps':4,'fixed_val_samples':4,'prediction_entries':len(resumedindex['entries']),
        'checks':['uninterrupted/resumed model state exactly equal','all task metrics and histories exactly equal',
                  'StepLR restored','epochs 1,2 CSV append','same fixed4 samples across jobs',
                  'prediction history preserved after resume','all input/GT/pred PNGs exist',
                  'train interval0 suppresses all train prediction output','pause/resume events at correct completed epoch',
                  'empirical epoch duration positive'], 'elapsed_seconds':time.monotonic()-started}
    report['python'] = sys.version.split()[0]
    report['torch'] = torch.__version__
    report['fixture_directory'] = str(FIXTURE) if keep_output else 'removed after verification'
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    if not keep_output:
        shutil.rmtree(str(FIXTURE))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-test-cpu-adaptation', action='store_true',
                        help='Explicitly permit this isolated CUDA-to-CPU test adaptation.')
    parser.add_argument('--keep-output', action='store_true', help='Keep generated fixtures/checkpoints for debugging.')
    parser.add_argument('--report', type=Path, default=PROJECT/'docs/training-engine-cpu-validation.json')
    parser.add_argument('--worker', nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.allow_test_cpu_adaptation:
        parser.error('This test adapts CUDA calls to CPU; pass --allow-test-cpu-adaptation explicitly. It does not verify CUDA.')
    if args.worker:
        config, output, mode, fixture = args.worker
        runner(Path(config), Path(output), mode, fixture)
    else:
        main(args.report, args.keep_output)
