"""Connect the UI queue to the existing train2.py, with honest preflight failures."""
import argparse
import json
import yaml
from pathlib import Path
import subprocess
import sys
from sdv.runtime.processes import prepare,ROOT
from sdv.config.validation import read_train_config,training_blockers,read_sensor_config

def main():
    prepare()
    parser=argparse.ArgumentParser(description='SDV 학습 대기열 검사 및 실행')
    parser.add_argument('--check',action='store_true',help='GPU를 시작하지 않고 준비 파일과 설정만 검사')
    parser.add_argument('--config',type=Path,help='실행할 대기열 YAML 하나')
    args=parser.parse_args()
    configs=[args.config] if args.config else sorted((ROOT/'data/loop').glob('train_config_*.yaml'))
    report={'ready':False,'entrypoint':'train2.py','experiments':[],'blockers':[]}
    if not configs:report['blockers'].append('저장된 실험 대기열이 없습니다.')
    try:read_sensor_config()
    except (OSError,ValueError,yaml.YAMLError) as error:report['blockers'].append('센서 설정: '+str(error))
    for path in configs:
        try:
            config=read_train_config(path)
            blockers=training_blockers(config)
            report['experiments'].append({'config':str(path),'epochs':config['Epochs'],'batch':config['batch'],'lr':config['LR'],'optimizer':next(k for k,v in config['opt_checkboxes'].items() if v),'blockers':blockers})
            report['blockers'].extend(blockers)
        except (OSError,ValueError,KeyError,yaml.YAMLError) as error:report['blockers'].append(str(error))
    report['blockers']=list(dict.fromkeys(report['blockers']))
    report['ready']=not report['blockers']
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not report['ready']:return 2
    if args.check:return 0
    for config_path in configs:
        completed=subprocess.run([sys.executable,'-B','-u',str(ROOT/'train2.py'),'--config',str(config_path.resolve())],cwd=ROOT)
        if completed.returncode:return completed.returncode
    return 0
if __name__=='__main__':sys.exit(main())
