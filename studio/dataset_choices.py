"""Read-only, project-local train/val pairs for the default training flow."""
import hashlib
import json
import os
from pathlib import Path


def _path(root,value):
    path=Path(value).expanduser();path=path if path.is_absolute() else root/path
    cursor=path
    while cursor!=root and cursor!=cursor.parent:
        if cursor.is_symlink():raise ValueError('데이터셋 심볼릭 링크는 사용할 수 없습니다.')
        cursor=cursor.parent
    path=path.resolve();path.relative_to(root)
    return path


def _split(root,path):
    result={'path':str(path),'samples':0,'scenes':0,'available':False,'issues':[]}
    try:
        path=_path(root,path)
        if not path.is_dir():raise ValueError('분할 폴더가 없습니다.')
        for folder,dirs,files in os.walk(str(path),followlinks=False):
            base=Path(folder);dirs[:]=[d for d in dirs if not d.startswith('.')]
            if any((base/d).is_symlink() for d in dirs):raise ValueError('분할 안에 심볼릭 링크가 있습니다.')
            if 'front_cam_rgb' in dirs:
                count=0
                for p in (base/'front_cam_rgb').iterdir():
                    if p.is_symlink():raise ValueError('RGB 파일 링크가 있습니다.')
                    if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg') and p.stat().st_size>0:count+=1
                result['samples']+=count;result['scenes']+=1;dirs[:]=[]
        if not result['samples']:raise ValueError('실제 RGB 표본이 없는 빈 분할입니다.')
        result['available']=True
    except (OSError,ValueError,TypeError) as error:result['issues'].append(str(error))
    return result


def dataset_choices(root,settings):
    root=Path(root).resolve();pairs={}
    data=root/'data'
    if data.is_dir():
        scanned=0
        for folder,dirs,files in os.walk(str(data),followlinks=False):
            scanned+=1
            if scanned>5000:break
            base=Path(folder);depth=len(base.relative_to(data).parts)
            dirs[:]=[d for d in dirs if not d.startswith('.') and not (base/d).is_symlink()]
            if 'train' in dirs or 'val' in dirs:
                pairs[(str(base/'train'),str(base/'val'))]=(base.name,base)
                dirs[:]=[d for d in dirs if d not in ('train','val')]
            if depth>=4 or 'front_cam_rgb' in dirs:dirs[:]=[]
    try:
        train=_path(root,settings.get('train_data','data/train'));val=_path(root,settings.get('val_data','data/val'))
        pairs.setdefault((str(train),str(val)),('현재 저장 경로',None))
    except (ValueError,OSError):pass
    try:current_pair=(str(_path(root,settings.get('train_data','data/train'))),str(_path(root,settings.get('val_data','data/val'))))
    except (ValueError,OSError):current_pair=None
    rows=[]
    for (train,val),(name,base) in sorted(pairs.items()):
        splits={key:_split(root,path) for key,path in [('train',train),('val',val)]}
        sensor=base/'sensor_config.yaml' if base else None
        if base is None:
            try:sensor=_path(root,settings.get('sensor_config','data/sensor_config.yaml'))
            except (ValueError,OSError):sensor=None
        manifest={}
        if base and (base/'dataset_manifest.json').is_file():
            try:
                if (base/'dataset_manifest.json').stat().st_size<=2*1024*1024:
                    value=json.loads((base/'dataset_manifest.json').read_text())
                    if isinstance(value,dict):manifest={k:value.get(k) for k in ('split_method','frame_randomization','calibration_sha256')}
            except (OSError,ValueError):pass
        available=all(s['available'] for s in splits.values())
        rows.append({'id':hashlib.sha256((train+'\0'+val).encode()).hexdigest()[:20],
            'current':(train,val)==current_pair,'name':name,'path':str(base.relative_to(root)) if base else 'custom saved paths',
            'train_data':train,'val_data':val,'sensor_config':str(sensor) if sensor and sensor.is_file() else None,
            'splits':splits,'available':available,'issues':[key+': '+msg for key,s in splits.items() for msg in s['issues']],
            'metadata':manifest,'validation_scope':'RGB inventory / safe paths; task labels and calibration require preflight'})
    return {'datasets':rows}
