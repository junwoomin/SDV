"""Delete exactly one recorded scene, with immutable job-path protection."""
import json
import shutil
import uuid
from pathlib import Path


def deletion_target(catalog, identifier):
    entry=catalog._scene_paths().get(identifier)
    if entry is None: raise FileNotFoundError('수집 장면을 찾을 수 없습니다.')
    path,kind=entry
    if kind!='collection': raise ValueError('주행 시퀀스 삭제는 원본 수집 Scene만 지원합니다.')
    absolute=path.absolute()
    if absolute==catalog.root or catalog.root not in absolute.parents:
        raise ValueError('프로젝트 외부 데이터는 삭제할 수 없습니다.')
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink(): raise ValueError('심볼릭 링크 경로는 삭제할 수 없습니다.')
        if candidate==catalog.root: break
    path=absolute.resolve()
    protected={catalog.root,catalog.root/'data',catalog.root/'data/prepared'}
    protected.update(catalog.path(catalog.settings.get(key,default)) for key,default in
        [('data_root','data/collections'),('train_data','data/train'),('val_data','data/val'),
         ('results_root','results'),('logs_root','.runtime/studio/logs')])
    if any(path==root or path in root.parents for root in protected):
        raise ValueError('저장 루트나 상위 폴더는 삭제할 수 없습니다. 개별 Scene을 선택하세요.')
    return path


def _overlaps(left,right):
    return left==right or left in right.parents or right in left.parents


def deletion_blockers(catalog,jobs,path):
    # Read all pending snapshots; the list UI's 200-row limit is inappropriate
    # for protecting data referenced by a paused or queued experiment.
    with jobs.db() as db:
        pending=[json.loads(row['payload']) for row in db.execute(
            "SELECT payload FROM jobs WHERE status IN ('Running','Pausing','Queued','Paused')")]
    blockers=[]
    for job in pending:
        if job.get('queue_removed'): continue
        references=[]
        if job.get('kind')=='collection':
            if job.get('output_dir'): references.append(Path(job['output_dir']).resolve())
        elif job.get('kind')=='training':
            settings=job.get('settings') or {}
            references.extend(catalog.path(settings[key]) for key in ('train_data','val_data') if settings.get(key))
        if any(_overlaps(path,reference) for reference in references):
            blockers.append('{} · {} · {} 작업이 이 데이터를 사용합니다. 작업을 완료하거나 대기열에서 제거한 후 삭제하세요.'.format(
                job.get('name') or job.get('id'),job.get('id'),job['status']))
    return blockers


def delete_scene(catalog,path,confirmation_name):
    if not isinstance(confirmation_name,str) or confirmation_name!=path.name:
        raise ValueError('삭제할 Scene 이름을 정확히 입력하세요.')
    # Same-directory rename is atomic and hides a partial deletion from readers.
    # rmtree unlinks contained symlinks; it does not follow their targets.
    tombstone=path.with_name('.sdv-delete-'+uuid.uuid4().hex)
    try:path.rename(tombstone)
    except OSError as error:
        raise ValueError('삭제를 시작할 수 없습니다. 폴더 권한과 사용 상태를 확인하세요: '+str(error)) from error
    try:
        shutil.rmtree(str(tombstone))
    except OSError as error:
        if tombstone.exists() and not path.exists():
            try:tombstone.rename(path)
            except OSError as restore_error:
                raise ValueError('삭제 및 폴더 복구를 완료하지 못했습니다. 남은 파일 위치: {}. {}'.format(tombstone,restore_error)) from error
        raise ValueError('삭제를 완료하지 못했습니다. 일부 파일은 이미 삭제되었을 수 있습니다. 권한과 파일 상태를 확인하세요: '+str(error)) from error
    finally:
        catalog.invalidate()
