"""Deletion is scoped to a confirmed scene and immutable pending job paths."""
import json
from pathlib import Path

import pytest

from test_sequence_api import sequence_app, HEADERS, record


def delete(client,identifier,name='scene_a',headers=HEADERS):
    return client.request('DELETE','/api/datasets/'+identifier,
                          json={'confirmation_name':name},headers=headers)


def pending(app,status,kind,path):
    app.state.jobs.reconcile=lambda:None
    job={'id':'protect-data','name':'Referenced experiment','kind':kind,'status':status,
         'output_dir':str(path),'settings':{'train_data':str(path),'val_data':'data/other-validation'}}
    with app.state.jobs.db() as db:
        db.execute('INSERT INTO jobs(id,status,created_at,payload) VALUES(?,?,?,?)',
                   (job['id'],status,'2026-10-12',json.dumps(job)))


def test_delete_exact_scene_and_invalidate_catalog_and_images(sequence_app):
    app,client,scene,ids=sequence_app
    identifier=ids['scene_a'];base='/api/datasets/'+identifier
    assert client.get(base+'/frame?index=0&channel=rgb').status_code==200
    assert app.state.sequences._images
    info=client.get(base+'/deletion').json()
    assert info['can_delete'] and info['path'].endswith('scene_a')
    result=delete(client,identifier)
    assert result.status_code==200,result.text
    assert result.json()['deleted'] is True
    assert not scene.exists()
    assert not app.state.sequences._images
    assert client.get(base+'/sequence').status_code==404
    assert client.get(base+'/frame?index=0&channel=rgb').status_code==404
    rows=client.get('/api/datasets').json()['datasets']
    assert [row['name'] for row in rows]==['scene_b']
    assert client.get('/api/datasets/'+ids['scene_b']+'/sequence').status_code==200
    assert delete(client,identifier).status_code==404


@pytest.mark.parametrize('name',['wrong-name','scene_a ',None])
def test_exact_confirmation_required(sequence_app,name):
    app,client,scene,ids=sequence_app
    assert delete(client,ids['scene_a'],name).status_code==422
    assert (scene/'front_cam_rgb/2_a.png').is_file()


def test_mutation_header_and_body_required(sequence_app):
    app,client,scene,ids=sequence_app
    assert delete(client,ids['scene_a'],headers={}).status_code==403
    assert client.request('DELETE','/api/datasets/'+ids['scene_a'],json={},headers=HEADERS).status_code==422
    assert scene.exists()


@pytest.mark.parametrize('status',['Running','Pausing','Queued','Paused'])
def test_training_snapshots_protect_referenced_scene(sequence_app,status):
    app,client,scene,ids=sequence_app
    pending(app,status,'training',scene.parent)
    # Current settings differ; protection must use the saved experiment paths.
    info=client.get('/api/datasets/'+ids['scene_a']+'/deletion').json()
    assert not info['can_delete'] and status in info['blockers'][0]
    response=delete(client,ids['scene_a'])
    assert response.status_code==409,response.text
    assert scene.exists()


def test_active_collection_protects_output_descendants(sequence_app):
    app,client,scene,ids=sequence_app
    pending(app,'Running','collection',scene.parents[1])
    assert delete(client,ids['scene_a']).status_code==409
    assert scene.exists()


def test_unrelated_active_training_does_not_block_delete(sequence_app):
    app,client,scene,ids=sequence_app
    pending(app,'Running','training',scene.parents[2]/'unrelated')
    assert delete(client,ids['scene_a']).status_code==200


def test_contained_symlink_does_not_delete_external_target(sequence_app,tmp_path):
    app,client,scene,ids=sequence_app
    external=tmp_path/'keep.txt';external.write_text('keep')
    (scene/'external-link').symlink_to(external)
    assert delete(client,ids['scene_a']).status_code==200
    assert external.read_text()=='keep'


def test_storage_root_cannot_be_deleted_even_if_it_looks_like_scene(sequence_app):
    app,client,scene,ids=sequence_app
    root=app.state.config.path(app.state.config.settings()['data_root'])
    record(root,'1_a',(1,2,3))
    app.state.catalog.invalidate()
    identifier=next(row['id'] for row in client.get('/api/datasets').json()['datasets'] if row['name']==root.name)
    response=delete(client,identifier,root.name)
    assert response.status_code==422,response.text
    assert root.exists() and scene.exists()


def test_project_external_catalog_entry_cannot_be_deleted(sequence_app,tmp_path,monkeypatch):
    app,client,scene,ids=sequence_app
    external=tmp_path.parent/'outside-delete-guard';external.mkdir(exist_ok=True)
    monkeypatch.setattr(app.state.catalog,'_scene_paths',lambda:{'external':(external,'collection')})
    assert delete(client,'external',external.name).status_code==422
    assert external.exists()


def test_delete_permission_failure_is_actionable_and_restores_scene(sequence_app,monkeypatch):
    app,client,scene,ids=sequence_app
    def denied(path):raise PermissionError('controlled permission failure')
    monkeypatch.setattr('studio.dataset_delete.shutil.rmtree',denied)
    response=delete(client,ids['scene_a'])
    assert response.status_code==422,response.text
    assert '권한' in response.json()['detail']
    assert (scene/'front_cam_rgb/2_a.png').exists()
    assert not list(scene.parent.glob('.sdv-delete-*'))
