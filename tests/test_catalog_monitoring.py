"""Saved experiment metadata and best metrics use immutable real run files."""
import json

import pytest
import yaml

from studio.catalog import Catalog


def _run(tmp_path, name='job_monitor'):
    run = tmp_path / 'results' / name
    (run / 'config').mkdir(parents=True)
    (run / 'config/settings.json').write_text('{}')
    (run / 'output').mkdir()
    return run, Catalog(tmp_path, {'results_root': 'results'})


def _config(encoder='ResNet50', optimizer='AdamW'):
    return {'Epochs': 20, 'batch': 4, 'LR': 0.0002,
            'model': {'encoder': encoder, 'decoder': 'UNet', 'bev_decoder': None},
            'model_conf': {'BEV': False, 'Seg': True, 'Depth': True, 'od': False},
            'sensor_use': {'front_cam': True, 'left_cam': False, 'rear_cam': True, 'lidar': False},
            'opt_checkboxes': {name: name == optimizer for name in ('Adam', 'AdamW', 'SGD')}}


def test_collection_snapshot_is_not_a_training_experiment(tmp_path):
    run, catalog = _run(tmp_path, 'job_collection')
    (run / 'config/experiment.json').write_text(json.dumps({'kind':'collection','name':'Real sensors'}))
    (run / 'config/train_config.yaml').write_text(yaml.safe_dump(_config()))
    assert catalog.experiments() == []


def test_declared_training_and_legacy_results_remain_visible(tmp_path):
    run, catalog = _run(tmp_path, 'job_training')
    (run / 'config/experiment.json').write_text(json.dumps({'kind':'training','name':'Real CUDA'}))
    assert catalog.experiments()[0]['name'] == 'Real CUDA'


def test_experiment_list_and_detail_use_saved_name_and_configuration(tmp_path):
    run, catalog = _run(tmp_path)
    (run / 'config/experiment.json').write_text(json.dumps({'name': 'Depth + Seg / ResNet50'}))
    (run / 'config/train_config.yaml').write_text(yaml.safe_dump(_config()))
    (run / 'output/train_metrics.csv').write_text('epoch,val_loss\n1,0.8\n2,0.6\n')
    row = catalog.experiments()[0]
    assert row['name'] == 'Depth + Seg / ResNet50'
    assert row['optimizer'] == 'AdamW'
    assert row['encoder'] == 'ResNet50'
    assert row['decoder'] == 'UNet'
    assert row['bev_decoder'] is None
    assert row['learning_rate'] == 0.0002
    assert row['batch'] == 4
    assert row['configured_epochs'] == 20
    assert row['epochs'] == 2
    assert row['selected_sensors'] == ['front_cam', 'rear_cam']
    assert row['tasks'] == ['Seg', 'Depth']
    assert row['config_source'] == 'config/train_config.yaml'
    assert row['status'] == 'Recorded'
    detail = catalog.experiment(row['id'])
    assert detail['optimizer'] == 'AdamW'
    assert detail['config']['config/experiment.json']['name'] == row['name']


@pytest.mark.parametrize('source', ['config/train_config.yaml', 'output/train_config.yaml', 'train_config.yaml'])
def test_legacy_configuration_paths_are_supported_without_current_settings(tmp_path, source):
    run, catalog = _run(tmp_path)
    (run / source).write_text(yaml.safe_dump(_config('MobileNetV3-Small', 'SGD')))
    row = catalog.experiments()[0]
    assert row['encoder'] == 'MobileNetV3-Small'
    assert row['optimizer'] == 'SGD'
    assert row['config_source'] == source


def test_configuration_snapshot_has_priority_over_output_and_legacy(tmp_path):
    run, catalog = _run(tmp_path)
    (run / 'config/train_config.yaml').write_text(yaml.safe_dump(_config('ResNet18', 'SGD')))
    (run / 'output/train_config.yaml').write_text(yaml.safe_dump(_config('ResNet50', 'AdamW')))
    (run / 'train_config.yaml').write_text(yaml.safe_dump(_config('ResNet101', 'Adam')))
    row = catalog.experiments()[0]
    assert row['encoder'] == 'ResNet18'
    assert row['optimizer'] == 'SGD'


def test_malformed_configuration_falls_back_with_clear_warning(tmp_path):
    run, catalog = _run(tmp_path)
    (run / 'config/train_config.yaml').write_text('model: [unterminated')
    (run / 'output/train_config.yaml').write_text(yaml.safe_dump(_config()))
    row = catalog.experiments()[0]
    assert row['config_source'] == 'output/train_config.yaml'
    assert any('config/train_config.yaml' in warning for warning in row['warnings'])


def test_no_configuration_does_not_invent_optimizer_model_or_tasks(tmp_path):
    run, catalog = _run(tmp_path)
    row = catalog.experiments()[0]
    assert row['name'] == run.name
    assert row['optimizer'] is None
    assert row['encoder'] is None
    assert row['learning_rate'] is None
    assert row['batch'] is None
    assert row['tasks'] == row['selected_sensors'] == []
    assert row['best_metrics'] == {}


def test_multiple_optimizer_flags_remain_unknown_not_guessed(tmp_path):
    run, catalog = _run(tmp_path)
    config = _config()
    config['opt_checkboxes']['Adam'] = True
    (run / 'config/train_config.yaml').write_text(yaml.safe_dump(config))
    row = catalog.experiments()[0]
    assert row['optimizer'] is None
    assert any('옵티마이저' in warning for warning in row['warnings'])


def test_best_metrics_choose_min_loss_depth_and_max_task_quality(tmp_path):
    run, catalog = _run(tmp_path)
    (run / 'output/train_metrics.csv').write_text(
        'epoch,val_loss,val_seg_miou,val_depth_rmse,val_od_ap50_95,unknown_score\n'
        '1,0.8,0.1,3.5,0.2,9\n'
        '2,0.4,0.7,1.2,0.8,8\n'
        '3,0.5,0.6,2.0,0.6,7\n')
    row = catalog.experiments()[0]
    assert row['best_metrics'] == {
        'val_loss': {'value': 0.4, 'epoch': 2, 'direction': 'min'},
        'val_seg_miou': {'value': 0.7, 'epoch': 2, 'direction': 'max'},
        'val_depth_rmse': {'value': 1.2, 'epoch': 2, 'direction': 'min'},
        'val_od_ap50_95': {'value': 0.8, 'epoch': 2, 'direction': 'max'}}
    assert row['latest_metrics']['val_loss'] == 0.5
    assert 'observed_unknown_score' not in row['best_metrics']


def test_best_metrics_preserve_actual_zero_and_ignore_nonfinite(tmp_path):
    run, catalog = _run(tmp_path)
    (run / 'output/train_metrics.csv').write_text(
        'epoch,val_loss,val_seg_miou\n1,0,0\n2,NaN,Infinity\n3,0.1,0.2\n')
    best = catalog.experiments()[0]['best_metrics']
    assert best['val_loss'] == {'value': 0, 'epoch': 1, 'direction': 'min'}
    assert best['val_seg_miou'] == {'value': 0.2, 'epoch': 3, 'direction': 'max'}


def test_best_metrics_computed_before_graph_downsampling(tmp_path):
    run, catalog = _run(tmp_path)
    rows = ['epoch,val_loss,val_seg_miou']
    for epoch in range(1, 5003):
        rows.append('{},{},{}'.format(epoch, 0.001 if epoch == 2 else 1,
                                      0.99 if epoch == 2 else 0.1))
    (run / 'output/train_metrics.csv').write_text('\n'.join(rows))
    detail = catalog.experiment(catalog.experiments()[0]['id'])
    assert detail['best_metrics']['val_loss'] == {'value': 0.001, 'epoch': 2, 'direction': 'min'}
    assert detail['best_metrics']['val_seg_miou'] == {'value': 0.99, 'epoch': 2, 'direction': 'max'}
    for series in detail['metrics']:
        assert series['total_points'] == 5002
        assert len(series['points']) <= 2000
        assert 2 not in [point['x'] for point in series['points']]
        assert series['points'][-1]['x'] == 5002
        assert series['best']['epoch'] == 2


def test_best_ties_use_earliest_recorded_epoch_even_if_rows_out_of_order(tmp_path):
    run, catalog = _run(tmp_path)
    (run / 'output/train_metrics.csv').write_text('epoch,val_loss\n3,0.2\n1,0.2\n2,0.3\n')
    assert catalog.experiments()[0]['best_metrics']['val_loss']['epoch'] == 1


def test_event_best_excludes_batch_values_and_unknown_metrics(tmp_path):
    run, catalog = _run(tmp_path)
    events = [
        {'event': 'training_step', 'epoch': 1, 'global_step': 10, 'metrics': {'train_loss': 0.001}},
        {'event': 'training_epoch', 'epoch': 1, 'metrics': {'train_loss': 0.7, 'val_seg_miou': 0.3}},
        {'event': 'training_epoch', 'epoch': 2, 'metrics': {'train_loss': 0.4, 'val_seg_miou': 0.6}},
        {'event': 'training_epoch', 'epoch': 3, 'metrics': {'mysterious_metric': 100}},
    ]
    (run / 'output/events.jsonl').write_text('\n'.join(json.dumps(event) for event in events))
    best = catalog.experiments()[0]['best_metrics']
    assert best['train_loss'] == {'value': 0.4, 'epoch': 2, 'direction': 'min'}
    assert best['val_seg_miou'] == {'value': 0.6, 'epoch': 2, 'direction': 'max'}
    assert len(best) == 2


def test_disabled_tasks_are_excluded_from_best_metrics(tmp_path):
    run, catalog = _run(tmp_path)
    config = _config()
    config['model_conf']['Depth'] = False
    (run / 'config/train_config.yaml').write_text(yaml.safe_dump(config))
    (run / 'output/train_metrics.csv').write_text('epoch,val_loss,val_depth_rmse,val_seg_miou\n1,0.8,2,0.3\n')
    best = catalog.experiments()[0]['best_metrics']
    assert 'val_depth_rmse' not in best
    assert 'val_seg_miou' in best


def test_metadata_and_configuration_symlinks_cannot_read_outside_run(tmp_path):
    run, catalog = _run(tmp_path)
    (tmp_path / 'secret.json').write_text(json.dumps({'name': 'Private name'}))
    (run / 'config/experiment.json').symlink_to(tmp_path / 'secret.json')
    (tmp_path / 'secret.yaml').write_text(yaml.safe_dump(_config()))
    (run / 'config/train_config.yaml').symlink_to(tmp_path / 'secret.yaml')
    row = catalog.experiments()[0]
    assert row['name'] == run.name
    assert row['encoder'] is None
    assert any('허용된 폴더 밖' in warning for warning in row['warnings'])


def test_legacy_log_and_web_history_do_not_invent_epoch_best(tmp_path):
    run, catalog = _run(tmp_path)
    (run / 'output/results.log').write_text('TRAIN[1] [10] Loss: 0.2\nVAL[1] Loss: 0.3\n')
    assert catalog.experiments()[0]['best_metrics'] == {}
    (run / 'output/metrics_history.json').write_text(json.dumps(
        {'tasks': {'Seg': {'iou': {'x': [1, 2], 'train': [.2, .3], 'val': [.1, .4]}}}}))
    catalog.invalidate()
    assert catalog.experiments()[0]['best_metrics'] == {}
