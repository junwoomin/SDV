"""Isolated artifacts exercise the catalog; fixtures are never shipped as results."""
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from studio.catalog import Catalog, _series_from_csv, _series_from_history, _series_from_log, _series_from_events
from studio.dataset_tools import validate_scene, validate_training_inputs


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = Catalog(self.root, {'data_root': 'collections', 'results_root': 'results'})
        self.sensor = {'sensors': {'front_cam': {'width': 400, 'height': 300, 'fov': 100, 'x': 1.3, 'y': 0, 'z': 2.3, 'yaw': 0, 'pitch': 0, 'roll': 0}}}
        self.config = {'sensor_use': {'front_cam': True}, 'model_conf': {'BEV': True}}

    def tearDown(self):
        self.temp.cleanup()

    def scene(self, name, x=1.3):
        from PIL import Image
        scene = self.root / 'collections' / 'Town01' / name
        for channel in ('front_cam_rgb', 'das', 'lane', 'vehicle_mask', 'walker_mask'):
            (scene / channel).mkdir(parents=True)
            for frame in ('00001', '00002'):
                Image.new('RGB' if channel == 'front_cam_rgb' else 'L', (400, 300) if channel == 'front_cam_rgb' else (500, 500)).save(scene / channel / f'{frame}.png')
        config = json.loads(json.dumps(self.sensor))
        config['sensors']['front_cam']['x'] = x
        (scene / 'sensor_config.yaml').write_text(yaml.safe_dump(config))
        return scene

    def test_csv_preserves_real_zero_skips_nonfinite_and_malformed_epoch(self):
        path = self.root / 'train_metrics.csv'
        path.write_text('epoch,train_loss,val_loss\n1,0,0.5\n2,NaN,0.25\n,9,9\n')
        result = _series_from_csv(path)
        self.assertEqual(result[0]['points'], [{'x': 1., 'y': 0.}])
        self.assertEqual(result[1]['points'][-1], {'x': 2., 'y': .25})

    def test_original_web_history_format(self):
        path = self.root / 'metrics_history.json'
        path.write_text(json.dumps({'tasks': {'BEV': {'iou': {'x': [1, 2], 'train': [.2, .3], 'val': [.1, None]}}}}))
        result = _series_from_history(path)
        self.assertEqual(result[1]['points'], [{'x': 1., 'y': .1}])

    def test_log_batch_values_never_labeled_as_epoch_means(self):
        path = self.root / 'results.log'
        path.write_text('TRAIN[1] [10] [1/100] Loss: 0.450 lane_iou 0.3\nVAL[1] Loss: 0.55\n')
        result = _series_from_log(path)
        self.assertEqual(result[0]['granularity'], 'logged_batch')
        self.assertEqual(result[0]['points'][0]['x'], 10)
        self.assertEqual(result[-1]['x_label'], 'epoch (logged batch value)')

    def test_event_incomplete_tail_is_ignored(self):
        path = self.root / 'events.jsonl'
        path.write_text(json.dumps({'event': 'training_epoch', 'epoch': 1, 'metrics': {'train_loss': 0.4}}) + '\n{"epoch":')
        self.assertEqual(_series_from_events(path)[0]['points'], [{'x': 1., 'y': .4}])

    def test_step_events_use_global_steps_and_keep_epoch_series_separate(self):
        path = self.root / 'events.jsonl'
        events = [
            {'event': 'training_step', 'epoch': 1, 'step': 1, 'global_step': 10, 'metrics': {'train_loss': .4}},
            {'event': 'training_step', 'epoch': 1, 'step': 2, 'global_step': 11, 'metrics': {'train_loss': .3}},
            {'event': 'training_epoch', 'epoch': 1, 'metrics': {'train_loss': .35}},
        ]
        path.write_text('\n'.join(json.dumps(event) for event in events))
        result = _series_from_events(path)
        self.assertEqual([p['x'] for p in result[0]['points']], [10., 11.])
        self.assertEqual(result[0]['x_label'], 'global step')
        self.assertEqual(result[0]['granularity'], 'batch')
        self.assertEqual(result[1]['x_label'], 'epoch')

    def test_artifact_rejects_traversal_and_symlink(self):
        run = self.root / 'results' / 'run1'
        run.mkdir(parents=True)
        (run / 'train_metrics.csv').write_text('epoch,train_loss\n1,.2\n')
        (self.root / 'private.txt').write_text('private')
        (run / 'escape.txt').symlink_to(self.root / 'private.txt')
        identifier = self.catalog.experiments()[0]['id']
        with self.assertRaises(ValueError):
            self.catalog.artifact(identifier, '../../private.txt')
        with self.assertRaises(ValueError):
            self.catalog.artifact(identifier, 'escape.txt')
        self.assertEqual(len(self.catalog.experiment(identifier)['artifacts']), 1)

    def test_preparation_preserves_masks_source_and_scene_split(self):
        self.scene('a')
        self.scene('b')
        rows = self.catalog.datasets()
        train, val = rows[0]['id'], rows[1]['id']
        result = self.catalog.prepare_dataset('study', [train], [val], self.config)
        self.assertEqual(result['samples'], {'train': 2, 'val': 2})
        self.assertTrue((Path(result['settings']['sensor_config'])).is_file())
        self.assertFalse(list((self.root / 'data/prepared/study').glob('*/*/bev')))
        self.assertEqual(len(list((self.root / 'collections').glob('*/*/das/*.png'))), 4)
        original = self.catalog.scene_path(train) / 'das/00001.png'
        copied = next(Path(result['settings']['train_data']).glob('*/das/00001.png'))
        self.assertEqual(original.read_bytes(), copied.read_bytes())
        self.assertNotEqual(original.stat().st_ino, copied.stat().st_ino)
        self.assertEqual(validate_training_inputs(self.root, result['settings'], self.config), [])
        wrong = yaml.safe_load(Path(result['settings']['sensor_config']).read_text())
        wrong['sensors']['front_cam']['x'] = 9
        Path(result['settings']['sensor_config']).write_text(yaml.safe_dump(wrong))
        self.assertTrue(any('선택한 학습 보정 파일' in value for value in validate_training_inputs(self.root, result['settings'], self.config)))

    def test_preparation_rejects_duplicate_leakage_and_calibration_change(self):
        self.scene('a')
        self.scene('b', x=2.0)
        rows = self.catalog.datasets()
        with self.assertRaisesRegex(ValueError, '동시에'):
            self.catalog.prepare_dataset('study', [rows[0]['id']], [rows[0]['id']], self.config)
        with self.assertRaisesRegex(ValueError, '보정값'):
            self.catalog.prepare_dataset('study', [rows[0]['id']], [rows[1]['id']], self.config)

    def test_validation_rejects_misaligned_frame_and_missing_snapshot(self):
        scene = self.scene('a')
        (scene / 'lane/00002.png').unlink()
        with self.assertRaisesRegex(ValueError, 'frame ID'):
            validate_scene(scene, self.config)
        (scene / 'lane/00001.png').replace(scene / 'lane/00002.png')
        (scene / 'sensor_config.yaml').unlink()
        with self.assertRaises(ValueError):
            validate_scene(scene, self.config)

    def test_scan_never_marks_file_presence_completed(self):
        run = self.root / 'Training_result' / 'legacy'
        run.mkdir(parents=True)
        (run / 'train_metrics.csv').write_text('epoch,train_loss\n1,.5\n')
        self.assertEqual(self.catalog.experiments()[0]['status'], 'Recorded')

    def test_job_output_metrics_and_configuration_are_one_experiment(self):
        run = self.root / 'results' / 'job_123'
        (run / 'output').mkdir(parents=True)
        (run / 'config').mkdir()
        (run / 'events.jsonl').write_text('{"event":"started"}\n')
        (run / 'output/train_metrics.csv').write_text('epoch,train_loss,val_loss\n1,.5,.6\n')
        (run / 'config/train_config.yaml').write_text('Epochs: 10\n')
        rows = self.catalog.experiments()
        self.assertEqual(len(rows), 1)
        detail = self.catalog.experiment(rows[0]['id'])
        self.assertEqual(detail['metrics'][0]['source'], 'output/train_metrics.csv')
        self.assertEqual(detail['config']['config/train_config.yaml']['Epochs'], 10)

    def test_failed_job_before_first_event_stays_one_parent_experiment(self):
        run = self.root / 'results' / 'job_failed'
        (run / 'config').mkdir(parents=True)
        (run / 'config/settings.json').write_text('{}')
        (run / 'config/train_config.yaml').write_text('Epochs: 10\n')
        (run / 'exit.json').write_text('{"return_code": 1}')
        rows = self.catalog.experiments()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], 'job_failed')
        self.assertEqual(rows[0]['job_id'], 'failed')
        self.assertNotEqual(rows[0]['status'], 'Completed')
        self.assertEqual(rows[0]['latest_metrics'], {})


if __name__ == '__main__':
    unittest.main()
