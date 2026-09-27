"""Tests for the BabelBrainFno local data source, SF-03.

A run declaring ``data_source`` starts without a dataset upload, reads only
the store named by the site's ``BABELBRAIN_FL_STORE``, and refuses config
that tries to name a path.
"""

import json
import os
import shutil
import tempfile
from unittest import TestCase, skipUnless
from unittest.mock import patch

from starfish.controller.tasks.babel_brain_fno.store import STORE_ENV
from starfish.controller.tasks.babel_brain_fno.synthetic import write_synthetic_store
from starfish.controller.tasks.babel_brain_fno.task import BabelBrainFno
from starfish.controller.tasks.data_source import validate_data_source
from starfish.controller.tasks_validator import TaskValidator

DATA_SOURCE = {'type': 'babelbrain_store', 'bucket_hz': 250000}

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def make_config(**overrides):
    config = {'total_round': 1, 'current_round': 1,
              'data_source': dict(DATA_SOURCE), 'min_samples': 10}
    config.update(overrides)
    return config


def make_run(config=None, role='participant', status='Standby'):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': status, 'cur_seq': 1,
        'tasks': [{'seq': 1, 'model': 'BabelBrainFno',
                   'config': config if config is not None else make_config()}],
    }


class TaskTestCase(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = os.path.join(self.tmp, 'controller')
        self.root = os.path.join(self.tmp, 'store')
        write_synthetic_store(self.root, n_groups=5, per_group=4, val_groups=1)
        patchers = [
            patch('starfish.controller.file.file_utils.base_folder', self.base),
            patch.dict(os.environ, {STORE_ENV: self.root}),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def logs_text(self):
        path = os.path.join(self.base, '42', '1', '1', 'logs.txt')
        with open(path) as f:
            return f.read()


class DataSourceValidationTest(TestCase):

    def test_valid_data_source(self):
        self.assertIsNone(validate_data_source(DATA_SOURCE))

    def test_path_keys_are_refused(self):
        for key in ('path', 'root', 'store_root', 'file'):
            error = validate_data_source(dict(DATA_SOURCE, **{key: '/etc'}))
            self.assertIn(key, error)

    def test_unknown_type_and_bad_bucket_are_refused(self):
        self.assertIsNotNone(validate_data_source({'type': 'nfs_mount'}))
        self.assertIsNotNone(validate_data_source('babelbrain_store'))
        self.assertIsNotNone(validate_data_source(
            dict(DATA_SOURCE, bucket_hz=300000)))
        self.assertIsNotNone(validate_data_source(
            {'type': 'babelbrain_store'}))

    def _validate(self, config):
        tasks = [{'seq': 1, 'model': 'BabelBrainFno', 'config': config}]
        validator = TaskValidator(json.dumps(tasks))
        return validator.get_validated_tasks(), validator.get_error_msg()

    def test_task_validator_accepts_babelbrain_task(self):
        tasks, error = self._validate(make_config())
        self.assertIsNone(error)
        self.assertEqual(tasks[0]['model'], 'BabelBrainFno')

    def test_task_validator_refuses_a_path(self):
        config = make_config(data_source=dict(DATA_SOURCE, path='/home/other'))
        tasks, error = self._validate(config)
        self.assertIsNone(tasks)
        self.assertIn('path', error)


@patch.object(BabelBrainFno, 'notify')
class StandbyTest(TaskTestCase):
    """The first round moves to Preparing, status 3, with no dataset upload."""

    def test_local_data_source_starts_without_upload(self, notify):
        run = make_run()
        BabelBrainFno(run).standby(run)
        notify.assert_called_once_with(3)
        self.assertFalse(os.path.exists(
            os.path.join(self.base, '42', 'dataset')))

    def test_bad_data_source_fails_the_run(self, notify):
        run = make_run(make_config(data_source=dict(DATA_SOURCE, path='/etc')))
        BabelBrainFno(run).standby(run)
        notify.assert_called_once_with(1)

    def test_task_without_data_source_still_waits_for_upload(self, notify):
        config = make_config()
        del config['data_source']
        run = make_run(config)
        BabelBrainFno(run).standby(run)
        notify.assert_not_called()

    @skipUnless(HAS_TORCH, 'preparing builds the seed model, which needs torch')
    def test_standby_then_preparing_reaches_running_for_coordinator(self, notify):
        run = make_run(role='coordinator')
        task = BabelBrainFno(run)
        task.standby(run)
        with patch.object(BabelBrainFno, 'runs_in_fails', return_value=False), \
                patch.object(BabelBrainFno, 'runs_in_same_state', return_value=True):
            task.preparing(run)
        self.assertEqual([c.args[0] for c in notify.call_args_list], [3, 4])


class PrepareDataTest(TaskTestCase):

    def test_reads_the_store_from_the_environment(self):
        task = BabelBrainFno(make_run())
        self.assertTrue(task._prepare_samples())
        self.assertEqual(len(task.train_records), 16)
        self.assertEqual(len(task.val_records), 4)
        self.assertIn('16 train, 4 val', self.logs_text())

    def test_refuses_fewer_than_min_samples(self):
        task = BabelBrainFno(make_run(make_config(min_samples=17)))
        self.assertFalse(task._prepare_samples())
        self.assertIn('min_samples', self.logs_text())

    def test_default_min_samples_is_20(self):
        config = make_config()
        del config['min_samples']
        self.assertFalse(BabelBrainFno(make_run(config))._prepare_samples())

    def test_missing_environment_variable_fails(self):
        with patch.dict(os.environ, {}, clear=False):
            del os.environ[STORE_ENV]
            self.assertFalse(BabelBrainFno(make_run())._prepare_samples())

    def test_config_path_is_never_read(self):
        other = os.path.join(self.tmp, 'other-store')
        write_synthetic_store(other, n_groups=10,
                              per_group=4, val_groups=0, seed=3)
        config = make_config(data_source=dict(DATA_SOURCE, path=other))
        task = BabelBrainFno(make_run(config))
        with patch.object(BabelBrainFno, 'open_store') as open_store:
            self.assertFalse(task._prepare_samples())
            open_store.assert_not_called()

    def test_uploaded_logs_hold_no_local_paths(self):
        bucket_dir = os.path.join(self.root, 'v1', '250000')
        victim = sorted(os.listdir(bucket_dir))[0]
        with open(os.path.join(bucket_dir, victim), 'ab') as f:
            f.write(b'tamper')
        task = BabelBrainFno(make_run())
        self.assertTrue(task._prepare_samples())
        text = self.logs_text()
        self.assertIn('sha256_mismatch', text)
        self.assertNotIn(self.tmp, text)
        self.assertNotIn(os.path.realpath(self.tmp), text)

    def test_hash_cache_lives_in_the_controller_folder(self):
        BabelBrainFno(make_run())._prepare_samples()
        self.assertTrue(os.path.isfile(
            os.path.join(self.base, 'babelbrain_fl', 'sha256_cache.json')))
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, 'v1'))),
                         ['.salt', '250000', 'manifest.jsonl'])
