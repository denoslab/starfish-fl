import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

from starfish.controller.tasks.kaplan_meier.task import KaplanMeier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_survival_data(n=200, seed=42):
    """Return (X, y): X has [group, feature1, feature2, time], y has event."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, 2))
    group = (features[:, 0] > 0).astype(float)
    log_hr = features @ np.array([0.5, -0.3])
    scale = 1.0 / (0.1 * np.exp(log_hr))
    time = rng.exponential(scale)
    censor_time = rng.uniform(0, np.percentile(time, 80), n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(float)
    X = np.column_stack([group, features, observed_time])
    return X, event


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class KMTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return KaplanMeier(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, seed=42):
        import os
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        X, y = make_survival_data(n, seed)
        data = np.column_stack([X, y])
        np.savetxt(csv_path, data, delimiter=',', fmt='%.6f')
        return csv_path


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class KMPrepareDataTest(KMTestBase):

    @patch.object(KaplanMeier, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    @patch.object(KaplanMeier, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_task().prepare_data())

    @patch.object(KaplanMeier, 'is_first_round', return_value=True)
    def test_sets_correct_sample_size(self, _):
        self._setup_dataset(n=200)
        task = self._make_task()
        task.prepare_data()
        self.assertEqual(task.sample_size, 200)

    @patch.object(KaplanMeier, 'is_first_round', return_value=True)
    def test_identifies_two_groups(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertEqual(len(np.unique(task.group)), 2)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class KMTrainingTest(KMTestBase):

    @patch.object(KaplanMeier, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(KaplanMeier, 'is_first_round', return_value=True)
    def test_training_produces_km_results(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        self.assertTrue(os.path.exists(mid_artifact_path))
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertIn('km_results', result)
        self.assertIn('logrank', result)
        self.assertIn('at_risk_table', result)

    @patch.object(KaplanMeier, 'is_first_round', return_value=True)
    def test_survival_probabilities_decrease_monotonically(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        for group_name, km in result['km_results'].items():
            probs = km['survival_probability']
            for i in range(1, len(probs)):
                self.assertLessEqual(probs[i], probs[i - 1] + 1e-10)

    @patch.object(KaplanMeier, 'is_first_round', return_value=True)
    def test_logrank_test_present_for_two_groups(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertIsNotNone(result['logrank'])
        self.assertIn('test_statistic', result['logrank'])
        self.assertIn('p_value', result['logrank'])
        self.assertGreaterEqual(result['logrank']['p_value'], 0.0)
        self.assertLessEqual(result['logrank']['p_value'], 1.0)


# ---------------------------------------------------------------------------
# do_aggregate
# ---------------------------------------------------------------------------

class KMAggregationTest(KMTestBase):

    def _write_mid_artifact(self, filename, group_data, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'km_results': group_data,
            'logrank': None,
            'at_risk_table': {
                'group_0': {
                    'times': [1.0, 2.0, 3.0],
                    'events': [2, 1, 1],
                    'at_risk': [50, 40, 30],
                },
                'group_1': {
                    'times': [1.0, 2.0, 4.0],
                    'events': [1, 2, 1],
                    'at_risk': [50, 45, 35],
                },
            },
        }))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts(self):
        self._empty_artifact_dir()
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(KaplanMeier, 'upload', return_value=True)
    def test_returns_true_with_single_site(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', {}, 100)
        task = self._make_task()
        self.assertTrue(task.do_aggregate())

    @patch.object(KaplanMeier, 'upload', return_value=True)
    def test_pooled_km_has_correct_sample_size(self, _):
        self._write_mid_artifact('siteA-1-1-mid-artifacts', {}, 60)
        self._write_mid_artifact('siteB-1-1-mid-artifacts', {}, 40)
        task = self._make_task()
        task.do_aggregate()
        import os
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        self.assertEqual(result['sample_size'], 100)

    @patch.object(KaplanMeier, 'upload', return_value=True)
    def test_pooled_survival_probabilities_decrease(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', {}, 100)
        task = self._make_task()
        task.do_aggregate()
        import os
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        for group_name, km in result['km_results'].items():
            probs = km['survival_probability']
            for i in range(1, len(probs)):
                self.assertLessEqual(probs[i], probs[i - 1] + 1e-10)
