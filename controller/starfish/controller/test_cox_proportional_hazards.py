import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

from starfish.controller.tasks.cox_proportional_hazards.task import CoxProportionalHazards


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
    """Return (X, y) for survival data: X has [feature1, feature2, time], y has event."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, 2))
    log_hr = features @ np.array([0.5, -0.3])
    scale = 1.0 / (0.1 * np.exp(log_hr))
    time = rng.exponential(scale)
    censor_time = rng.uniform(0, np.percentile(time, 80), n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(float)
    X = np.column_stack([features, observed_time])
    return X, event


def write_survival_csv(path, n=200, seed=42):
    X, y = make_survival_data(n, seed)
    data = np.column_stack([X, y])
    np.savetxt(path, data, delimiter=',', fmt='%.6f')


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CoxPHTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return CoxProportionalHazards(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, seed=42):
        import os
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        write_survival_csv(csv_path, n=n, seed=seed)
        return csv_path


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class CoxPHPrepareDataTest(CoxPHTestBase):

    @patch.object(CoxProportionalHazards, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    @patch.object(CoxProportionalHazards, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_task().prepare_data())

    @patch.object(CoxProportionalHazards, 'is_first_round', return_value=True)
    def test_sets_correct_sample_size(self, _):
        self._setup_dataset(n=200)
        task = self._make_task()
        task.prepare_data()
        self.assertEqual(task.sample_size, 200)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class CoxPHTrainingTest(CoxPHTestBase):

    @patch.object(CoxProportionalHazards, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(CoxProportionalHazards, 'is_first_round', return_value=True)
    def test_training_produces_mid_artifacts(self, _):
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
        expected_keys = {
            'sample_size', 'coef', 'se', 'hazard_ratio',
            'p_values', 'ci_lower', 'ci_upper', 'concordance_index',
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(CoxProportionalHazards, 'is_first_round', return_value=True)
    def test_hazard_ratios_are_positive(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        for hr in result['hazard_ratio']:
            self.assertGreater(hr, 0)

    @patch.object(CoxProportionalHazards, 'is_first_round', return_value=True)
    def test_concordance_index_in_valid_range(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertGreaterEqual(result['concordance_index'], 0.0)
        self.assertLessEqual(result['concordance_index'], 1.0)

    @patch.object(CoxProportionalHazards, 'is_first_round', return_value=True)
    def test_p_values_in_valid_range(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        for p in result['p_values']:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


# ---------------------------------------------------------------------------
# do_aggregate
# ---------------------------------------------------------------------------

class CoxPHAggregationTest(CoxPHTestBase):

    def _write_mid_artifact(self, filename, coef, se, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        hr = np.exp(coef).tolist() if isinstance(coef, np.ndarray) else [np.exp(c) for c in coef]
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'coef': coef,
            'se': se,
            'hazard_ratio': hr,
            'p_values': [0.05, 0.01],
            'ci_lower': [c - 1.96 * s for c, s in zip(coef, se)],
            'ci_upper': [c + 1.96 * s for c, s in zip(coef, se)],
            'concordance_index': 0.65,
            'feature_names': ['x0', 'x1'],
        }))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts(self):
        self._empty_artifact_dir()
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(CoxProportionalHazards, 'upload', return_value=True)
    def test_returns_true_with_single_site(self, _):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', [0.5, -0.3], [0.1, 0.2], 100)
        task = self._make_task()
        self.assertTrue(task.do_aggregate())

    @patch.object(CoxProportionalHazards, 'upload', return_value=True)
    def test_inverse_variance_weighted_average(self, _):
        """
        Site A: coef=[0.6, -0.2], se=[0.1, 0.1], weight=[100, 100]
        Site B: coef=[0.4, -0.4], se=[0.1, 0.1], weight=[100, 100]
        Equal weights → pooled = [0.5, -0.3]
        """
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts', [0.6, -0.2], [0.1, 0.1], 100)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts', [0.4, -0.4], [0.1, 0.1], 100)
        task = self._make_task()
        task.do_aggregate()
        import os
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        np.testing.assert_allclose(result['coef'], [0.5, -0.3], atol=1e-6)

    @patch.object(CoxProportionalHazards, 'upload', return_value=True)
    def test_aggregate_calls_upload(self, mock_upload):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', [0.5, -0.3], [0.1, 0.2], 100)
        task = self._make_task()
        task.do_aggregate()
        mock_upload.assert_called_once_with(True)
