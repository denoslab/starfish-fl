import json
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

from starfish.controller.tasks.poisson_regression.task import PoissonRegression


def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=1):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_count_data(n=200, seed=42):
    """Return (X, y) for count data: X has [feature1, feature2, offset], y has counts."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, 2))
    offset = rng.uniform(0, 2, n)  # log-exposure
    log_mu = 0.5 + features @ np.array([0.3, -0.2]) + offset
    y = rng.poisson(np.exp(log_mu)).astype(float)
    X = np.column_stack([features, offset])
    return X, y


def write_count_csv(path, n=200, seed=42):
    X, y = make_count_data(n, seed)
    data = np.column_stack([X, y])
    np.savetxt(path, data, delimiter=',', fmt='%.6f')


class PoissonTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return PoissonRegression(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, seed=42):
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        write_count_csv(csv_path, n=n, seed=seed)
        return csv_path


class PoissonPrepareDataTest(PoissonTestBase):

    @patch.object(PoissonRegression, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    @patch.object(PoissonRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_task().prepare_data())

    @patch.object(PoissonRegression, 'is_first_round', return_value=True)
    def test_rejects_negative_counts(self, _):
        dataset_dir = os.path.join(self.tmp_dir, '42')
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        X, y = make_count_data()
        y[0] = -1
        np.savetxt(csv_path, np.column_stack([X, y]), delimiter=',', fmt='%.6f')
        task = self._make_task()
        self.assertFalse(task.prepare_data())


class PoissonTrainingTest(PoissonTestBase):

    @patch.object(PoissonRegression, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(PoissonRegression, 'is_first_round', return_value=True)
    def test_training_produces_expected_keys(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        expected_keys = {
            'sample_size', 'coef', 'se', 'rate_ratios',
            'p_values', 'deviance', 'aic',
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(PoissonRegression, 'is_first_round', return_value=True)
    def test_rate_ratios_are_positive(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        for rr in result['rate_ratios']:
            self.assertGreater(rr, 0)

    @patch.object(PoissonRegression, 'is_first_round', return_value=True)
    def test_deviance_is_positive(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertGreater(result['deviance'], 0)


class PoissonAggregationTest(PoissonTestBase):

    def _write_mid_artifact(self, filename, coef, se, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        rr = [float(np.exp(c)) for c in coef]
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'coef': coef,
            'se': se,
            'z_values': [c / s for c, s in zip(coef, se)],
            'p_values': [0.05] * len(coef),
            'ci_lower': [c - 1.96 * s for c, s in zip(coef, se)],
            'ci_upper': [c + 1.96 * s for c, s in zip(coef, se)],
            'rate_ratios': rr,
            'deviance': 150.0,
            'pearson_chi2': 160.0,
            'aic': 300.0,
            'feature_names': ['const', 'x1', 'x2'],
        }))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts(self):
        self._empty_artifact_dir()
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(PoissonRegression, 'upload', return_value=True)
    def test_inverse_variance_weighted_average(self, _):
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts', [0.6, 0.4, -0.2], [0.1, 0.1, 0.1], 100)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts', [0.4, 0.2, -0.4], [0.1, 0.1, 0.1], 100)
        task = self._make_task()
        task.do_aggregate()
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        np.testing.assert_allclose(result['coef'], [0.5, 0.3, -0.3], atol=1e-6)
        self.assertEqual(result['sample_size'], 200)
