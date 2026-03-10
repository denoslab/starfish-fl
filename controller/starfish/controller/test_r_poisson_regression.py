import json
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest import skipUnless
from unittest.mock import patch

from starfish.controller.tasks.r_poisson_regression.task import RPoissonRegression

R_AVAILABLE = shutil.which('Rscript') is not None


def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=1):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_count_csv(path, n=200, seed=42):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, 2))
    offset = rng.uniform(0, 2, n)
    log_mu = 0.5 + features @ np.array([0.3, -0.2]) + offset
    y = rng.poisson(np.exp(log_mu)).astype(float)
    data = np.column_stack([features, offset, y])
    np.savetxt(path, data, delimiter=',', fmt='%.6f')


@skipUnless(R_AVAILABLE, 'Rscript not found on PATH')
class RPoissonTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return RPoissonRegression(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, seed=42):
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        make_count_csv(csv_path, n=n, seed=seed)
        return csv_path


class RPoissonPrepareDataTest(RPoissonTestBase):

    @patch.object(RPoissonRegression, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    def test_returns_false_when_no_dataset(self):
        task = self._make_task()
        self.assertFalse(task.prepare_data())


class RPoissonTrainingTest(RPoissonTestBase):

    @patch.object(RPoissonRegression, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(RPoissonRegression, 'is_first_round', return_value=True)
    def test_training_produces_expected_keys(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        expected_keys = {'sample_size', 'coef', 'se', 'rate_ratios',
                         'p_values', 'deviance'}
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(RPoissonRegression, 'is_first_round', return_value=True)
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


class RPoissonAggregationTest(RPoissonTestBase):

    def _write_mid_artifact(self, filename, coef, se, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'coef': coef,
            'se': se,
            'z_values': [c / s for c, s in zip(coef, se)],
            'p_values': [0.05] * len(coef),
            'ci_lower': [c - 1.96 * s for c, s in zip(coef, se)],
            'ci_upper': [c + 1.96 * s for c, s in zip(coef, se)],
            'rate_ratios': [float(np.exp(c)) for c in coef],
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

    @patch.object(RPoissonRegression, 'upload', return_value=True)
    def test_weighted_average_with_equal_se(self, _):
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
