import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

from starfish.controller.tasks.censored_regression.task import CensoredRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_censored_data(n=200, seed=42):
    """Return (X, y) for censored regression data.

    X has [feature1, feature2, outcome], y has censoring indicator.
    True model: outcome = 2 + 0.5*x1 - 0.3*x2 + noise, right-censored at threshold.
    """
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, 2))
    noise = rng.normal(0, 1, n)
    latent_y = 2.0 + 0.5 * features[:, 0] - 0.3 * features[:, 1] + noise

    # Right-censor at the 70th percentile
    threshold = np.percentile(latent_y, 70)
    observed_y = np.where(latent_y <= threshold, latent_y, threshold)
    censor = np.where(latent_y <= threshold, 0.0, 1.0)

    X = np.column_stack([features, observed_y])
    return X, censor


def write_censored_csv(path, n=200, seed=42):
    X, y = make_censored_data(n, seed)
    data = np.column_stack([X, y])
    np.savetxt(path, data, delimiter=',', fmt='%.6f')


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CensoredRegressionTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return CensoredRegression(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, seed=42):
        import os
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        write_censored_csv(csv_path, n=n, seed=seed)
        return csv_path


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class CensoredRegressionPrepareDataTest(CensoredRegressionTestBase):

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    @patch.object(CensoredRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_task().prepare_data())

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
    def test_sets_correct_sample_size(self, _):
        self._setup_dataset(n=200)
        task = self._make_task()
        task.prepare_data()
        self.assertEqual(task.sample_size, 200)

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
    def test_rejects_invalid_censoring_values(self, _):
        """Censoring values outside {-1, 0, 1} should cause prepare_data to fail."""
        import os
        dataset_dir = os.path.join(self.tmp_dir, '42')
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 2))
        outcome = rng.normal(0, 1, 50)
        censor = np.full(50, 2.0)  # invalid
        data = np.column_stack([features, outcome, censor])
        np.savetxt(csv_path, data, delimiter=',', fmt='%.6f')

        task = self._make_task()
        self.assertFalse(task.prepare_data())


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class CensoredRegressionTrainingTest(CensoredRegressionTestBase):

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
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
            'sample_size', 'coef', 'se', 'sigma',
            'p_values', 'ci_lower', 'ci_upper', 'log_likelihood',
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
    def test_sigma_is_positive(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertGreater(result['sigma'], 0)

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
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

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
    def test_diagnostics_present(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertIn('diagnostics', result)
        diag = result['diagnostics']
        self.assertIn('censoring_summary', diag)
        self.assertIn('residual_summary', diag)
        self.assertIn('aic', diag)
        self.assertIn('bic', diag)

    @patch.object(CensoredRegression, 'is_first_round', return_value=True)
    def test_feature_names_include_intercept(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        import os
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertEqual(result['feature_names'][0], 'intercept')
        # intercept + 2 features = 3
        self.assertEqual(len(result['feature_names']), 3)


# ---------------------------------------------------------------------------
# do_aggregate
# ---------------------------------------------------------------------------

class CensoredRegressionAggregationTest(CensoredRegressionTestBase):

    def _write_mid_artifact(self, filename, coef, se, sigma, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'coef': coef,
            'se': se,
            'sigma': sigma,
            'p_values': [0.05] * len(coef),
            'ci_lower': [c - 1.96 * s for c, s in zip(coef, se)],
            'ci_upper': [c + 1.96 * s for c, s in zip(coef, se)],
            'log_likelihood': -100.0,
            'feature_names': ['intercept', 'x0', 'x1'],
        }))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts(self):
        self._empty_artifact_dir()
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(CensoredRegression, 'upload', return_value=True)
    def test_returns_true_with_single_site(self, _):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts',
            [2.0, 0.5, -0.3], [0.1, 0.1, 0.1], 1.0, 100)
        task = self._make_task()
        self.assertTrue(task.do_aggregate())

    @patch.object(CensoredRegression, 'upload', return_value=True)
    def test_inverse_variance_weighted_average(self, _):
        """
        Site A: coef=[2.0, 0.6, -0.2], se=[0.1, 0.1, 0.1], sigma=1.0
        Site B: coef=[2.0, 0.4, -0.4], se=[0.1, 0.1, 0.1], sigma=1.0
        Equal weights → pooled = [2.0, 0.5, -0.3]
        """
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts',
            [2.0, 0.6, -0.2], [0.1, 0.1, 0.1], 1.0, 100)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts',
            [2.0, 0.4, -0.4], [0.1, 0.1, 0.1], 1.0, 100)
        task = self._make_task()
        task.do_aggregate()
        import os
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        np.testing.assert_allclose(result['coef'], [2.0, 0.5, -0.3], atol=1e-6)

    @patch.object(CensoredRegression, 'upload', return_value=True)
    def test_aggregate_pools_sigma(self, _):
        """Sigma should be pooled by sample-size weighting."""
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts',
            [2.0, 0.5, -0.3], [0.1, 0.1, 0.1], 0.8, 100)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts',
            [2.0, 0.5, -0.3], [0.1, 0.1, 0.1], 1.2, 100)
        task = self._make_task()
        task.do_aggregate()
        import os
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        self.assertAlmostEqual(result['sigma'], 1.0, places=5)

    @patch.object(CensoredRegression, 'upload', return_value=True)
    def test_aggregate_calls_upload(self, mock_upload):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts',
            [2.0, 0.5, -0.3], [0.1, 0.1, 0.1], 1.0, 100)
        task = self._make_task()
        task.do_aggregate()
        mock_upload.assert_called_once_with(True)
