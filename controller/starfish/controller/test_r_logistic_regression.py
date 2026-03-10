import json
import os
import shutil
import tempfile

import numpy as np
from django.test import TestCase
from unittest import skipUnless
from unittest.mock import patch
from pathlib import Path

from starfish.controller.tasks.r_logistic_regression.task import RLogisticRegression

R_AVAILABLE = shutil.which('Rscript') is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_binary_csv(path, n=200, features=3, seed=42):
    """Write a headerless CSV with binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, features))
    logits = X @ np.array([2.0, -1.5, 1.0])
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float)
    data = np.column_stack([X, y])
    np.savetxt(path, data, delimiter=',', fmt='%.6f')


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@skipUnless(R_AVAILABLE, 'Rscript not found on PATH')
class RLogisticRegressionTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return RLogisticRegression(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, features=3, seed=42):
        """Create dataset file in the expected location and return its path."""
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        make_binary_csv(csv_path, n=n, features=features, seed=seed)
        return csv_path


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class RLogisticRegressionPrepareDataTest(RLogisticRegressionTestBase):

    @patch.object(RLogisticRegression, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    def test_returns_false_when_no_dataset(self):
        task = self._make_task()
        self.assertFalse(task.prepare_data())

    @patch.object(RLogisticRegression, 'is_first_round', return_value=True)
    def test_sets_correct_sample_size(self, _):
        self._setup_dataset(n=200)
        task = self._make_task()
        task.prepare_data()
        self.assertEqual(task.sample_size, 200)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class RLogisticRegressionTrainingTest(RLogisticRegressionTestBase):

    @patch.object(RLogisticRegression, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(RLogisticRegression, 'is_first_round', return_value=True)
    def test_training_produces_mid_artifacts(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        self.assertTrue(os.path.exists(mid_artifact_path))
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        expected_keys = {
            'sample_size', 'coef_', 'intercept_',
            'metric_acc', 'metric_auc',
            'metric_sensitivity', 'metric_specificity',
            'metric_npv', 'metric_ppv',
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(RLogisticRegression, 'is_first_round', return_value=True)
    def test_training_metrics_in_valid_range(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        for key in ('metric_acc', 'metric_auc', 'metric_sensitivity',
                    'metric_specificity', 'metric_npv', 'metric_ppv'):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    @patch.object(RLogisticRegression, 'is_first_round', return_value=True)
    def test_training_coef_is_nested_list(self, _):
        self._setup_dataset(features=3)
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertIsInstance(result['coef_'], list)
        self.assertIsInstance(result['coef_'][0], list)
        self.assertEqual(len(result['coef_'][0]), 3)

    @patch.object(RLogisticRegression, 'is_first_round', return_value=True)
    def test_training_output_is_json_serialisable(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# do_aggregate (federated weighted averaging)
# ---------------------------------------------------------------------------

class RLogisticRegressionAggregationTest(RLogisticRegressionTestBase):

    def _write_mid_artifact(self, filename, coef, intercept, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps(
            {'sample_size': sample_size, 'coef_': coef, 'intercept_': intercept}))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts_exist(self):
        self._empty_artifact_dir()
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(RLogisticRegression, 'upload', return_value=True)
    def test_returns_true_with_single_participant(self, _):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', [[1.0, 2.0, 3.0]], [0.5], 160)
        task = self._make_task()
        self.assertTrue(task.do_aggregate())

    @patch.object(RLogisticRegression, 'upload', return_value=True)
    def test_weighted_average_of_coefficients_two_sites(self, _):
        """
        Site A: n=60, coef=[[1.0, 2.0, 3.0]], intercept=[0.5]
        Site B: n=40, coef=[[3.0, 0.0, 1.0]], intercept=[1.5]

        Expected coef      = [[1.8, 1.2, 2.2]]
        Expected intercept = [0.9]
        """
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts', [[1.0, 2.0, 3.0]], [0.5], 60)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts', [[3.0, 0.0, 1.0]], [1.5], 40)
        task = self._make_task()
        task.do_aggregate()
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        np.testing.assert_allclose(
            result['coef_'], [[1.8, 1.2, 2.2]], atol=1e-9)
        np.testing.assert_allclose(
            result['intercept_'], [0.9], atol=1e-9)

    @patch.object(RLogisticRegression, 'upload', return_value=True)
    def test_aggregated_sample_size_equals_sum(self, _):
        self._write_mid_artifact(
            's1-1-1-mid-artifacts', [[1.0, 2.0]], [0.5], 70)
        self._write_mid_artifact(
            's2-1-1-mid-artifacts', [[1.0, 2.0]], [0.5], 30)
        task = self._make_task()
        task.do_aggregate()
        self.assertEqual(task.sample_size, 100)

    @patch.object(RLogisticRegression, 'upload', return_value=True)
    def test_aggregate_calls_upload_with_true(self, mock_upload):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', [[1.0, 2.0, 3.0]], [0.5], 160)
        task = self._make_task()
        task.do_aggregate()
        mock_upload.assert_called_once_with(True)
