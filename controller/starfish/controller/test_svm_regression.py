import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

import sklearn.svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from starfish.controller.tasks.svm_regression.task import SvmRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_numeric_data(n=100, features=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, features))
    y = X @ np.array([1.5, -2.0, 0.5]) + rng.standard_normal(n) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SvmRegressionTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_svm(self, **kwargs):
        return SvmRegression(make_run(**kwargs))

    def _setup_trained_svm(self, svm, n=100, features=3, seed=42):
        X, y = make_numeric_data(n=n, features=features, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        svm.X_train_scaled = scaler.fit_transform(X_train)
        svm.X_test_scaled = scaler.transform(X_test)
        svm.y_train = y_train
        svm.y_test = y_test
        svm.sample_size = n
        svm.svmRegr = sklearn.svm.SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svm.svmRegr.fit(svm.X_train_scaled, svm.y_train)


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class SvmRegressionPrepareDataTest(SvmRegressionTestBase):

    @patch.object(SvmRegression, 'is_first_round', return_value=True)
    @patch.object(SvmRegression, 'read_dataset')
    def test_returns_true_with_valid_numeric_data(self, mock_read, _):
        mock_read.return_value = make_numeric_data()
        self.assertTrue(self._make_svm().prepare_data())

    @patch.object(SvmRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_svm().prepare_data())

    @patch.object(SvmRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_empty(self, mock_read):
        mock_read.return_value = (np.array([]), np.array([]))
        self.assertFalse(self._make_svm().prepare_data())

    @patch.object(SvmRegression, 'is_first_round', return_value=True)
    @patch.object(SvmRegression, 'read_dataset')
    def test_sets_correct_sample_size_and_split(self, mock_read, _):
        X, y = make_numeric_data(n=100)
        mock_read.return_value = (X, y)
        svm = self._make_svm()
        svm.prepare_data()
        self.assertEqual(svm.sample_size, 100)
        self.assertEqual(svm.X_train_scaled.shape[0], 80)
        self.assertEqual(svm.X_test_scaled.shape[0], 20)

    @patch.object(SvmRegression, 'is_first_round', return_value=True)
    @patch.object(SvmRegression, 'read_dataset')
    def test_initialises_svr_model_with_correct_params(self, mock_read, _):
        mock_read.return_value = make_numeric_data()
        svm = self._make_svm()
        svm.prepare_data()
        self.assertIsInstance(svm.svmRegr, sklearn.svm.SVR)
        self.assertEqual(svm.svmRegr.kernel, 'rbf')
        self.assertEqual(svm.svmRegr.C, 1.0)
        self.assertEqual(svm.svmRegr.epsilon, 0.1)


# ---------------------------------------------------------------------------
# training / calculate_statistics
# ---------------------------------------------------------------------------

class SvmRegressionTrainingTest(SvmRegressionTestBase):
    def setUp(self):
        super().setUp()
        self.svm = self._make_svm()
        self._setup_trained_svm(self.svm)

    def test_training_returns_true(self):
        self.svm.svmRegr = sklearn.svm.SVR(kernel='rbf', C=1.0, epsilon=0.1)
        self.assertTrue(self.svm.training())

    def test_training_produces_fitted_model(self):
        self.svm.svmRegr = sklearn.svm.SVR(kernel='rbf', C=1.0, epsilon=0.1)
        self.svm.training()
        self.assertIsNotNone(self.svm.svmRegr.support_vectors_)

    def test_calculate_statistics_contains_required_keys(self):
        stats = self.svm.calculate_statistics()
        expected = {
            'sample_size', 'dual_coef', 'intercept',
            'metric_mse', 'metric_rmse', 'metric_mae', 'metric_r2',
        }
        self.assertTrue(expected.issubset(stats.keys()))

    def test_calculate_statistics_dual_coef_is_nested_list(self):
        stats = self.svm.calculate_statistics()
        self.assertIsInstance(stats['dual_coef'], list)
        # dual_coef_ has shape (1, n_support_vectors) → nested list
        self.assertIsInstance(stats['dual_coef'][0], list)

    def test_calculate_statistics_intercept_is_float(self):
        stats = self.svm.calculate_statistics()
        self.assertIsInstance(stats['intercept'], float)

    def test_calculate_statistics_rmse_equals_sqrt_of_mse(self):
        stats = self.svm.calculate_statistics()
        self.assertAlmostEqual(
            stats['metric_rmse'] ** 2, stats['metric_mse'], places=10)

    def test_calculate_statistics_output_is_json_serialisable(self):
        stats = self.svm.calculate_statistics()
        json.dumps(stats)  # must not raise


# ---------------------------------------------------------------------------
# do_aggregate  (federated weighted averaging)
# ---------------------------------------------------------------------------

class SvmRegressionAggregationTest(SvmRegressionTestBase):
    def setUp(self):
        super().setUp()
        self.svm = self._make_svm()
        self._setup_trained_svm(self.svm)
        self.svm.sample_size = 0

    def _write_mid_artifact(self, filename, dual_coef, intercept, sample_size):
        """dual_coef should be a nested list [[...]], intercept a float."""
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps(
            {'sample_size': sample_size, 'dual_coef': dual_coef, 'intercept': intercept}))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts_exist(self):
        self._empty_artifact_dir()
        self.assertFalse(self.svm.do_aggregate())

    @patch.object(SvmRegression, 'upload', return_value=True)
    def test_returns_true_with_single_participant(self, _):
        """Write the fitted model's own dual_coef so shape is preserved for predict."""
        actual_dual_coef = self.svm.svmRegr.dual_coef_.tolist()
        actual_intercept = float(self.svm.svmRegr.intercept_[0])
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', actual_dual_coef, actual_intercept, 80)
        self.assertTrue(self.svm.do_aggregate())

    @patch.object(SvmRegression, 'upload', return_value=True)
    @patch.object(SvmRegression, 'save_artifacts', return_value=True)
    @patch.object(SvmRegression, 'calculate_statistics',
                  return_value={'sample_size': 100, 'dual_coef': [[0.0]],
                                'intercept': 0.0, 'metric_mse': 0.1,
                                'metric_rmse': 0.316, 'metric_mae': 0.2, 'metric_r2': 0.9})
    def test_weighted_intercept_averaging_two_sites(self, _mock_stats, _mock_save, _mock_up):
        """
        calculate_statistics is mocked to avoid predict() with mismatched
        support_vectors (SVR dual_coef shapes differ across sites).

        Site A: n=60, intercept=1.0
        Site B: n=40, intercept=3.0
        Expected: (1.0*60 + 3.0*40) / 100 = 1.8
        """
        self._write_mid_artifact('siteA-1-1-mid-artifacts', [[0.5, -0.3]], 1.0, 60)
        self._write_mid_artifact('siteB-1-1-mid-artifacts', [[1.5, -0.9]], 3.0, 40)
        self.svm.do_aggregate()
        self.assertAlmostEqual(float(self.svm.svmRegr.intercept_[0]), 1.8)

    @patch.object(SvmRegression, 'upload', return_value=True)
    @patch.object(SvmRegression, 'save_artifacts', return_value=True)
    @patch.object(SvmRegression, 'calculate_statistics',
                  return_value={'sample_size': 100, 'dual_coef': [[0.0]],
                                'intercept': 0.0, 'metric_mse': 0.1,
                                'metric_rmse': 0.316, 'metric_mae': 0.2, 'metric_r2': 0.9})
    def test_aggregated_sample_size_equals_sum(self, _m1, _m2, _m3):
        self._write_mid_artifact('s1-1-1-mid-artifacts', [[0.5]], 1.0, 70)
        self._write_mid_artifact('s2-1-1-mid-artifacts', [[1.5]], 3.0, 30)
        self.svm.do_aggregate()
        self.assertEqual(self.svm.sample_size, 100)
