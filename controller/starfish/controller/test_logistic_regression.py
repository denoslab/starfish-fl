import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from starfish.controller.tasks.logistic_regression import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_binary_data(n=200, features=3, seed=42):
    """Return (X, y) for a binary classification task with clear signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, features))
    logits = X @ np.array([2.0, -1.5, 1.0])
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LogisticRegressionTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_lr(self, **kwargs):
        return LogisticRegression(make_run(**kwargs))

    def _setup_trained_lr(self, lr, n=200, features=3, seed=42):
        X, y = make_binary_data(n=n, features=features, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        lr.X_train_scaled = scaler.fit_transform(X_train)
        lr.X_test_scaled = scaler.transform(X_test)
        lr.y_train = y_train
        lr.y_test = y_test
        lr.sample_size = n
        lr.logisticRegr = sklearn.linear_model.LogisticRegression(
            penalty='l2', max_iter=1000)
        lr.logisticRegr.fit(lr.X_train_scaled, lr.y_train)


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class LogisticRegressionPrepareDataTest(LogisticRegressionTestBase):

    @patch.object(LogisticRegression, 'is_first_round', return_value=True)
    @patch.object(LogisticRegression, 'read_dataset')
    def test_returns_true_with_valid_data(self, mock_read, _):
        mock_read.return_value = make_binary_data()
        self.assertTrue(self._make_lr().prepare_data())

    @patch.object(LogisticRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_lr().prepare_data())

    @patch.object(LogisticRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_empty(self, mock_read):
        mock_read.return_value = (np.array([]), np.array([]))
        self.assertFalse(self._make_lr().prepare_data())

    @patch.object(LogisticRegression, 'is_first_round', return_value=True)
    @patch.object(LogisticRegression, 'read_dataset')
    def test_sets_correct_sample_size_and_split(self, mock_read, _):
        X, y = make_binary_data(n=200)
        mock_read.return_value = (X, y)
        lr = self._make_lr()
        lr.prepare_data()
        self.assertEqual(lr.sample_size, 200)
        self.assertEqual(lr.X_train_scaled.shape[0], 160)
        self.assertEqual(lr.X_test_scaled.shape[0], 40)

    @patch.object(LogisticRegression, 'is_first_round', return_value=True)
    @patch.object(LogisticRegression, 'read_dataset')
    def test_initialises_logistic_regression_model(self, mock_read, _):
        mock_read.return_value = make_binary_data()
        lr = self._make_lr()
        lr.prepare_data()
        self.assertIsInstance(
            lr.logisticRegr, sklearn.linear_model.LogisticRegression)
        self.assertEqual(lr.logisticRegr.max_iter, 1)
        self.assertTrue(lr.logisticRegr.warm_start)


# ---------------------------------------------------------------------------
# training / calculate_statistics
# ---------------------------------------------------------------------------

class LogisticRegressionTrainingTest(LogisticRegressionTestBase):
    def setUp(self):
        super().setUp()
        self.lr = self._make_lr()
        self._setup_trained_lr(self.lr)

    def test_training_returns_true(self):
        self.lr.logisticRegr = sklearn.linear_model.LogisticRegression(
            penalty='l2', max_iter=1000)
        self.assertTrue(self.lr.training())

    def test_training_produces_fitted_model(self):
        self.lr.logisticRegr = sklearn.linear_model.LogisticRegression(
            penalty='l2', max_iter=1000)
        self.lr.training()
        self.assertEqual(self.lr.logisticRegr.coef_.shape[1], 3)

    def test_calculate_statistics_contains_required_keys(self):
        stats = self.lr.calculate_statistics()
        expected = {
            'sample_size', 'coef_', 'intercept_',
            'metric_acc', 'metric_auc',
            'metric_sensitivity', 'metric_specificity',
            'metric_npv', 'metric_ppv',
        }
        self.assertTrue(expected.issubset(stats.keys()))

    def test_calculate_statistics_metrics_in_valid_range(self):
        stats = self.lr.calculate_statistics()
        for key in ('metric_acc', 'metric_auc', 'metric_sensitivity',
                    'metric_specificity', 'metric_npv', 'metric_ppv'):
            self.assertGreaterEqual(stats[key], 0.0)
            self.assertLessEqual(stats[key], 1.0)

    def test_calculate_statistics_output_is_json_serialisable(self):
        stats = self.lr.calculate_statistics()
        json.dumps(stats)  # must not raise

    def test_calculate_statistics_coef_is_nested_list(self):
        """sklearn binary LogisticRegression coef_ has shape (1, n_features)."""
        stats = self.lr.calculate_statistics()
        self.assertIsInstance(stats['coef_'], list)
        self.assertIsInstance(stats['coef_'][0], list)
        self.assertEqual(len(stats['coef_'][0]), 3)


# ---------------------------------------------------------------------------
# do_aggregate  (federated weighted averaging)
# ---------------------------------------------------------------------------

class LogisticRegressionAggregationTest(LogisticRegressionTestBase):
    def setUp(self):
        super().setUp()
        self.lr = self._make_lr()
        self._setup_trained_lr(self.lr)
        self.lr.sample_size = 0

    def _write_mid_artifact(self, filename, coef, intercept, sample_size):
        """coef should be nested list e.g. [[...]], intercept a list e.g. [...]."""
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps(
            {'sample_size': sample_size, 'coef_': coef, 'intercept_': intercept}))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts_exist(self):
        self._empty_artifact_dir()
        self.assertFalse(self.lr.do_aggregate())

    @patch.object(LogisticRegression, 'upload', return_value=True)
    def test_returns_true_with_single_participant(self, _):
        actual_coef = self.lr.logisticRegr.coef_.tolist()
        actual_intercept = self.lr.logisticRegr.intercept_.tolist()
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', actual_coef, actual_intercept, 160)
        self.assertTrue(self.lr.do_aggregate())

    @patch.object(LogisticRegression, 'upload', return_value=True)
    def test_weighted_average_of_coefficients_two_sites(self, _):
        """
        Federated averaging: coef = sum(n_i * coef_i) / sum(n_i)

        Site A: n=60, coef=[[1.0, 2.0, 3.0]], intercept=[0.5]
        Site B: n=40, coef=[[3.0, 0.0, 1.0]], intercept=[1.5]

        Expected coef      = [[(60+120)/100, (120+0)/100, (180+40)/100]]
                           = [[1.8, 1.2, 2.2]]
        Expected intercept = [(30+60)/100] = [0.9]
        """
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts', [[1.0, 2.0, 3.0]], [0.5], 60)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts', [[3.0, 0.0, 1.0]], [1.5], 40)
        self.lr.do_aggregate()
        np.testing.assert_allclose(
            self.lr.logisticRegr.coef_, [[1.8, 1.2, 2.2]], atol=1e-9)
        np.testing.assert_allclose(
            self.lr.logisticRegr.intercept_, [0.9], atol=1e-9)

    @patch.object(LogisticRegression, 'upload', return_value=True)
    def test_aggregated_sample_size_equals_sum(self, _):
        actual_coef = self.lr.logisticRegr.coef_.tolist()
        actual_intercept = self.lr.logisticRegr.intercept_.tolist()
        self._write_mid_artifact('s1-1-1-mid-artifacts', actual_coef, actual_intercept, 70)
        self._write_mid_artifact('s2-1-1-mid-artifacts', actual_coef, actual_intercept, 30)
        self.lr.do_aggregate()
        self.assertEqual(self.lr.sample_size, 100)

    @patch.object(LogisticRegression, 'upload', return_value=True)
    def test_aggregate_calls_upload_with_true(self, mock_upload):
        actual_coef = self.lr.logisticRegr.coef_.tolist()
        actual_intercept = self.lr.logisticRegr.intercept_.tolist()
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', actual_coef, actual_intercept, 160)
        self.lr.do_aggregate()
        mock_upload.assert_called_once_with(True)
