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

from starfish.controller.tasks.linear_regression import LinearRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    """Return a minimal run dict compatible with AbstractTask.__init__."""
    return {
        'id': 42,
        'project': 7,
        'batch': 1,
        'role': role,
        'status': 'standby',
        'cur_seq': cur_seq,
        'tasks': [
            {
                'config': {
                    'current_round': current_round,
                    'total_round': total_round,
                }
            }
        ],
    }


def make_numeric_data(n=100, features=3, seed=42):
    """Return (X, y) with a near-perfect linear relationship."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, features))
    coef = np.array([1.5, -2.0, 0.5])
    y = X @ coef + rng.standard_normal(n) * 0.05
    return X, y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LinearRegressionTestBase(TestCase):
    """
    Redirects all file I/O to a temporary directory by patching
    file_utils.base_folder, so tests do not depend on
    /starfish-controller/local being present.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder',
            self.tmp_dir,
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_lr(self, **kwargs):
        return LinearRegression(make_run(**kwargs))

    def _setup_trained_lr(self, lr, n=100, features=3, seed=42):
        """Populate lr with split/scaled data and a fitted sklearn model."""
        X, y = make_numeric_data(n=n, features=features, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        lr.X_train_scaled = scaler.fit_transform(X_train)
        lr.X_test_scaled = scaler.transform(X_test)
        lr.y_train = y_train
        lr.y_test = y_test
        lr.sample_size = n
        lr.linearRegr = sklearn.linear_model.LinearRegression()
        lr.linearRegr.fit(lr.X_train_scaled, lr.y_train)
        return scaler


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class LinearRegressionPrepareDataTest(LinearRegressionTestBase):
    """Tests for LinearRegression.prepare_data()"""

    @patch.object(LinearRegression, 'is_first_round', return_value=True)
    @patch.object(LinearRegression, 'read_dataset')
    def test_returns_true_with_valid_numeric_data(self, mock_read, _):
        mock_read.return_value = make_numeric_data()
        lr = self._make_lr()
        self.assertTrue(lr.prepare_data())

    @patch.object(LinearRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        lr = self._make_lr()
        self.assertFalse(lr.prepare_data())

    @patch.object(LinearRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_empty(self, mock_read):
        mock_read.return_value = (np.array([]), np.array([]))
        lr = self._make_lr()
        self.assertFalse(lr.prepare_data())

    @patch.object(LinearRegression, 'is_first_round', return_value=True)
    @patch.object(LinearRegression, 'read_dataset')
    def test_sets_correct_sample_size_and_split_ratio(self, mock_read, _):
        X, y = make_numeric_data(n=100)
        mock_read.return_value = (X, y)
        lr = self._make_lr()
        lr.prepare_data()
        self.assertEqual(lr.sample_size, 100)
        self.assertEqual(lr.X_train_scaled.shape[0], 80)  # 80 % train
        self.assertEqual(lr.X_test_scaled.shape[0], 20)   # 20 % test

    @patch.object(LinearRegression, 'is_first_round', return_value=True)
    @patch.object(LinearRegression, 'read_dataset')
    def test_drops_categorical_columns(self, mock_read, _):
        """Columns that cannot be cast to float must be silently dropped."""
        n = 60
        rng = np.random.default_rng(0)
        X_num = rng.standard_normal((n, 2))
        X_cat = np.array(
            ['cat' if i % 2 == 0 else 'dog' for i in range(n)]
        ).reshape(-1, 1)
        X_mixed = np.hstack([X_num.astype(object), X_cat])
        y = rng.standard_normal(n)
        mock_read.return_value = (X_mixed, y)
        lr = self._make_lr()
        self.assertTrue(lr.prepare_data())
        self.assertEqual(lr.X_train_scaled.shape[1], 2)

    @patch.object(LinearRegression, 'is_first_round', return_value=True)
    @patch.object(LinearRegression, 'read_dataset')
    def test_returns_false_when_target_is_non_numeric(self, mock_read, _):
        X, _ = make_numeric_data(n=40)
        y = np.array(['invalid'] * 40)
        mock_read.return_value = (X, y)
        lr = self._make_lr()
        self.assertFalse(lr.prepare_data())

    @patch.object(LinearRegression, 'is_first_round', return_value=True)
    @patch.object(LinearRegression, 'read_dataset')
    def test_initialises_sklearn_linear_regression_model(self, mock_read, _):
        mock_read.return_value = make_numeric_data()
        lr = self._make_lr()
        lr.prepare_data()
        self.assertIsInstance(lr.linearRegr, sklearn.linear_model.LinearRegression)


# ---------------------------------------------------------------------------
# training / calculate_statistics
# ---------------------------------------------------------------------------

class LinearRegressionTrainingTest(LinearRegressionTestBase):
    """Tests for LinearRegression.training() and calculate_statistics()"""

    def setUp(self):
        super().setUp()
        self.lr = self._make_lr()
        self._setup_trained_lr(self.lr)

    def test_training_returns_true(self):
        # Reset to unfitted model so training() performs the fit
        self.lr.linearRegr = sklearn.linear_model.LinearRegression()
        self.assertTrue(self.lr.training())

    def test_training_produces_fitted_coefficients(self):
        self.lr.linearRegr = sklearn.linear_model.LinearRegression()
        self.lr.training()
        self.assertEqual(len(self.lr.linearRegr.coef_), 3)

    def test_calculate_statistics_contains_required_keys(self):
        stats = self.lr.calculate_statistics()
        expected_keys = {
            'sample_size', 'coef_', 'intercept_',
            'metric_mse', 'metric_rmse', 'metric_mae', 'metric_r2',
        }
        self.assertTrue(expected_keys.issubset(stats.keys()))

    def test_calculate_statistics_coef_is_json_serialisable_list(self):
        stats = self.lr.calculate_statistics()
        self.assertIsInstance(stats['coef_'], list)
        self.assertIsInstance(stats['intercept_'], float)
        # Should not raise
        json.dumps(stats)

    def test_calculate_statistics_r2_high_for_near_linear_data(self):
        """Model trained on near-linear synthetic data should achieve R² > 0.99."""
        stats = self.lr.calculate_statistics()
        self.assertGreater(stats['metric_r2'], 0.99)

    def test_calculate_statistics_rmse_equals_sqrt_of_mse(self):
        stats = self.lr.calculate_statistics()
        self.assertAlmostEqual(
            stats['metric_rmse'] ** 2, stats['metric_mse'], places=10
        )

    def test_calculate_statistics_nonnegative_error_metrics(self):
        stats = self.lr.calculate_statistics()
        self.assertGreaterEqual(stats['metric_mse'], 0.0)
        self.assertGreaterEqual(stats['metric_rmse'], 0.0)
        self.assertGreaterEqual(stats['metric_mae'], 0.0)


# ---------------------------------------------------------------------------
# do_aggregate  (federated averaging)
# ---------------------------------------------------------------------------

class LinearRegressionAggregationTest(LinearRegressionTestBase):
    """Tests for LinearRegression.do_aggregate() — the federated averaging step."""

    def setUp(self):
        super().setUp()
        self.lr = self._make_lr()
        self._setup_trained_lr(self.lr)
        self.lr.sample_size = 0  # do_aggregate is responsible for setting this

    # ------------------------------------------------------------------ helpers

    def _write_mid_artifact(self, filename, coef, intercept, sample_size):
        """
        Create a mid-artifact file inside the directory that do_aggregate
        scans via gen_all_mid_artifacts_url(project_id=7, batch=1).

        The filename must match the glob pattern  *-{cur_seq}-{round}-mid-artifacts
        used inside do_aggregate (here: *-1-1-mid-artifacts).
        """
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            'sample_size': sample_size,
            'coef_': coef,
            'intercept_': intercept,
        })
        (dir_path / filename).write_text(payload)

    def _empty_artifact_dir(self):
        """Create the scan directory but leave it empty."""
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ tests

    def test_returns_false_when_no_mid_artifacts_exist(self):
        self._empty_artifact_dir()
        self.assertFalse(self.lr.do_aggregate())

    @patch.object(LinearRegression, 'upload', return_value=True)
    def test_returns_true_with_a_single_participant(self, _):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', [1.0, 2.0, 0.5], 0.5, 80
        )
        self.assertTrue(self.lr.do_aggregate())

    @patch.object(LinearRegression, 'upload', return_value=True)
    def test_weighted_average_of_coefficients_two_sites(self, _):
        """
        Federated averaging formula:  coef = sum(n_i * coef_i) / sum(n_i)

        Site A: n=60, coef=[2.0, 4.0, 0.0], intercept=1.0
        Site B: n=40, coef=[4.0, 0.0, 0.0], intercept=3.0

        Expected coef      = [(2*60 + 4*40)/100, (4*60 + 0*40)/100, 0]
                           = [2.8, 2.4, 0.0]
        Expected intercept = (1*60 + 3*40) / 100 = 1.8
        """
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts', [2.0, 4.0, 0.0], 1.0, 60
        )
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts', [4.0, 0.0, 0.0], 3.0, 40
        )
        self.lr.do_aggregate()
        np.testing.assert_allclose(
            self.lr.linearRegr.coef_, [2.8, 2.4, 0.0], atol=1e-9
        )
        self.assertAlmostEqual(float(self.lr.linearRegr.intercept_), 1.8)

    @patch.object(LinearRegression, 'upload', return_value=True)
    def test_equal_sample_sizes_produce_simple_mean(self, _):
        """With equal sample sizes federated average equals arithmetic mean."""
        self._write_mid_artifact(
            's1-1-1-mid-artifacts', [1.0, 2.0, 3.0], 1.0, 50
        )
        self._write_mid_artifact(
            's2-1-1-mid-artifacts', [3.0, 4.0, 5.0], 3.0, 50
        )
        self.lr.do_aggregate()
        np.testing.assert_allclose(
            self.lr.linearRegr.coef_, [2.0, 3.0, 4.0], atol=1e-9
        )
        self.assertAlmostEqual(float(self.lr.linearRegr.intercept_), 2.0)

    @patch.object(LinearRegression, 'upload', return_value=True)
    def test_aggregated_sample_size_equals_sum_of_participants(self, _):
        self._write_mid_artifact(
            's1-1-1-mid-artifacts', [1.0, 0.0, 0.0], 0.0, 70
        )
        self._write_mid_artifact(
            's2-1-1-mid-artifacts', [3.0, 0.0, 0.0], 0.0, 30
        )
        self.lr.do_aggregate()
        self.assertEqual(self.lr.sample_size, 100)

    @patch.object(LinearRegression, 'upload', return_value=True)
    def test_aggregate_uploads_artifact_after_success(self, mock_upload):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts', [0.5, 1.5, -0.5], 0.2, 100
        )
        self.lr.do_aggregate()
        mock_upload.assert_called_once_with(True)
