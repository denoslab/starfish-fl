import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from starfish.controller.tasks.stats_models.logistic_regression_stats import (
    LogisticRegressionStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=1):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_binary_data(n=100, features=2, seed=42):
    """Return (X, y_binary) using Bernoulli sampling to avoid perfect separation."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, features))
    logits = X @ np.array([1.5, -1.0])
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LogisticRegressionStatsTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_lrs(self, **kwargs):
        return LogisticRegressionStats(make_run(**kwargs))

    def _setup_trained_lrs(self, lrs, n=100, seed=42):
        X, y = make_binary_data(n=n, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        lrs.X = X_train
        lrs.y = y_train
        lrs.X_test = X_test
        lrs.y_test = y_test
        lrs.sample_size = n
        lrs.X_with_const = sm.add_constant(X_train)
        model = sm.Logit(y_train, lrs.X_with_const)
        lrs.model_result = model.fit(disp=0)


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class LogisticRegressionStatsPrepareDataTest(LogisticRegressionStatsTestBase):

    @patch.object(LogisticRegressionStats, 'is_first_round', return_value=True)
    @patch.object(LogisticRegressionStats, 'read_dataset')
    def test_returns_true_with_valid_binary_data(self, mock_read, _):
        mock_read.return_value = make_binary_data()
        self.assertTrue(self._make_lrs().prepare_data())

    @patch.object(LogisticRegressionStats, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_lrs().prepare_data())

    @patch.object(LogisticRegressionStats, 'read_dataset')
    def test_returns_false_when_dataset_is_empty(self, mock_read):
        mock_read.return_value = (np.array([]), np.array([]))
        self.assertFalse(self._make_lrs().prepare_data())

    @patch.object(LogisticRegressionStats, 'is_first_round', return_value=True)
    @patch.object(LogisticRegressionStats, 'read_dataset')
    def test_returns_false_with_non_binary_target(self, mock_read, _):
        """Target with more than 2 unique values must be rejected."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 2))
        y = rng.integers(0, 3, 60).astype(float)  # 3 classes
        mock_read.return_value = (X, y)
        self.assertFalse(self._make_lrs().prepare_data())

    @patch.object(LogisticRegressionStats, 'is_first_round', return_value=True)
    @patch.object(LogisticRegressionStats, 'read_dataset')
    def test_adds_constant_column(self, mock_read, _):
        mock_read.return_value = make_binary_data(n=100)
        lrs = self._make_lrs()
        lrs.prepare_data()
        self.assertEqual(lrs.X_with_const.shape[1], lrs.X.shape[1] + 1)


# ---------------------------------------------------------------------------
# training / calculate_statistics
# ---------------------------------------------------------------------------

class LogisticRegressionStatsTrainingTest(LogisticRegressionStatsTestBase):
    def setUp(self):
        super().setUp()
        self.lrs = self._make_lrs()
        self._setup_trained_lrs(self.lrs)

    def test_training_returns_true(self):
        fresh = self._make_lrs()
        X, y = make_binary_data()
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        fresh.X = X_train
        fresh.y = y_train
        fresh.X_with_const = sm.add_constant(X_train)
        fresh.sample_size = 100
        self.assertTrue(fresh.training())

    def test_calculate_statistics_contains_required_keys(self):
        stats = self.lrs.calculate_statistics()
        expected = {
            'sample_size', 'coef_', 'std_err', 'z_values', 'p_values',
            'conf_int_lower', 'conf_int_upper', 'odds_ratios',
            'prsquared', 'llr', 'llr_pvalue', 'llf', 'llnull',
        }
        self.assertTrue(expected.issubset(stats.keys()))

    def test_odds_ratios_are_positive(self):
        stats = self.lrs.calculate_statistics()
        for OR in stats['odds_ratios']:
            self.assertGreater(OR, 0.0)

    def test_pseudo_r_squared_between_0_and_1(self):
        stats = self.lrs.calculate_statistics()
        self.assertGreaterEqual(stats['prsquared'], 0.0)
        self.assertLessEqual(stats['prsquared'], 1.0)

    def test_output_is_json_serialisable(self):
        stats = self.lrs.calculate_statistics()
        json.dumps(stats)  # must not raise

    def test_odds_ratios_equal_exp_of_coef(self):
        stats = self.lrs.calculate_statistics()
        expected_OR = np.exp(stats['coef_']).tolist()
        np.testing.assert_allclose(stats['odds_ratios'], expected_OR, rtol=1e-6)


# ---------------------------------------------------------------------------
# do_aggregate  (inverse-variance weighted meta-analysis)
# ---------------------------------------------------------------------------

class LogisticRegressionStatsAggregationTest(LogisticRegressionStatsTestBase):
    def setUp(self):
        super().setUp()
        self.lrs = self._make_lrs()
        self._setup_trained_lrs(self.lrs)

    def _write_mid_artifact(self, filename, payload):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps(payload))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def _make_artifact(self, coef, std_err, sample_size,
                       prsquared=0.2, llf=-30.0, llnull=-40.0):
        return {
            'sample_size': sample_size,
            'coef_': coef,
            'std_err': std_err,
            'prsquared': prsquared,
            'llf': llf,
            'llnull': llnull,
        }

    def test_returns_false_when_no_artifacts(self):
        self._empty_artifact_dir()
        self.assertFalse(self.lrs.do_aggregate())

    @patch.object(LogisticRegressionStats, 'upload', return_value=True)
    def test_returns_true_with_single_site(self, _):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts',
            self._make_artifact([0.8, -0.4, 0.2], [0.3, 0.2, 0.1], 40))
        self.assertTrue(self.lrs.do_aggregate())

    @patch.object(LogisticRegressionStats, 'upload', return_value=True)
    def test_inverse_variance_weighted_pooling(self, _):
        """
        Site A: coef=[1.0], std_err=[0.5]   → w = 4
        Site B: coef=[3.0], std_err=[1.0]   → w = 1
        total_w = 5
        pooled_coef = (1.0*4 + 3.0*1) / 5 = 7/5 = 1.4
        """
        self._write_mid_artifact('sA-1-1-mid-artifacts',
            self._make_artifact([1.0], [0.5], 50, llf=-20.0, llnull=-30.0))
        self._write_mid_artifact('sB-1-1-mid-artifacts',
            self._make_artifact([3.0], [1.0], 50, llf=-15.0, llnull=-22.0))
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.lrs.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertAlmostEqual(saved['coef_'][0], 1.4, places=9)

    @patch.object(LogisticRegressionStats, 'upload', return_value=True)
    def test_pooled_llr_equals_sum_of_site_llrs(self, _):
        """
        pooled_llr = -2*(sum(llnull) - sum(llf))
                   = -2*((-30 + -22) - (-20 + -15))
                   = -2*(-52 + 35) = -2*(-17) = 34
        Which equals sum of individual LLRs:
            site A: -2*(-30 - (-20)) = 20
            site B: -2*(-22 - (-15)) = 14  → total = 34
        """
        self._write_mid_artifact('sA-1-1-mid-artifacts',
            self._make_artifact([1.0], [0.5], 50, llf=-20.0, llnull=-30.0))
        self._write_mid_artifact('sB-1-1-mid-artifacts',
            self._make_artifact([3.0], [1.0], 50, llf=-15.0, llnull=-22.0))
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.lrs.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertAlmostEqual(saved['llr'], 34.0, places=9)

    @patch.object(LogisticRegressionStats, 'upload', return_value=True)
    def test_pooled_odds_ratios_equal_exp_of_pooled_coef(self, _):
        self._write_mid_artifact('s1-1-1-mid-artifacts',
            self._make_artifact([0.5, -0.3], [0.2, 0.1], 60))
        self._write_mid_artifact('s2-1-1-mid-artifacts',
            self._make_artifact([0.7, -0.4], [0.3, 0.15], 40))
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.lrs.do_aggregate()
        saved = json.loads(result_path.read_text())
        expected_OR = np.exp(saved['coef_']).tolist()
        np.testing.assert_allclose(saved['odds_ratios'], expected_OR, rtol=1e-6)
