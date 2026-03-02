import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split

from starfish.controller.tasks.ordinal_logistic_regression import (
    OrdinalLogisticRegression,
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


def make_ordinal_data(n=200, n_categories=4, n_features=2, seed=42):
    """Return (X, y_ordinal) with y in {0, 1, ..., n_categories-1}."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    logits = X @ np.array([1.5, -0.8])
    # Use quantile-based thresholds so all categories are populated
    thresholds = np.quantile(logits, np.linspace(0, 1, n_categories + 1)[1:-1])
    y = np.digitize(logits, thresholds).astype(int)  # 0, 1, ..., n_categories-1
    return X, y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class OrdinalTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_olr(self, **kwargs):
        return OrdinalLogisticRegression(make_run(**kwargs))

    def _setup_trained_olr(self, olr, n=200, seed=42):
        X, y = make_ordinal_data(n=n, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        olr.X = X_train
        olr.y = y_train
        olr.X_test = X_test
        olr.y_test = y_test
        olr.sample_size = n
        olr.n_categories = 4
        olr.category_counts = {cat: int(np.sum(y == cat)) for cat in range(4)}
        model = OrderedModel(y_train, X_train, distr='logit')
        olr.model_result = model.fit(method='bfgs', disp=0)


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class OrdinalPrepareDataTest(OrdinalTestBase):

    @patch.object(OrdinalLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(OrdinalLogisticRegression, 'read_dataset')
    def test_returns_true_with_valid_ordinal_data(self, mock_read, _):
        mock_read.return_value = make_ordinal_data()
        self.assertTrue(self._make_olr().prepare_data())

    @patch.object(OrdinalLogisticRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_olr().prepare_data())

    @patch.object(OrdinalLogisticRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_empty(self, mock_read):
        mock_read.return_value = (np.array([]), np.array([]))
        self.assertFalse(self._make_olr().prepare_data())

    @patch.object(OrdinalLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(OrdinalLogisticRegression, 'read_dataset')
    def test_returns_false_with_fewer_than_3_categories(self, mock_read, _):
        """Binary target (2 categories) must be rejected."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 2))
        y = rng.integers(0, 2, 100)  # only 2 categories
        mock_read.return_value = (X, y)
        self.assertFalse(self._make_olr().prepare_data())

    @patch.object(OrdinalLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(OrdinalLogisticRegression, 'read_dataset')
    def test_remaps_non_consecutive_categories(self, mock_read, _):
        """Non-consecutive integer categories (e.g. 1, 3, 5) should be accepted and remapped."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 2))
        y = np.tile([1, 3, 5], 40)  # 3 non-consecutive categories
        mock_read.return_value = (X, y)
        olr = self._make_olr()
        self.assertTrue(olr.prepare_data())
        self.assertEqual(olr.n_categories, 3)

    @patch.object(OrdinalLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(OrdinalLogisticRegression, 'read_dataset')
    def test_sets_correct_sample_size(self, mock_read, _):
        X, y = make_ordinal_data(n=200)
        mock_read.return_value = (X, y)
        olr = self._make_olr()
        olr.prepare_data()
        self.assertEqual(olr.sample_size, 200)


# ---------------------------------------------------------------------------
# training / calculate_statistics
# ---------------------------------------------------------------------------

class OrdinalTrainingTest(OrdinalTestBase):
    def setUp(self):
        super().setUp()
        self.olr = self._make_olr()
        self._setup_trained_olr(self.olr)

    def test_training_returns_true(self):
        fresh = self._make_olr()
        X, y = make_ordinal_data()
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        fresh.X = X_train
        fresh.y = y_train
        fresh.n_categories = 4
        fresh.category_counts = {cat: int(np.sum(y == cat)) for cat in range(4)}
        fresh.sample_size = 200
        self.assertTrue(fresh.training())

    def test_calculate_statistics_contains_required_keys(self):
        stats = self.olr.calculate_statistics()
        expected = {
            'sample_size', 'n_categories', 'coef_', 'std_err', 'z_values',
            'p_values', 'conf_int_lower', 'conf_int_upper', 'odds_ratios',
            'thresholds', 'prsquared', 'llf', 'llnull', 'llr', 'aic', 'bic',
        }
        self.assertTrue(expected.issubset(stats.keys()))

    def test_thresholds_count_equals_n_categories_minus_1(self):
        stats = self.olr.calculate_statistics()
        self.assertEqual(len(stats['thresholds']), self.olr.n_categories - 1)

    def test_odds_ratios_equal_exp_of_coef(self):
        stats = self.olr.calculate_statistics()
        expected_OR = np.exp(stats['coef_']).tolist()
        np.testing.assert_allclose(stats['odds_ratios'], expected_OR, rtol=1e-6)

    def test_prsquared_between_0_and_1(self):
        stats = self.olr.calculate_statistics()
        self.assertGreaterEqual(stats['prsquared'], 0.0)
        self.assertLessEqual(stats['prsquared'], 1.0)

    def test_output_is_json_serialisable(self):
        stats = self.olr.calculate_statistics()
        json.dumps(stats)  # must not raise

    def test_llr_equals_minus2_times_llnull_minus_llf(self):
        stats = self.olr.calculate_statistics()
        expected_llr = -2 * (stats['llnull'] - stats['llf'])
        self.assertAlmostEqual(stats['llr'], expected_llr, places=9)


# ---------------------------------------------------------------------------
# do_aggregate  (pass-through / centralized)
# ---------------------------------------------------------------------------

class OrdinalAggregationTest(OrdinalTestBase):
    def setUp(self):
        super().setUp()
        self.olr = self._make_olr()
        self._setup_trained_olr(self.olr)

    def _write_mid_artifact(self, filename, payload):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps(payload))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def _make_artifact(self):
        return {
            'sample_size': 160,
            'n_categories': 4,
            'category_counts': {0: 40, 1: 40, 2: 40, 3: 40},
            'coef_': [0.5, -0.3],
            'std_err': [0.2, 0.1],
            'z_values': [2.5, -3.0],
            'p_values': [0.012, 0.003],
            'conf_int_lower': [0.1, -0.5],
            'conf_int_upper': [0.9, -0.1],
            'odds_ratios': [1.65, 0.74],
            'odds_ratio_ci_lower': [1.1, 0.6],
            'odds_ratio_ci_upper': [2.46, 0.9],
            'thresholds': [-1.0, 0.0, 1.0],
            'threshold_std_err': [0.1, 0.1, 0.1],
            'threshold_z_values': [-10.0, 0.0, 10.0],
            'threshold_p_values': [0.0, 1.0, 0.0],
            'threshold_conf_int_lower': [-1.2, -0.2, 0.8],
            'threshold_conf_int_upper': [-0.8, 0.2, 1.2],
            'prsquared': 0.15,
            'llf': -200.0,
            'llnull': -240.0,
            'llr': 80.0,
            'llr_df': 2,
            'llr_pvalue': 0.001,
            'aic': 406.0,
            'bic': 425.0,
        }

    def test_returns_false_when_no_artifacts(self):
        self._empty_artifact_dir()
        self.assertFalse(self.olr.do_aggregate())

    @patch.object(OrdinalLogisticRegression, 'upload', return_value=True)
    def test_returns_true_with_single_site(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', self._make_artifact())
        self.assertTrue(self.olr.do_aggregate())

    @patch.object(OrdinalLogisticRegression, 'upload', return_value=True)
    def test_saved_artifact_has_analysis_type_centralized(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', self._make_artifact())
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.olr.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved['analysis_type'], 'centralized')

    @patch.object(OrdinalLogisticRegression, 'upload', return_value=True)
    def test_saved_artifact_has_n_sites_1(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', self._make_artifact())
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.olr.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved['n_sites'], 1)

    @patch.object(OrdinalLogisticRegression, 'upload', return_value=True)
    def test_saved_artifact_preserves_site_stats(self, _):
        artifact = self._make_artifact()
        self._write_mid_artifact('site1-1-1-mid-artifacts', artifact)
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.olr.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved['coef_'], artifact['coef_'])
        self.assertEqual(saved['n_categories'], 4)
        self.assertEqual(saved['thresholds'], artifact['thresholds'])
