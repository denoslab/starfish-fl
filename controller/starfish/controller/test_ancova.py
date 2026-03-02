import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from starfish.controller.tasks.ancova import Ancova


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=1,
             n_group_columns=1):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {
            'current_round': current_round,
            'total_round': total_round,
            'n_group_columns': n_group_columns,
        }}],
    }


def make_ancova_data(n=100, seed=42):
    """
    Returns (X, y) where X[:,0] is a binary group indicator and X[:,1:] are
    continuous covariates.  y is a continuous outcome with a group effect.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n).astype(float)
    cov = rng.standard_normal((n, 2))
    X = np.column_stack([group, cov])
    y = 1.5 * group + cov @ np.array([0.5, -0.3]) + rng.standard_normal(n) * 0.3
    return X, y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AncovaTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_ancova(self, **kwargs):
        return Ancova(make_run(**kwargs))

    def _setup_trained_ancova(self, ancova, n=100, seed=42):
        X, y = make_ancova_data(n=n, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        ancova.X = X_train
        ancova.y = y_train
        ancova.X_test = X_test
        ancova.y_test = y_test
        ancova.sample_size = n
        ancova.n_group_cols = 1
        ancova.X_with_const = sm.add_constant(X_train)
        model = sm.OLS(y_train, ancova.X_with_const)
        ancova.model_result = model.fit()


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class AncovaaPrepareDataTest(AncovaTestBase):

    @patch.object(Ancova, 'is_first_round', return_value=True)
    @patch.object(Ancova, 'read_dataset')
    def test_returns_true_with_valid_data(self, mock_read, _):
        mock_read.return_value = make_ancova_data()
        self.assertTrue(self._make_ancova().prepare_data())

    @patch.object(Ancova, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_ancova().prepare_data())

    @patch.object(Ancova, 'read_dataset')
    def test_returns_false_when_dataset_is_empty(self, mock_read):
        mock_read.return_value = (np.array([]), np.array([]))
        self.assertFalse(self._make_ancova().prepare_data())

    @patch.object(Ancova, 'is_first_round', return_value=True)
    @patch.object(Ancova, 'read_dataset')
    def test_reads_n_group_columns_from_config(self, mock_read, _):
        mock_read.return_value = make_ancova_data()
        ancova = self._make_ancova(n_group_columns=2)
        ancova.prepare_data()
        self.assertEqual(ancova.n_group_cols, 2)

    @patch.object(Ancova, 'is_first_round', return_value=True)
    @patch.object(Ancova, 'read_dataset')
    def test_adds_constant_to_design_matrix(self, mock_read, _):
        mock_read.return_value = make_ancova_data(n=100)
        ancova = self._make_ancova()
        ancova.prepare_data()
        # X_with_const should have one more column than X (the constant)
        self.assertEqual(
            ancova.X_with_const.shape[1], ancova.X.shape[1] + 1)

    @patch.object(Ancova, 'is_first_round', return_value=True)
    @patch.object(Ancova, 'read_dataset')
    def test_sets_correct_sample_size(self, mock_read, _):
        mock_read.return_value = make_ancova_data(n=100)
        ancova = self._make_ancova()
        ancova.prepare_data()
        self.assertEqual(ancova.sample_size, 100)


# ---------------------------------------------------------------------------
# training / calculate_statistics
# ---------------------------------------------------------------------------

class AncovaTrainingTest(AncovaTestBase):
    def setUp(self):
        super().setUp()
        self.ancova = self._make_ancova()
        self._setup_trained_ancova(self.ancova)

    def test_training_returns_true(self):
        ancova = self._make_ancova()
        self._setup_trained_ancova(ancova)
        # Re-run training on a fresh instance
        fresh = self._make_ancova()
        X, y = make_ancova_data()
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        fresh.X = X_train
        fresh.y = y_train
        fresh.X_with_const = sm.add_constant(X_train)
        fresh.sample_size = 100
        fresh.n_group_cols = 1
        self.assertTrue(fresh.training())

    def test_calculate_statistics_contains_required_keys(self):
        stats = self.ancova.calculate_statistics()
        expected = {
            'sample_size', 'coef_', 'std_err', 't_values', 'p_values',
            'conf_int_lower', 'conf_int_upper', 'r_squared', 'adj_r_squared',
            'f_statistic', 'f_pvalue', 'ss_model', 'ss_residual', 'ss_total',
            'df_model', 'df_residual', 'partial_eta_squared', 'n_group_columns',
        }
        self.assertTrue(expected.issubset(stats.keys()))

    def test_calculate_statistics_r_squared_between_0_and_1(self):
        stats = self.ancova.calculate_statistics()
        self.assertGreaterEqual(stats['r_squared'], 0.0)
        self.assertLessEqual(stats['r_squared'], 1.0)

    def test_calculate_statistics_partial_eta_squared_between_0_and_1(self):
        stats = self.ancova.calculate_statistics()
        self.assertGreaterEqual(stats['partial_eta_squared'], 0.0)
        self.assertLessEqual(stats['partial_eta_squared'], 1.0)

    def test_calculate_statistics_output_is_json_serialisable(self):
        stats = self.ancova.calculate_statistics()
        json.dumps(stats)  # must not raise

    def test_calculate_statistics_ss_total_equals_model_plus_residual(self):
        stats = self.ancova.calculate_statistics()
        self.assertAlmostEqual(
            stats['ss_total'], stats['ss_model'] + stats['ss_residual'], places=6)


# ---------------------------------------------------------------------------
# do_aggregate  (inverse-variance weighted meta-analysis)
# ---------------------------------------------------------------------------

class AncovaAggregationTest(AncovaTestBase):
    def setUp(self):
        super().setUp()
        self.ancova = self._make_ancova()
        self._setup_trained_ancova(self.ancova)

    def _write_mid_artifact(self, filename, payload):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps(payload))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def _make_artifact(self, coef, std_err, sample_size,
                       ss_model=50.0, ss_residual=200.0,
                       df_model=1, df_residual=78, partial_eta=0.1):
        return {
            'sample_size': sample_size,
            'coef_': coef,
            'std_err': std_err,
            'ss_model': ss_model,
            'ss_residual': ss_residual,
            'df_model': df_model,
            'df_residual': df_residual,
            'partial_eta_squared': partial_eta,
        }

    def test_returns_false_when_no_artifacts(self):
        self._empty_artifact_dir()
        self.assertFalse(self.ancova.do_aggregate())

    @patch.object(Ancova, 'upload', return_value=True)
    def test_returns_true_with_single_site(self, _):
        self._write_mid_artifact(
            'site1-1-1-mid-artifacts',
            self._make_artifact([1.0, 0.5, 0.3], [0.2, 0.1, 0.15], 80))
        self.assertTrue(self.ancova.do_aggregate())

    @patch.object(Ancova, 'upload', return_value=True)
    def test_inverse_variance_weighted_pooling_of_coefficients(self, _):
        """
        Inverse-variance weighting: w_i = 1/SE_i^2

        Site A: coef=[1.0], std_err=[0.5]   → w = 4
        Site B: coef=[2.0], std_err=[2.0]   → w = 0.25
        total_w = 4.25
        pooled_coef = (1.0*4 + 2.0*0.25) / 4.25 = 4.5/4.25
        """
        self._write_mid_artifact('sA-1-1-mid-artifacts',
            self._make_artifact([1.0], [0.5], 60,
                                ss_model=100.0, ss_residual=400.0, df_residual=58))
        self._write_mid_artifact('sB-1-1-mid-artifacts',
            self._make_artifact([2.0], [2.0], 40,
                                ss_model=50.0, ss_residual=200.0, df_residual=38))
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.ancova.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertAlmostEqual(saved['coef_'][0], 4.5 / 4.25, places=9)

    @patch.object(Ancova, 'upload', return_value=True)
    def test_pooled_sample_size_is_sum(self, _):
        self._write_mid_artifact('s1-1-1-mid-artifacts',
            self._make_artifact([1.0], [0.5], 60))
        self._write_mid_artifact('s2-1-1-mid-artifacts',
            self._make_artifact([2.0], [2.0], 40))
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.ancova.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved['total_sample_size'], 100)

    @patch.object(Ancova, 'upload', return_value=True)
    def test_pooled_f_statistic_computed_from_ss(self, _):
        """
        pooled_f = (total_ss_model/df_model) / (total_ss_residual/total_df_residual)
        With ss_model=100+50=150, df_model=1, ss_residual=400+200=600, df_residual=96:
        pooled_f = (150/1) / (600/96) = 150 / 6.25 = 24.0
        """
        self._write_mid_artifact('sA-1-1-mid-artifacts',
            self._make_artifact([1.0], [0.5], 60,
                                ss_model=100.0, ss_residual=400.0,
                                df_model=1, df_residual=58))
        self._write_mid_artifact('sB-1-1-mid-artifacts',
            self._make_artifact([2.0], [2.0], 40,
                                ss_model=50.0, ss_residual=200.0,
                                df_model=1, df_residual=38))
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.ancova.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertAlmostEqual(saved['f_statistic'], 24.0, places=6)
