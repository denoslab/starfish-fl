import json
import shutil
import tempfile
import numpy as np
from pathlib import Path
from scipy import sparse

from django.test import TestCase
from unittest.mock import patch

from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.model_selection import train_test_split

from starfish.controller.tasks.mixed_effects_logistic_regression import (
    MixedEffectsLogisticRegression,
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


def make_mixed_effects_data(n_groups=6, n_per_group=20, n_features=2, seed=42):
    """
    Return (X_with_group, y_binary) where X_with_group[:, 0] is the integer
    group column and X_with_group[:, 1:] are continuous predictors.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    X = rng.standard_normal((n, n_features))
    group_intercepts = rng.standard_normal(n_groups) * 0.5
    logits = X @ np.array([1.5, -1.0]) + group_intercepts[groups]
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    X_with_group = np.column_stack([groups.astype(float), X])
    return X_with_group, y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MixedEffectsTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_me(self, **kwargs):
        return MixedEffectsLogisticRegression(make_run(**kwargs))

    def _setup_trained_me(self, me, n_groups=6, n_per_group=20, seed=42):
        X_full, y = make_mixed_effects_data(n_groups, n_per_group, seed=seed)
        groups_raw = X_full[:, 0].astype(int)
        X_pred = X_full[:, 1:]
        y = y.astype(int)

        unique_groups = np.unique(groups_raw)
        group_mapping = {g: i for i, g in enumerate(unique_groups)}
        groups_coded = np.array([group_mapping[g] for g in groups_raw])

        X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
            X_pred, y, groups_coded,
            test_size=0.2, random_state=42, stratify=groups_coded)

        me.X = X_train.astype(float)
        me.y = y_train
        me.groups = g_train
        me.X_test = X_test.astype(float)
        me.y_test = y_test
        me.groups_test = g_test
        me.sample_size = n_groups * n_per_group
        me.n_groups = n_groups
        me.group_labels = unique_groups.tolist()
        me.group_mapping = group_mapping
        me.group_counts = {
            me.group_labels[i]: int(np.sum(groups_coded == i))
            for i in range(n_groups)
        }
        me.vcp_p = 1.0
        me.fe_p = 2.0

        # Fit the model (mirrors training())
        n_obs = len(me.groups)
        n_groups_train = len(np.unique(me.groups))
        X_with_intercept = np.column_stack([np.ones(n_obs), me.X])
        exog_vc = sparse.csr_matrix(
            (np.ones(n_obs), (np.arange(n_obs), me.groups)),
            shape=(n_obs, n_groups_train))
        ident = np.zeros(n_groups_train, dtype=int)
        n_pred = me.X.shape[1]
        fep_names = ['Intercept'] + [f'X{i+1}' for i in range(n_pred)]
        vcp_names = ['Group_Intercept_SD']
        vc_names = [f'RE_Group_{me.group_labels[i]}' for i in range(n_groups_train)]
        model = BinomialBayesMixedGLM(
            endog=me.y, exog=X_with_intercept,
            exog_vc=exog_vc, ident=ident,
            vcp_p=me.vcp_p, fe_p=me.fe_p,
            fep_names=fep_names, vcp_names=vcp_names, vc_names=vc_names)
        me.model_result = model.fit_vb(verbose=False)


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class MixedEffectsPrepareDataTest(MixedEffectsTestBase):

    @patch.object(MixedEffectsLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(MixedEffectsLogisticRegression, 'read_dataset')
    def test_returns_true_with_valid_grouped_data(self, mock_read, _):
        mock_read.return_value = make_mixed_effects_data()
        self.assertTrue(self._make_me().prepare_data())

    @patch.object(MixedEffectsLogisticRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_me().prepare_data())

    @patch.object(MixedEffectsLogisticRegression, 'read_dataset')
    def test_returns_false_when_dataset_is_empty(self, mock_read):
        mock_read.return_value = (np.array([]), np.array([]))
        self.assertFalse(self._make_me().prepare_data())

    @patch.object(MixedEffectsLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(MixedEffectsLogisticRegression, 'read_dataset')
    def test_returns_false_with_non_binary_target(self, mock_read, _):
        """Target with more than 2 unique values must be rejected."""
        X_full, _ = make_mixed_effects_data()
        y_multi = np.tile([0, 1, 2], len(X_full) // 3 + 1)[:len(X_full)]
        mock_read.return_value = (X_full, y_multi)
        self.assertFalse(self._make_me().prepare_data())

    @patch.object(MixedEffectsLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(MixedEffectsLogisticRegression, 'read_dataset')
    def test_extracts_group_column_and_sets_n_groups(self, mock_read, _):
        """First column of X should be used as group identifiers."""
        mock_read.return_value = make_mixed_effects_data(n_groups=6, n_per_group=20)
        me = self._make_me()
        me.prepare_data()
        self.assertEqual(me.n_groups, 6)

    @patch.object(MixedEffectsLogisticRegression, 'is_first_round', return_value=True)
    @patch.object(MixedEffectsLogisticRegression, 'read_dataset')
    def test_sets_correct_sample_size(self, mock_read, _):
        mock_read.return_value = make_mixed_effects_data(n_groups=6, n_per_group=20)
        me = self._make_me()
        me.prepare_data()
        self.assertEqual(me.sample_size, 120)


# ---------------------------------------------------------------------------
# training / calculate_statistics
# ---------------------------------------------------------------------------

class MixedEffectsTrainingTest(MixedEffectsTestBase):
    def setUp(self):
        super().setUp()
        self.me = self._make_me()
        self._setup_trained_me(self.me)

    def test_training_returns_true(self):
        fresh = self._make_me()
        X_full, y = make_mixed_effects_data()
        groups_raw = X_full[:, 0].astype(int)
        X_pred = X_full[:, 1:]
        unique_groups = np.unique(groups_raw)
        group_mapping = {g: i for i, g in enumerate(unique_groups)}
        groups_coded = np.array([group_mapping[g] for g in groups_raw])
        X_train, _, y_train, _, g_train, _ = train_test_split(
            X_pred, y.astype(int), groups_coded,
            test_size=0.2, random_state=42, stratify=groups_coded)
        fresh.X = X_train.astype(float)
        fresh.y = y_train
        fresh.groups = g_train
        fresh.n_groups = len(unique_groups)
        fresh.group_labels = unique_groups.tolist()
        fresh.vcp_p = 1.0
        fresh.fe_p = 2.0
        fresh.sample_size = 120
        self.assertTrue(fresh.training())

    def test_calculate_statistics_contains_required_keys(self):
        stats = self.me.calculate_statistics()
        expected = {
            'sample_size', 'n_groups', 'fe_coef', 'fe_std_err',
            'fe_z_values', 'fe_p_values', 'odds_ratios',
            'vcp_mean_log', 'random_effect_sd', 'random_intercepts', 'icc',
        }
        self.assertTrue(expected.issubset(stats.keys()))

    def test_icc_between_0_and_1(self):
        stats = self.me.calculate_statistics()
        self.assertGreaterEqual(stats['icc'], 0.0)
        self.assertLessEqual(stats['icc'], 1.0)

    def test_odds_ratios_are_positive(self):
        stats = self.me.calculate_statistics()
        for OR in stats['odds_ratios']:
            self.assertGreater(OR, 0.0)

    def test_odds_ratios_equal_exp_of_fe_coef(self):
        stats = self.me.calculate_statistics()
        expected_OR = np.exp(stats['fe_coef']).tolist()
        np.testing.assert_allclose(stats['odds_ratios'], expected_OR, rtol=1e-6)

    def test_output_is_json_serialisable(self):
        stats = self.me.calculate_statistics()
        json.dumps(stats)  # must not raise


# ---------------------------------------------------------------------------
# do_aggregate  (pass-through / centralized)
# ---------------------------------------------------------------------------

class MixedEffectsAggregationTest(MixedEffectsTestBase):
    def setUp(self):
        super().setUp()
        self.me = self._make_me()
        self._setup_trained_me(self.me)

    def _write_mid_artifact(self, filename, payload):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps(payload))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def _make_artifact(self):
        return {
            'sample_size': 96,
            'n_groups': 6,
            'group_counts': {str(i): 16 for i in range(6)},
            'group_labels': [str(i) for i in range(6)],
            'fe_coef': [0.1, 0.5, -0.3],
            'fe_std_err': [0.2, 0.3, 0.15],
            'fe_z_values': [0.5, 1.7, -2.0],
            'fe_p_values': [0.6, 0.09, 0.045],
            'fe_conf_int_lower': [-0.3, -0.1, -0.6],
            'fe_conf_int_upper': [0.5, 1.1, 0.0],
            'odds_ratios': [1.1, 1.65, 0.74],
            'odds_ratio_ci_lower': [0.74, 0.9, 0.55],
            'odds_ratio_ci_upper': [1.65, 3.0, 1.0],
            'vcp_mean_log': [-0.5],
            'vcp_sd_log': [0.3],
            'random_effect_sd': [0.6],
            'random_effect_var': [0.36],
            'random_intercepts': {str(i): {'mean': 0.0, 'sd': 0.2} for i in range(6)},
            'icc': 0.1,
            'vcp_p': 1.0,
            'fe_p': 2.0,
        }

    def test_returns_false_when_no_artifacts(self):
        self._empty_artifact_dir()
        self.assertFalse(self.me.do_aggregate())

    @patch.object(MixedEffectsLogisticRegression, 'upload', return_value=True)
    def test_returns_true_with_single_site(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', self._make_artifact())
        self.assertTrue(self.me.do_aggregate())

    @patch.object(MixedEffectsLogisticRegression, 'upload', return_value=True)
    def test_saved_artifact_has_analysis_type_centralized(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', self._make_artifact())
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.me.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved['analysis_type'], 'centralized')

    @patch.object(MixedEffectsLogisticRegression, 'upload', return_value=True)
    def test_saved_artifact_has_n_sites_1(self, _):
        self._write_mid_artifact('site1-1-1-mid-artifacts', self._make_artifact())
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.me.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved['n_sites'], 1)

    @patch.object(MixedEffectsLogisticRegression, 'upload', return_value=True)
    def test_saved_artifact_preserves_site_stats(self, _):
        artifact = self._make_artifact()
        self._write_mid_artifact('site1-1-1-mid-artifacts', artifact)
        result_path = Path(self.tmp_dir) / '42' / '1' / '1' / 'artifacts'
        self.me.do_aggregate()
        saved = json.loads(result_path.read_text())
        self.assertEqual(saved['fe_coef'], artifact['fe_coef'])
        self.assertAlmostEqual(saved['icc'], artifact['icc'])
