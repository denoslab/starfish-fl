"""
Regression tests for Cox Proportional Hazards using the veteran dataset.

Dataset: Veterans' Administration Lung Cancer Trial
Source: Kalbfleisch & Prentice (1980), R survival package
Reference: lifelines CoxPHFitter / R coxph() on full dataset (137 obs)

Tests verify:
1. Centralized results match published reference values
2. Federated (3-site split) results are consistent with centralized
3. Python and R implementations agree on the same data

Note: The Python task class (CoxProportionalHazards) performs an internal
80/20 train/test split, while the R task (RCoxProportionalHazards) trains
on the full dataset. Centralized and cross-language tests therefore use
lifelines directly for an apples-to-apples comparison on the full dataset.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from pathlib import Path

from django.test import TestCase
from unittest import skipUnless
from unittest.mock import patch

from starfish.controller.tasks.cox_proportional_hazards.task import CoxProportionalHazards
from starfish.controller.tasks.r_cox_proportional_hazards.task import RCoxProportionalHazards

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'veteran')
VETERAN_PY_CSV = os.path.join(FIXTURE_DIR, 'veteran_py.csv')
VETERAN_R_CSV = os.path.join(FIXTURE_DIR, 'veteran_r.csv')

R_AVAILABLE = shutil.which('Rscript') is not None

# ---------------------------------------------------------------------------
# Reference values from R coxph() on full veteran dataset (137 obs).
# Reproduced by lifelines CoxPHFitter on the same data.
# See fixtures/veteran/README.md for details.
#
# Feature order: trt_test, celltype_smallcell, celltype_adeno,
#   celltype_large, karno, diagtime, age, prior_yes
# ---------------------------------------------------------------------------
REF_COEF = [0.294605, 0.861556, 1.196075, 0.40129,
            -0.032816, 8.3e-05, -0.008706, 0.071586]
REF_SE = [0.20755, 0.275284, 0.300917, 0.282689,
          0.005508, 0.009136, 0.0093, 0.232305]
REF_HR = [1.342596, 2.36684, 3.30711, 1.49375,
          0.967717, 1.000083, 0.991332, 1.07421]
REF_CONCORDANCE = 0.736029

# Indices of features significant at alpha=0.05
#   celltype_smallcell (1), celltype_adeno (2), karno (4)
SIGNIFICANT_IDX = [1, 2, 4]
NON_SIGNIFICANT_IDX = [0, 3, 5, 6, 7]

N_FEATURES = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=1):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round,
                               'total_round': total_round}}],
    }


def load_veteran_py():
    """Load the Python-format veteran CSV as (X, y)."""
    data = np.loadtxt(VETERAN_PY_CSV, delimiter=',')
    X = data[:, :-1]   # features + time
    y = data[:, -1]     # event
    return X, y


def fit_full_dataset():
    """Fit CoxPHFitter on the full veteran dataset (no train/test split).

    Returns the fitted model's summary dict with keys matching the task
    output format: coef, se, hazard_ratio, p_values, concordance_index.
    """
    X, y = load_veteran_py()
    features = X[:, :-1]
    time = X[:, -1]
    col_names = [f'x{i}' for i in range(N_FEATURES)]
    df = pd.DataFrame(features, columns=col_names)
    df['time'] = time
    df['event'] = y

    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    summary = cph.summary
    return {
        'coef': cph.params_.values.tolist(),
        'se': summary['se(coef)'].values.tolist(),
        'hazard_ratio': summary['exp(coef)'].values.tolist(),
        'p_values': summary['p'].values.tolist(),
        'concordance_index': cph.concordance_index_,
    }


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RegressionCoxPHTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _setup_py_dataset(self, run_id=42, X=None, y=None):
        """Write Python-format CSV for a given run."""
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        if X is None or y is None:
            shutil.copy(VETERAN_PY_CSV, csv_path)
        else:
            data = np.column_stack([X, y])
            np.savetxt(csv_path, data, delimiter=',', fmt='%.6f')
        return csv_path

    def _setup_r_dataset(self, run_id=42, data=None):
        """Write R-format CSV for a given run."""
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        if data is None:
            shutil.copy(VETERAN_R_CSV, csv_path)
        else:
            np.savetxt(csv_path, data, delimiter=',', fmt='%.6f')
        return csv_path

    def _read_mid_artifacts(self, run_id=42, cur_seq=1, round_no=1):
        path = os.path.join(
            self.tmp_dir, str(run_id), str(cur_seq),
            str(round_no), 'mid-artifacts')
        with open(path, 'r') as f:
            return json.load(f)

    def _read_artifacts(self, run_id=42, cur_seq=1, round_no=1):
        path = os.path.join(
            self.tmp_dir, str(run_id), str(cur_seq),
            str(round_no), 'artifacts')
        with open(path, 'r') as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# 1. Centralized regression test — lifelines on full dataset vs reference
# ---------------------------------------------------------------------------

class CoxPHCentralizedRegressionTest(TestCase):
    """Fit lifelines CoxPHFitter on the full veteran dataset (no train/test
    split) and compare to published reference values from R coxph()."""

    def test_coefficients_match_reference(self):
        result = fit_full_dataset()
        np.testing.assert_allclose(
            result['coef'], REF_COEF, atol=0.01,
            err_msg='Coefficients deviate from reference values')

    def test_standard_errors_match_reference(self):
        result = fit_full_dataset()
        np.testing.assert_allclose(
            result['se'], REF_SE, atol=0.01,
            err_msg='Standard errors deviate from reference values')

    def test_hazard_ratios_match_reference(self):
        result = fit_full_dataset()
        np.testing.assert_allclose(
            result['hazard_ratio'], REF_HR, atol=0.05,
            err_msg='Hazard ratios deviate from reference values')

    def test_significance_matches_reference(self):
        """Verify the same variables are significant / non-significant."""
        result = fit_full_dataset()
        for idx in SIGNIFICANT_IDX:
            self.assertLess(
                result['p_values'][idx], 0.05,
                f'Feature {idx} should be significant (p < 0.05)')
        for idx in NON_SIGNIFICANT_IDX:
            self.assertGreaterEqual(
                result['p_values'][idx], 0.05,
                f'Feature {idx} should be non-significant (p >= 0.05)')

    def test_concordance_index_matches_reference(self):
        result = fit_full_dataset()
        self.assertAlmostEqual(
            result['concordance_index'], REF_CONCORDANCE, delta=0.02,
            msg='Concordance index deviates from reference')


# ---------------------------------------------------------------------------
# 2. Federated consistency test — 3-site split vs centralized
# ---------------------------------------------------------------------------

class CoxPHFederatedConsistencyTest(RegressionCoxPHTestBase):
    """Split veteran into 3 partitions, run federated training + aggregation
    using the task class, and compare to full-dataset reference values.

    The task class does an internal 80/20 train/test split per site, and
    each site only has ~46 rows, so wider tolerances are expected."""

    def _split_data(self):
        """Deterministic 3-way split of the veteran dataset."""
        X, y = load_veteran_py()
        rng = np.random.RandomState(123)
        idx = rng.permutation(len(y))
        splits = np.array_split(idx, 3)
        return [(X[s], y[s]) for s in splits]

    def _run_federated(self):
        """Train on 3 partitions, aggregate, return aggregated result."""
        partitions = self._split_data()

        # Train each partition and collect mid-artifacts
        site_results = []
        for i, (X_part, y_part) in enumerate(partitions):
            run_id = 100 + i
            self._setup_py_dataset(run_id=run_id, X=X_part, y=y_part)
            run = make_run()
            run['id'] = run_id
            task = CoxProportionalHazards(run)
            with patch.object(CoxProportionalHazards, 'is_first_round',
                              return_value=True):
                self.assertTrue(task.prepare_data())
                self.assertTrue(task.training())
            result = self._read_mid_artifacts(run_id=run_id)
            site_results.append(result)

        # Write mid-artifacts for aggregation
        agg_dir = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        agg_dir.mkdir(parents=True, exist_ok=True)
        for i, result in enumerate(site_results):
            artifact_path = agg_dir / f'site{i}-1-1-mid-artifacts'
            artifact_path.write_text(json.dumps(result))

        # Run aggregation
        task = CoxProportionalHazards(make_run())
        with patch.object(CoxProportionalHazards, 'upload',
                          return_value=True):
            self.assertTrue(task.do_aggregate())

        return self._read_artifacts(), site_results

    def test_federated_coefficients_close_to_centralized(self):
        """With ~37 training rows per site (80% of ~46), wider tolerance
        is needed compared to the full-dataset reference."""
        agg_result, _ = self._run_federated()
        np.testing.assert_allclose(
            agg_result['coef'], REF_COEF, atol=0.5,
            err_msg='Federated coefficients too far from centralized')

    def test_federated_total_sample_size(self):
        agg_result, site_results = self._run_federated()
        expected_total = sum(r['sample_size'] for r in site_results)
        self.assertEqual(agg_result['sample_size'], expected_total)
        self.assertEqual(expected_total, 137)

    def test_federated_significance_direction_matches(self):
        """Sign of significant coefficients should match centralized."""
        agg_result, _ = self._run_federated()
        for idx in SIGNIFICANT_IDX:
            ref_sign = np.sign(REF_COEF[idx])
            fed_sign = np.sign(agg_result['coef'][idx])
            self.assertEqual(
                ref_sign, fed_sign,
                f'Feature {idx}: federated sign ({fed_sign}) != '
                f'centralized sign ({ref_sign})')


# ---------------------------------------------------------------------------
# 3. Cross-language test — R task vs lifelines (both on full dataset)
# ---------------------------------------------------------------------------

@skipUnless(R_AVAILABLE, 'Rscript not found on PATH')
class CoxPHCrossLanguageTest(RegressionCoxPHTestBase):
    """Compare R task output (trains on full dataset) against lifelines
    CoxPHFitter on the full dataset. Both fit on 100% of the data, making
    this a true apples-to-apples comparison."""

    def _train_r(self):
        self._setup_r_dataset(run_id=60)
        run = make_run()
        run['id'] = 60
        task = RCoxProportionalHazards(run)
        with patch.object(RCoxProportionalHazards, 'is_first_round',
                          return_value=True):
            task.prepare_data()
            task.training()
        return self._read_mid_artifacts(run_id=60)

    def test_r_coefficients_match_reference(self):
        r = self._train_r()
        np.testing.assert_allclose(
            r['coef'], REF_COEF, atol=0.01,
            err_msg='R coefficients deviate from reference values')

    def test_r_standard_errors_match_reference(self):
        r = self._train_r()
        np.testing.assert_allclose(
            r['se'], REF_SE, atol=0.01,
            err_msg='R standard errors deviate from reference values')

    def test_r_hazard_ratios_match_reference(self):
        r = self._train_r()
        np.testing.assert_allclose(
            r['hazard_ratio'], REF_HR, atol=0.05,
            err_msg='R hazard ratios deviate from reference values')

    def test_python_r_concordance_agree(self):
        r = self._train_r()
        py = fit_full_dataset()
        self.assertAlmostEqual(
            py['concordance_index'], r['concordance_index'], delta=0.03,
            msg='Python and R concordance indices disagree')
