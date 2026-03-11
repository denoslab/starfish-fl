import json
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest.mock import patch

from starfish.controller.tasks.multiple_imputation.task import MultipleImputation


def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=1):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {
            'current_round': current_round, 'total_round': total_round,
            'm': 3, 'max_iter': 5,
        }}],
    }


def make_mice_data(n=200, missing_frac=0.1, seed=42):
    """Return (X, y) with some NaN values injected into X."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = 1.0 + X @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.5, n)

    # Inject missing values into features
    mask = rng.random((n, 3)) < missing_frac
    X[mask] = np.nan
    return X, y


def write_mice_csv(path, n=200, missing_frac=0.1, seed=42):
    X, y = make_mice_data(n, missing_frac, seed)
    data = np.column_stack([X, y])
    # Use custom writer to preserve NaN as empty cells
    with open(path, 'w') as f:
        for row in data:
            parts = []
            for val in row:
                if np.isnan(val):
                    parts.append('')
                else:
                    parts.append('{:.6f}'.format(val))
            f.write(','.join(parts) + '\n')


class MiceTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return MultipleImputation(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, missing_frac=0.1, seed=42):
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        write_mice_csv(csv_path, n=n, missing_frac=missing_frac, seed=seed)
        return csv_path


class MicePrepareDataTest(MiceTestBase):

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    @patch.object(MultipleImputation, 'read_dataset')
    def test_returns_false_when_dataset_is_none(self, mock_read):
        mock_read.return_value = (None, None)
        self.assertFalse(self._make_task().prepare_data())

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_reads_config_m_and_max_iter(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertEqual(task.m, 3)
        self.assertEqual(task.max_iter, 5)

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_data_contains_nans(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(np.any(np.isnan(task.X_raw)))


class MiceTrainingTest(MiceTestBase):

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_training_produces_expected_keys(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        expected_keys = {
            'sample_size', 'complete_cases', 'm', 'coef', 'se',
            'within_var', 'between_var', 'p_values',
            'missingness_fractions', 'feature_names',
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_missingness_fractions_reported(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        fracs = result['missingness_fractions']
        # Features should have some missing, outcome should have none
        self.assertGreater(max(fracs[:3]), 0)
        self.assertEqual(fracs[-1], 0.0)

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_complete_cases_less_than_total(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        self.assertLess(result['complete_cases'], result['sample_size'])

    @patch.object(MultipleImputation, 'is_first_round', return_value=True)
    def test_within_and_between_variance_positive(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        for v in result['within_var']:
            self.assertGreaterEqual(v, 0)
        for v in result['between_var']:
            self.assertGreaterEqual(v, 0)


class MiceAggregationTest(MiceTestBase):

    def _write_mid_artifact(self, filename, coef, se, sample_size,
                            complete_cases=None):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        if complete_cases is None:
            complete_cases = int(sample_size * 0.9)
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'complete_cases': complete_cases,
            'm': 5,
            'coef': coef,
            'se': se,
            'within_var': [s ** 2 * 0.8 for s in se],
            'between_var': [s ** 2 * 0.2 for s in se],
            't_values': [c / s for c, s in zip(coef, se)],
            'p_values': [0.05] * len(coef),
            'ci_lower': [c - 1.96 * s for c, s in zip(coef, se)],
            'ci_upper': [c + 1.96 * s for c, s in zip(coef, se)],
            'df': [50.0] * len(coef),
            'missingness_fractions': [0.1, 0.1, 0.1, 0.0],
            'feature_names': ['const', 'x1', 'x2', 'x3'],
        }))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts(self):
        self._empty_artifact_dir()
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(MultipleImputation, 'upload', return_value=True)
    def test_inverse_variance_weighted_average(self, _):
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts',
            [1.0, 0.5, -0.3, 0.2], [0.1, 0.1, 0.1, 0.1], 100)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts',
            [1.0, 0.5, -0.3, 0.2], [0.1, 0.1, 0.1, 0.1], 100)
        task = self._make_task()
        task.do_aggregate()
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        np.testing.assert_allclose(
            result['coef'], [1.0, 0.5, -0.3, 0.2], atol=1e-6)
        self.assertEqual(result['sample_size'], 200)

    @patch.object(MultipleImputation, 'upload', return_value=True)
    def test_aggregated_result_has_expected_keys(self, _):
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts',
            [1.0, 0.5, -0.3, 0.2], [0.1, 0.1, 0.1, 0.1], 100)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts',
            [0.8, 0.3, -0.1, 0.4], [0.2, 0.2, 0.2, 0.2], 100)
        task = self._make_task()
        task.do_aggregate()
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        expected_keys = {
            'sample_size', 'complete_cases', 'n_sites', 'coef', 'se',
            'p_values', 'missingness_fractions', 'feature_names',
        }
        self.assertTrue(expected_keys.issubset(result.keys()))
