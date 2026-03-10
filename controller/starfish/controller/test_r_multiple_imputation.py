import json
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest import skipUnless
from unittest.mock import patch

from starfish.controller.tasks.r_multiple_imputation.task import RMultipleImputation

R_AVAILABLE = shutil.which('Rscript') is not None


def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=1):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {
            'current_round': current_round, 'total_round': total_round,
            'm': 3, 'max_iter': 5,
        }}],
    }


def make_mice_csv(path, n=200, missing_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = 1.0 + X @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.5, n)

    # Inject missing values into features
    mask = rng.random((n, 3)) < missing_frac
    X[mask] = np.nan
    data = np.column_stack([X, y])

    with open(path, 'w') as f:
        for row in data:
            parts = []
            for val in row:
                if np.isnan(val):
                    parts.append('')
                else:
                    parts.append('{:.6f}'.format(val))
            f.write(','.join(parts) + '\n')


@skipUnless(R_AVAILABLE, 'Rscript not found on PATH')
class RMiceTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return RMultipleImputation(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, missing_frac=0.1, seed=42):
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        make_mice_csv(csv_path, n=n, missing_frac=missing_frac, seed=seed)
        return csv_path


class RMicePrepareDataTest(RMiceTestBase):

    @patch.object(RMultipleImputation, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    def test_returns_false_when_no_dataset(self):
        task = self._make_task()
        self.assertFalse(task.prepare_data())


class RMiceTrainingTest(RMiceTestBase):

    @patch.object(RMultipleImputation, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(RMultipleImputation, 'is_first_round', return_value=True)
    def test_training_produces_expected_keys(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        expected_keys = {'sample_size', 'coef', 'se', 'within_var',
                         'between_var', 'p_values', 'missingness_fractions'}
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(RMultipleImputation, 'is_first_round', return_value=True)
    def test_missingness_reported(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        fracs = result['missingness_fractions']
        # Some features should have missing data
        self.assertGreater(max(fracs[:3]), 0)


class RMiceAggregationTest(RMiceTestBase):

    def _write_mid_artifact(self, filename, coef, se, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'complete_cases': int(sample_size * 0.9),
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

    @patch.object(RMultipleImputation, 'upload', return_value=True)
    def test_weighted_average_with_equal_se(self, _):
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
