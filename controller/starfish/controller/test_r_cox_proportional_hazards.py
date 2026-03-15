import json
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

from django.test import TestCase
from unittest import skipUnless
from unittest.mock import patch

from starfish.controller.tasks.r_cox_proportional_hazards.task import RCoxProportionalHazards

R_AVAILABLE = shutil.which('Rscript') is not None


def make_run(role='coordinator', cur_seq=1, current_round=1, total_round=3):
    return {
        'id': 42, 'project': 7, 'batch': 1, 'role': role,
        'status': 'standby', 'cur_seq': cur_seq,
        'tasks': [{'config': {'current_round': current_round, 'total_round': total_round}}],
    }


def make_survival_csv(path, n=200, seed=42):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, 2))
    group = (features[:, 0] > 0).astype(float)
    log_hr = features @ np.array([0.5, -0.3])
    scale = 1.0 / (0.1 * np.exp(log_hr))
    time = rng.exponential(scale)
    censor_time = rng.uniform(0, np.percentile(time, 80), n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(float)
    data = np.column_stack([group, features, observed_time, event])
    np.savetxt(path, data, delimiter=',', fmt='%.6f')


@skipUnless(R_AVAILABLE, 'Rscript not found on PATH')
class RCoxPHTestBase(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patcher = patch(
            'starfish.controller.file.file_utils.base_folder', self.tmp_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_task(self, **kwargs):
        return RCoxProportionalHazards(make_run(**kwargs))

    def _setup_dataset(self, run_id=42, n=200, seed=42):
        dataset_dir = os.path.join(self.tmp_dir, str(run_id))
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, 'dataset')
        make_survival_csv(csv_path, n=n, seed=seed)
        return csv_path


class RCoxPHPrepareDataTest(RCoxPHTestBase):

    @patch.object(RCoxProportionalHazards, 'is_first_round', return_value=True)
    def test_returns_true_with_valid_data(self, _):
        self._setup_dataset()
        task = self._make_task()
        self.assertTrue(task.prepare_data())

    def test_returns_false_when_no_dataset(self):
        task = self._make_task()
        self.assertFalse(task.prepare_data())


class RCoxPHTrainingTest(RCoxPHTestBase):

    @patch.object(RCoxProportionalHazards, 'is_first_round', return_value=True)
    def test_training_returns_true(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        self.assertTrue(task.training())

    @patch.object(RCoxProportionalHazards, 'is_first_round', return_value=True)
    def test_training_produces_expected_keys(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        expected_keys = {'sample_size', 'coef', 'se', 'hazard_ratio',
                         'p_values', 'concordance_index'}
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(RCoxProportionalHazards, 'is_first_round', return_value=True)
    def test_hazard_ratios_are_positive(self, _):
        self._setup_dataset()
        task = self._make_task()
        task.prepare_data()
        task.training()
        mid_artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'mid-artifacts')
        with open(mid_artifact_path, 'r') as f:
            result = json.load(f)
        for hr in result['hazard_ratio']:
            self.assertGreater(hr, 0)


class RCoxPHAggregationTest(RCoxPHTestBase):

    def _write_mid_artifact(self, filename, coef, se, sample_size):
        dir_path = Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1'
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / filename).write_text(json.dumps({
            'sample_size': sample_size,
            'coef': coef,
            'se': se,
            'hazard_ratio': [float(np.exp(c)) for c in coef],
            'p_values': [0.05] * len(coef),
            'ci_lower': [c - 1.96 * s for c, s in zip(coef, se)],
            'ci_upper': [c + 1.96 * s for c, s in zip(coef, se)],
            'concordance_index': 0.65,
            'feature_names': ['x1', 'x2'],
        }))

    def _empty_artifact_dir(self):
        (Path(self.tmp_dir) / 'all-mid-artifacts' / '7' / '1').mkdir(
            parents=True, exist_ok=True)

    def test_returns_false_when_no_mid_artifacts(self):
        self._empty_artifact_dir()
        task = self._make_task()
        self.assertFalse(task.do_aggregate())

    @patch.object(RCoxProportionalHazards, 'upload', return_value=True)
    def test_weighted_average_with_equal_se(self, _):
        self._write_mid_artifact(
            'siteA-1-1-mid-artifacts', [0.6, -0.2], [0.1, 0.1], 100)
        self._write_mid_artifact(
            'siteB-1-1-mid-artifacts', [0.4, -0.4], [0.1, 0.1], 100)
        task = self._make_task()
        task.do_aggregate()
        artifact_path = os.path.join(
            self.tmp_dir, '42', '1', '1', 'artifacts')
        with open(artifact_path, 'r') as f:
            result = json.load(f)
        np.testing.assert_allclose(result['coef'], [0.5, -0.3], atol=1e-6)
