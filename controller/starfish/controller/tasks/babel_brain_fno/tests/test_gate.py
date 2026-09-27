"""Tests for the SF-05 metrics and evaluation gate."""

import json
import math
import os
import shutil
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from starfish.controller.file.artifact_io import load_artifact, save_artifact
from starfish.controller.tasks.babel_brain_fno import gate as G
from starfish.controller.tasks.babel_brain_fno import metrics as M
from starfish.controller.tasks.babel_brain_fno.synthetic import write_synthetic_store
from starfish.controller.tasks.babel_brain_fno.task import BabelBrainFno
from starfish.controller.tasks.babel_brain_fno.tests.test_fno_task import (
    HAS_TORCH, FnoTaskTestCase, make_run)

SHAPE = (24, 24, 32)


def focus(center, peak=1.0, width=3.0):
    """A complex field with one Gaussian focus: shape (2, X, Y, Z)."""
    grid = np.indices(SHAPE).astype(np.float64)
    r2 = sum((g - c) ** 2 for g, c in zip(grid, center))
    amp = peak * np.exp(-r2 / (2 * width ** 2))
    return np.stack([amp * 0.6, amp * 0.8])


class MetricsTest(TestCase):

    def setUp(self):
        self.mask = np.ones(SHAPE, dtype=np.uint8)
        self.true = focus((12, 12, 16))

    def test_identical_fields(self):
        m = M.sample_metrics(self.true, self.true, self.mask, 0.49)
        self.assertEqual(m['rel_l2_pct'], 0.0)
        self.assertAlmostEqual(m['ssim'], 1.0, places=9)
        self.assertTrue(math.isinf(m['psnr_db']))
        self.assertEqual(m['peak_distance_mm'], 0.0)
        self.assertEqual(m['peak_amplitude_error_pct'], 0.0)
        self.assertAlmostEqual(m['focal_centroid_distance_mm'], 0.0, places=9)

    def test_shifted_focus_gives_distances_in_mm(self):
        pred = focus((12, 12, 18))
        m = M.sample_metrics(pred, self.true, self.mask, 0.5)
        self.assertAlmostEqual(m['peak_distance_mm'], 1.0)
        self.assertAlmostEqual(m['focal_centroid_distance_mm'], 1.0, places=6)
        self.assertLess(m['ssim'], 1.0)

    def test_scaled_field_gives_relative_errors(self):
        m = M.sample_metrics(0.9 * self.true, self.true, self.mask, 0.49)
        self.assertAlmostEqual(m['rel_l2_pct'], 10.0, places=9)
        self.assertAlmostEqual(m['peak_amplitude_error_pct'], 10.0, places=9)
        self.assertEqual(m['peak_distance_mm'], 0.0)

    def test_focal_metrics_ignore_voxels_outside_the_brain(self):
        pred = self.true + focus((3, 3, 3), peak=5.0, width=1.0)
        mask = np.ones(SHAPE, dtype=np.uint8)
        mask[:8, :8, :8] = 0
        m = M.sample_metrics(pred, self.true, mask, 0.49)
        self.assertEqual(m['peak_distance_mm'], 0.0)
        self.assertGreater(m['rel_l2_pct'], 0.0)

    def test_empty_brain_mask_is_an_error(self):
        with self.assertRaises(ValueError):
            M.sample_metrics(self.true, self.true, np.zeros(SHAPE), 0.49)

    def test_summary_gives_mean_std_median(self):
        rows = [dict.fromkeys(M.METRICS, v) for v in (1.0, 2.0, 6.0)]
        s = M.summarize(rows)
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['rel_l2_pct'], {'mean': 3.0, 'std': float(np.std([1, 2, 6])),
                                           'median': 2.0})

    def test_metric_choices_confirmed_on_2026_09_27(self):
        """The metric details confirmed in T7. Change them only with the spec."""
        self.assertEqual(M.SSIM_WINDOW, 7)
        self.assertEqual(M.DATA_RANGE, 'true_max_minus_min')
        self.assertEqual(M.PEAK_ERROR_AT, 'own_peak')
        self.assertAlmostEqual(M.FOCAL_AMPLITUDE_FRACTION, 0.70795, places=5)
        self.assertFalse(M.CENTROID_WEIGHTED)
        self.assertTrue(G.REGION_BREAKDOWN_AVAILABLE)


class RegionBreakdownTest(TestCase):
    """D6: summaries per region class from the schema 1.1 labels."""

    class Record:
        def __init__(self, region):
            self.region = region

    def rows(self, *values):
        return [dict.fromkeys(M.METRICS, v) for v in values]

    def test_rows_are_grouped_by_region_in_contract_order(self):
        records = [self.Record(r) for r in ('TP8', 'P7', 'TP8', 'other')]
        regions = G.by_region(self.rows(1.0, 2.0, 3.0, 4.0), records)
        self.assertEqual(list(regions), ['P7', 'TP8', 'other'])
        self.assertEqual(regions['TP8']['n'], 2)
        self.assertEqual(regions['TP8']['rel_l2_pct']['mean'], 2.0)

    def test_no_labels_give_no_breakdown(self):
        records = [self.Record(None), self.Record(None)]
        self.assertIsNone(G.by_region(self.rows(1.0, 2.0), records))
        self.assertIsNone(G.region_breakdown(None, None))

    def test_breakdown_pairs_current_and_candidate(self):
        current = {'P7': {'n': 1}, 'P8': {'n': 2}}
        candidate = {'P7': {'n': 1}, 'P8': {'n': 2}}
        self.assertEqual(G.region_breakdown(current, candidate), {
            'P7': {'current': {'n': 1}, 'candidate': {'n': 1}},
            'P8': {'current': {'n': 2}, 'candidate': {'n': 2}}})


def scores(rel, peak):
    return {'rel_l2_pct': {'mean': rel}, 'peak_distance_mm': {'mean': peak}}


class DecideTest(TestCase):

    def test_default_margins_block_any_regression(self):
        cfg = G.gate_config({})
        self.assertTrue(G.decide(scores(10.0, 1.0), scores(10.0, 1.0), cfg)[0])
        self.assertTrue(G.decide(scores(9.0, 0.9), scores(10.0, 1.0), cfg)[0])
        accepted, reasons = G.decide(
            scores(10.01, 1.0), scores(10.0, 1.0), cfg)
        self.assertFalse(accepted)
        self.assertIn('rel_l2_pct', reasons[0])
        self.assertFalse(G.decide(scores(10.0, 1.2),
                         scores(10.0, 1.0), cfg)[0])

    def test_margins_from_config(self):
        cfg = G.gate_config({'gate': {'max_rel_l2_increase': 0.5,
                                      'max_peak_distance_increase_mm': 0.25}})
        self.assertTrue(G.decide(scores(10.4, 1.2), scores(10.0, 1.0), cfg)[0])
        self.assertFalse(G.decide(scores(10.6, 1.2),
                         scores(10.0, 1.0), cfg)[0])

    def test_non_finite_candidate_is_rejected(self):
        self.assertFalse(G.decide(scores(float('nan'), 1.0), scores(10.0, 1.0),
                                  G.gate_config({}))[0])


@skipUnless(HAS_TORCH, 'torch is not installed')
class GateInTaskTest(FnoTaskTestCase):

    def setUp(self):
        super().setUp()
        self.eval_store = os.path.join(self.tmp, 'eval')
        write_synthetic_store(self.eval_store, n_groups=2,
                              per_group=2, val_groups=0, seed=9)
        p = patch.dict(os.environ, {G.EVAL_STORE_ENV: self.eval_store})
        p.start()
        self.addCleanup(p.stop)

    def coordinator(self, **gate):
        site = self.trained_site(1, gate=dict({'enabled': True}, **gate))
        self.deliver(site, site)
        return site

    def report(self, site):
        with open(os.path.join(self.base, '1', '1', '1', 'eval_report.json')) as f:
            return json.load(f)

    def test_candidate_within_margins_is_published_with_its_scores(self):
        site = self.coordinator(max_rel_l2_increase=1e9,
                                max_peak_distance_increase_mm=1e9)
        self.assertTrue(self.aggregate(site, 1))
        _, meta = self.global_out(site)
        self.assertTrue(meta['metrics']['accepted'])
        self.assertEqual(meta['model_version'], 'standin-250k-p7-b1-t1-r1')
        self.assertEqual(meta['metrics']['eval']
                         ['model_version'], meta['model_version'])
        report = self.report(site)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['eval_samples'], 4)
        self.assertEqual(report['current_model'], 'init:0')
        for part in ('current', 'candidate'):
            self.assertEqual(set(report[part]) - {'n'}, set(M.METRICS))
        self.assertIsNone(report['region_breakdown'])

    def test_corrupted_delta_is_blocked_and_previous_model_stays_current(self):
        site = self.coordinator()
        base, base_meta = site.current_global()
        path = os.path.join(site._mids_dir(), '1-1-1-mid-artifacts')
        delta, meta = load_artifact(path)
        rng = np.random.default_rng(0)
        corrupted = {k: (v + 50.0 * rng.standard_normal(v.shape)).astype(v.dtype)
                     for k, v in delta.items()}
        save_artifact(path, corrupted, meta)
        self.assertTrue(self.aggregate(site, 1))
        result, gmeta = self.global_out(site)
        self.assertFalse(gmeta['metrics']['accepted'])
        self.assertEqual(gmeta['model_version'], base_meta['model_version'])
        for name in base:
            np.testing.assert_array_equal(result[name], base[name])
        report = self.report(site)
        self.assertFalse(report['accepted'])
        self.assertTrue(report['reasons'])

    def test_unchanged_model_passes_the_default_gate(self):
        site = self.coordinator()
        path = os.path.join(site._mids_dir(), '1-1-1-mid-artifacts')
        delta, meta = load_artifact(path)
        save_artifact(path, {k: np.zeros_like(v)
                      for k, v in delta.items()}, meta)
        self.assertTrue(self.aggregate(site, 1))
        self.assertTrue(self.global_out(site)[1]['metrics']['accepted'])

    def test_missing_eval_store_fails_the_round(self):
        site = self.coordinator()
        with patch.dict(os.environ, {G.EVAL_STORE_ENV: ''}):
            self.assertFalse(self.aggregate(site, 1))
        self.assertFalse(os.path.exists(site._global_out_path()))

    def test_current_models_score_is_reused_when_the_eval_set_is_unchanged(self):
        self.second_round_from_the_published_model()

    def second_round_from_the_published_model(self):
        """Round 2 on an unchanged eval set: the current model's cached score is used."""
        site = self.coordinator(max_rel_l2_increase=1e9,
                                max_peak_distance_increase_mm=1e9)
        self.assertTrue(self.aggregate(site, 1))
        published = site._global_out_path()
        round2 = BabelBrainFno(make_run(run_id=1, current_round=2, gate={
            'enabled': True, 'max_rel_l2_increase': 1e9, 'max_peak_distance_increase_mm': 1e9}))

        def fake_download(run_id, file_type, folder, task_seq=None, round_seq=None, all_runs=False):
            os.makedirs(folder, exist_ok=True)
            target = os.path.join(folder, '1-1-1-artifacts')
            shutil.copy(published, target)
            return [target]

        with patch('starfish.controller.tasks.babel_brain_fno.task.transfer.download_all',
                   side_effect=fake_download):
            self.assertTrue(round2.validate())
            self.assertTrue(round2.prepare_data())
            self.assertTrue(round2.training())
        self.deliver(round2, round2)
        with patch.object(G, 'evaluate', wraps=G.evaluate) as spy:
            self.assertTrue(self.aggregate(round2, 1))
        self.assertEqual(spy.call_count, 1,
                         'only the candidate should be scored')


class RegionGateInTaskTest(GateInTaskTest):
    """The same gate on a schema 1.1 eval store: the report breaks results down by region."""

    def setUp(self):
        super().setUp()
        shutil.rmtree(self.eval_store)
        write_synthetic_store(self.eval_store, n_groups=2, per_group=2,
                              val_groups=0, seed=9, regions=True)

    def test_candidate_within_margins_is_published_with_its_scores(self):
        site = self.coordinator(max_rel_l2_increase=1e9,
                                max_peak_distance_increase_mm=1e9)
        self.assertTrue(self.aggregate(site, 1))
        breakdown = self.report(site)['region_breakdown']
        self.assertEqual(list(breakdown), ['P7', 'P8', 'PO7', 'TP7'])
        for region in breakdown.values():
            for part in ('current', 'candidate'):
                self.assertEqual(region[part]['n'], 1)
                self.assertEqual(set(region[part]) - {'n'}, set(M.METRICS))
        self.assertEqual(self.global_out(site)[1]['metrics']['eval']['regions'],
                         {r: v['candidate'] for r, v in breakdown.items()})

    def test_current_models_score_is_reused_when_the_eval_set_is_unchanged(self):
        self.second_round_from_the_published_model()
        with open(os.path.join(self.base, '1', '1', '2', 'eval_report.json')) as f:
            breakdown = json.load(f)['region_breakdown']
        self.assertTrue(all(v['current'] for v in breakdown.values()),
                        'cached region scores of the current model are reused')


@skipUnless(HAS_TORCH, 'torch is not installed')
class RegistryPublishTest(GateInTaskTest):
    """SF-12: only a gate-approved global model goes to the registry."""

    def setUp(self):
        super().setUp()
        base = 'starfish.controller.tasks.babel_brain_fno.task.transfer.'
        self.latest = patch(base + 'latest_model',
                            return_value={'version': '250k-v0004'}).start()
        self.publish = patch(base + 'publish_model', return_value={
            'version': '250k-v0005', 'parent': '250k-v0004'}).start()
        self.addCleanup(patch.stopall)

    def test_accepted_model_is_published_with_its_parent_and_report(self):
        site = self.coordinator(max_rel_l2_increase=1e9,
                                max_peak_distance_increase_mm=1e9)
        self.assertTrue(self.aggregate(site, 1))
        self.publish.assert_called_once()
        args, kwargs = self.publish.call_args
        self.assertEqual(args[:5], (1, 1, 1, 250000,
                         'standin-250k-p7-b1-t1-r1'))
        self.assertEqual(kwargs['parent'], '250k-v0004')
        self.assertTrue(kwargs['eval_report']['accepted'])

    def test_rejected_model_is_not_published(self):
        site = self.coordinator()
        path = os.path.join(site._mids_dir(), '1-1-1-mid-artifacts')
        delta, meta = load_artifact(path)
        save_artifact(path, {k: (v + 50.0).astype(v.dtype)
                      for k, v in delta.items()}, meta)
        self.assertTrue(self.aggregate(site, 1))
        self.publish.assert_not_called()

    def test_ungated_model_is_not_published(self):
        site = self.trained_site(1)
        self.deliver(site, site)
        self.assertTrue(self.aggregate(site, 1))
        self.publish.assert_not_called()

    def test_publish_failure_does_not_fail_the_round(self):
        from starfish.controller.file.transfer import TransferFailed
        self.publish.side_effect = TransferFailed('router down')
        site = self.coordinator(max_rel_l2_increase=1e9,
                                max_peak_distance_increase_mm=1e9)
        self.assertTrue(self.aggregate(site, 1))
