"""Tests for SF-11 robust aggregation and FedProx."""

import os
from unittest import TestCase, skipUnless

import numpy as np

from starfish.controller.file.artifact_io import load_artifact, save_artifact
from starfish.controller.tasks.babel_brain_fno import weights as W
from starfish.controller.tasks.babel_brain_fno.tests.test_fno_task import (
    HAS_TORCH, FnoTaskTestCase)


def update(value, n=100):
    return {'a': np.full(n, value, dtype=np.float32)}


class ScreenTest(TestCase):

    def test_non_finite_delta_is_excluded(self):
        bad = update(1.0)
        bad['a'][3] = np.nan
        kept, excluded = W.screen([(update(1.0), 5), (bad, 5)])
        self.assertEqual([k[0] for k in kept], [0])
        self.assertEqual(excluded, [(1, 'non-finite values')])

    def test_scaled_delta_is_screened_out(self):
        updates = [(update(1.0), 5), (update(1.1), 5),
                   (update(100.0), 5), (update(0.9), 5)]
        kept, excluded = W.screen(updates, screen_factor=5)
        self.assertEqual([k[0] for k in kept], [0, 1, 3])
        self.assertEqual(excluded[0][0], 2)
        self.assertIn('times the median', excluded[0][1])

    def test_screening_needs_three_deltas(self):
        kept, excluded = W.screen(
            [(update(1.0), 5), (update(100.0), 5)], screen_factor=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(excluded, [])

    def test_clip_scales_long_deltas_to_the_norm(self):
        kept, _ = W.screen(
            [(update(1.0), 5), (update(0.01), 5)], clip_norm=2.0)
        self.assertAlmostEqual(W.update_norm(kept[0][1]), 2.0, places=5)
        np.testing.assert_array_equal(kept[1][1]['a'], update(0.01)['a'])

    def test_no_options_keeps_everything_unchanged(self):
        updates = [(update(1.0), 5), (update(3.0), 7)]
        kept, excluded = W.screen(updates)
        self.assertEqual(excluded, [])
        for (i, u, n), (orig, n0) in zip(kept, updates):
            np.testing.assert_array_equal(u['a'], orig['a'])
            self.assertEqual(n, n0)


@skipUnless(HAS_TORCH, 'torch is not installed')
class RobustInTaskTest(FnoTaskTestCase):

    def setUp(self):
        super().setUp()
        self.co = self.trained_site(1, aggregation={'screen_factor': 5})
        self.folder = self.deliver(self.co, self.co)
        self.delta, self.meta = load_artifact(self.co._mid_path())

    def add(self, run_id, factor=1.0, nan=False):
        tensors = {k: (v * factor).astype(v.dtype)
                   for k, v in self.delta.items()}
        if nan:
            next(iter(tensors.values())).flat[0] = np.nan
        save_artifact(os.path.join(self.folder, '{}-1-1-mid-artifacts'.format(run_id)),
                      tensors, dict(self.meta, model_version='x+run{}'.format(run_id)))

    def test_scaled_delta_is_excluded_and_logged(self):
        self.add(2, 1.05)
        self.add(3, 1000.0)
        self.assertTrue(self.aggregate(self.co, 3))
        result, gmeta = self.global_out(self.co)
        self.assertEqual(gmeta['metrics']['sites'], 2)
        self.assertEqual([e['run']
                         for e in gmeta['metrics']['excluded']], ['3'])
        base, _ = self.co.current_global()
        expected = W.fedavg(base, [(self.delta, self.meta['n_samples']),
                                   ({k: (v * 1.05).astype(v.dtype) for k, v in self.delta.items()},
                                    self.meta['n_samples'])])
        for name in base:
            np.testing.assert_allclose(result[name], expected[name], atol=1e-6)
        with open(os.path.join(self.base, '1', '1', '1', 'logs.txt')) as f:
            self.assertIn('Excluded the delta of run 3', f.read())

    def test_nan_delta_is_excluded_not_fatal(self):
        self.add(2, nan=True)
        self.assertTrue(self.aggregate(self.co, 2))
        _, gmeta = self.global_out(self.co)
        self.assertEqual(gmeta['metrics']['excluded'], [
                         {'run': '2', 'reason': 'non-finite values'}])

    def test_round_fails_when_every_delta_is_excluded(self):
        path = os.path.join(self.folder, '1-1-1-mid-artifacts')
        tensors = {k: v.copy() for k, v in self.delta.items()}
        next(iter(tensors.values())).flat[0] = np.inf
        save_artifact(path, tensors, self.meta)
        self.assertFalse(self.aggregate(self.co, 1))


@skipUnless(HAS_TORCH, 'torch is not installed')
class FedProxTest(FnoTaskTestCase):

    def delta_of(self, run_id, **config):
        site = self.trained_site(run_id, seed=5, **config)
        tensors, meta = load_artifact(site._mid_path())
        return tensors, meta

    def test_mu_zero_reproduces_fedavg_exactly(self):
        plain, _ = self.delta_of(1)
        zero, meta = self.delta_of(1, fedprox_mu=0.0)
        self.assertEqual(meta['metrics']['fedprox_mu'], 0.0)
        for name in plain:
            np.testing.assert_array_equal(plain[name], zero[name])

    def test_positive_mu_keeps_the_local_model_closer(self):
        plain, _ = self.delta_of(1, lr=1e-2)
        prox, _ = self.delta_of(1, lr=1e-2, fedprox_mu=100.0)
        self.assertLess(W.update_norm(prox), W.update_norm(plain))
