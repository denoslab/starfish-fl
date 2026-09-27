"""Tests for SF-07 delta compression."""

import os
from unittest import TestCase, skipUnless

import numpy as np

from starfish.controller.file.artifact_io import load_artifact
from starfish.controller.tasks.babel_brain_fno import codec
from starfish.controller.tasks.babel_brain_fno.tests.test_fno_task import (
    HAS_TORCH, FnoTaskTestCase)


def delta(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    return {'a': rng.standard_normal((n,)).astype(np.float32),
            'b': rng.standard_normal((10, 20)).astype(np.float32)}


class CodecRoundTripTest(TestCase):

    def setUp(self):
        self.d = delta()

    def test_none_is_exact(self):
        tensors, meta, _ = codec.encode(self.d, {})
        out = codec.decode(tensors, meta, self.d)
        for name in self.d:
            np.testing.assert_array_equal(out[name], self.d[name])

    def test_fp16_error_bound_and_size(self):
        tensors, meta, _ = codec.encode(self.d, {'method': 'fp16'})
        out = codec.decode(tensors, meta, self.d)
        for name in self.d:
            np.testing.assert_allclose(
                out[name], self.d[name], rtol=1e-3, atol=1e-4)
        self.assertEqual(codec.nbytes(tensors), codec.nbytes(self.d) // 2)

    def test_int8_error_bound_and_size(self):
        tensors, meta, _ = codec.encode(self.d, {'method': 'int8'})
        out = codec.decode(tensors, meta, self.d)
        for name in self.d:
            scale = np.abs(self.d[name]).max() / 127
            self.assertLessEqual(
                np.abs(out[name] - self.d[name]).max(), scale / 2 + 1e-7)
        self.assertLess(codec.nbytes(tensors), codec.nbytes(self.d) / 3.9)

    def test_int8_all_zero_tensor(self):
        zeros = {'a': np.zeros(5, dtype=np.float32)}
        tensors, meta, _ = codec.encode(zeros, {'method': 'int8'})
        np.testing.assert_array_equal(codec.decode(
            tensors, meta, zeros)['a'], zeros['a'])

    def test_topk_keeps_the_requested_share(self):
        tensors, meta, _ = codec.encode(self.d, {'method': 'topk', 'k': 0.01})
        self.assertEqual(tensors['a::idx'].size, 10)
        self.assertEqual(tensors['b::idx'].size, 2)
        out = codec.decode(tensors, meta, self.d)
        kept = np.flatnonzero(out['a'])
        largest = np.argsort(-np.abs(self.d['a']))[:10]
        self.assertEqual(set(kept), set(largest))

    def test_topk_size_at_one_percent(self):
        big = {'w': np.random.default_rng(
            1).standard_normal(1_000_000).astype(np.float32)}
        tensors, _, _ = codec.encode(big, {'method': 'topk', 'k': 0.01})
        ratio = codec.nbytes(big) / codec.nbytes(tensors)
        self.assertGreater(ratio, 60)

    def test_error_feedback_sends_everything_over_time(self):
        """Sent parts plus the final residual add up to the sum of the full deltas."""
        rng = np.random.default_rng(3)
        deltas = [{'a': rng.standard_normal(
            500).astype(np.float32)} for _ in range(6)]
        residual, sent = None, np.zeros(500, dtype=np.float64)
        for d in deltas:
            tensors, meta, residual = codec.encode(
                d, {'method': 'topk', 'k': 0.05}, residual)
            sent += codec.decode(tensors, meta, d)['a']
        total = sum(d['a'].astype(np.float64) for d in deltas)
        np.testing.assert_allclose(sent + residual['a'], total, atol=1e-4)

    def test_missing_tensors_decode_as_zeros(self):
        tensors, meta, _ = codec.encode({'a': self.d['a']}, {'method': 'int8'})
        out = codec.decode(tensors, meta, self.d)
        np.testing.assert_array_equal(out['b'], np.zeros_like(self.d['b']))


class CodecRejectionTest(TestCase):

    def setUp(self):
        self.d = delta()

    def test_bad_configs(self):
        for cfg in ({'method': 'zip'}, {'method': 'topk', 'k': 0}, {'method': 'topk', 'k': 2}):
            with self.assertRaises(codec.CodecError):
                codec.config(cfg)

    def test_out_of_range_or_repeated_indices(self):
        tensors, meta, _ = codec.encode(self.d, {'method': 'topk', 'k': 0.01})
        bad = dict(tensors, **{'a::idx': tensors['a::idx'].copy()})
        bad['a::idx'][0] = 10 ** 6
        with self.assertRaisesRegex(codec.CodecError, 'out of range'):
            codec.decode(bad, meta, self.d)
        bad['a::idx'][0] = bad['a::idx'][1]
        with self.assertRaisesRegex(codec.CodecError, 'repeats'):
            codec.decode(bad, meta, self.d)

    def test_unknown_tensor_or_codec(self):
        with self.assertRaisesRegex(codec.CodecError, 'does not have'):
            codec.decode({'z': np.zeros(3, np.float32)},
                         {'method': 'none'}, self.d)
        with self.assertRaises(codec.CodecError):
            codec.decode({}, {'method': 'magic'}, self.d)

    def test_bad_scale_and_non_finite_values(self):
        tensors, meta, _ = codec.encode(self.d, {'method': 'int8'})
        with self.assertRaisesRegex(codec.CodecError, 'scale'):
            codec.decode(
                dict(tensors, **{'a::scale': np.array([np.inf], np.float32)}), meta, self.d)
        tensors, meta, _ = codec.encode(self.d, {'method': 'fp16'})
        tensors['a'][0] = np.float16(np.inf)
        # Non-finite values decode as they are; robust aggregation, SF-11, excludes the delta
        self.assertTrue(np.isinf(codec.decode(tensors, meta, self.d)['a'][0]))


@skipUnless(HAS_TORCH, 'torch is not installed')
class CompressionInTaskTest(FnoTaskTestCase):

    def run_round(self, **config):
        site = self.trained_site(1, **config)
        self.deliver(site, site)
        self.assertTrue(self.aggregate(site, 1))
        return site

    def test_each_method_aggregates(self):
        for method in ('fp16', 'int8', 'topk'):
            with self.subTest(method=method):
                site = self.run_round(compression={'method': method, 'k': 0.1})
                _, meta = load_artifact(site._mid_path())
                self.assertEqual(meta['codec']['method'], method)
                self.assertLess(meta['metrics']['delta_bytes'],
                                meta['metrics']['raw_bytes'])
                new, _ = self.global_out(site)
                base, _ = site.current_global()
                self.assertTrue(
                    any(np.abs(new[n] - base[n]).max() > 0 for n in base))

    def test_topk_keeps_a_local_residual(self):
        site = self.run_round(compression={'method': 'topk', 'k': 0.05})
        self.assertTrue(os.path.exists(site._residual_path()))
        residual, _ = load_artifact(site._residual_path())
        self.assertTrue(any(np.abs(v).max() > 0 for v in residual.values()))

    def test_partial_fine_tuning_sends_only_trainable_tensors(self):
        site = self.run_round(trainable=['project.'])
        tensors, meta = load_artifact(site._mid_path())
        self.assertTrue(tensors)
        self.assertTrue(all(name.startswith('project.') for name in tensors))
        new, _ = self.global_out(site)
        base, _ = site.current_global()
        for name in base:
            if not name.startswith('project.'):
                np.testing.assert_array_equal(new[name], base[name])

    def test_trainable_matching_nothing_is_refused(self):
        from starfish.controller.tasks.babel_brain_fno.task import BabelBrainFno
        from starfish.controller.tasks.babel_brain_fno.tests.test_fno_task import make_run
        task = BabelBrainFno(make_run(trainable=['no.such.layer']))
        self.assertTrue(task.prepare_data())
        self.assertFalse(task.training())
