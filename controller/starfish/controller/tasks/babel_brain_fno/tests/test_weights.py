"""Tests for BabelBrainFno weights, deltas and FedAvg, SF-04."""

from unittest import TestCase, skipUnless

import numpy as np

from starfish.controller.tasks.babel_brain_fno import weights as W

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def arrays(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return {'a': (scale * rng.standard_normal((3, 4))).astype(np.float32),
            'b': (scale * rng.standard_normal(5)).astype(np.float32)}


class FedAvgTest(TestCase):

    def test_matches_hand_computed_fedavg(self):
        base = arrays(0)
        deltas = [arrays(1, 0.1), arrays(2, 0.1), arrays(3, 0.1)]
        counts = [10, 30, 60]
        result = W.fedavg(base, list(zip(deltas, counts)))
        for name in base:
            expected = base[name].astype(np.float64) + (
                0.1 * deltas[0][name] + 0.3 * deltas[1][name] + 0.6 * deltas[2][name])
            np.testing.assert_allclose(result[name], expected, atol=1e-6)
            self.assertEqual(result[name].dtype, np.float32)

    def test_one_update_gives_that_clients_model(self):
        base, local = arrays(0), arrays(5)
        result = W.fedavg(base, [(W.delta(local, base), 7)])
        for name in base:
            np.testing.assert_allclose(result[name], local[name], atol=1e-6)

    def test_mismatched_update_is_rejected(self):
        base = arrays(0)
        bad = dict(arrays(1), b=np.zeros(6, dtype=np.float32))
        with self.assertRaisesRegex(W.WeightsError, 'tensor b'):
            W.fedavg(base, [(bad, 1)])
        with self.assertRaisesRegex(W.WeightsError, 'names'):
            W.fedavg(base, [({'a': base['a']}, 1)])

    def test_non_finite_update_is_rejected(self):
        base = arrays(0)
        bad = arrays(1)
        bad['a'][0, 0] = np.nan
        with self.assertRaisesRegex(W.WeightsError, 'non-finite'):
            W.fedavg(base, [(bad, 1)])

    def test_updates_without_samples_are_rejected(self):
        with self.assertRaises(W.WeightsError):
            W.fedavg(arrays(0), [(arrays(1), 0)])
        with self.assertRaises(W.WeightsError):
            W.fedavg(arrays(0), [])

    def test_digest_ignores_order_but_not_values(self):
        a = arrays(0)
        self.assertEqual(W.digest(a), W.digest(
            dict(reversed(list(a.items())))))
        b = {k: v.copy() for k, v in a.items()}
        b['a'][0, 0] += 1e-6
        self.assertNotEqual(W.digest(a), W.digest(b))


@skipUnless(HAS_TORCH, 'torch is not installed')
class StateConversionTest(TestCase):

    def test_complex_parameters_round_trip_as_real_arrays(self):
        state = {'w': torch.randn(
            2, 3, dtype=torch.cfloat), 'b': torch.randn(4)}
        arrs, complex_keys = W.state_to_arrays(state)
        self.assertEqual(complex_keys, ['w'])
        self.assertEqual(arrs['w'].shape, (2, 3, 2))
        self.assertEqual(arrs['w'].dtype, np.float32)
        back = W.arrays_to_state(arrs, complex_keys)
        self.assertTrue(torch.equal(back['w'], state['w']))
        self.assertTrue(torch.equal(back['b'], state['b']))
