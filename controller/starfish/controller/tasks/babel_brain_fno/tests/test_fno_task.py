"""Tests for the BabelBrainFno training and aggregation lifecycle, SF-04.

Runs the real task on the stand-in model and synthetic stores, with router
calls stubbed: uploads succeed, and a coordinator's download of the deltas
is a copy from each site's local folder.
"""

import glob
import os
import shutil
import tempfile
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from starfish.controller.file.artifact_io import load_artifact, save_artifact
from starfish.controller.tasks.babel_brain_fno import standin, training as T
from starfish.controller.tasks.babel_brain_fno import weights as W
from starfish.controller.tasks.babel_brain_fno.store import STORE_ENV
from starfish.controller.tasks.babel_brain_fno.synthetic import write_synthetic_store
from starfish.controller.tasks.babel_brain_fno.task import SEED_DIR_ENV, BabelBrainFno

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DATA_SOURCE = {'type': 'babelbrain_store', 'bucket_hz': 250000}


def make_run(run_id=1, role='coordinator', current_round=1, total_round=3, **config):
    cfg = {'total_round': total_round, 'current_round': current_round,
           'data_source': dict(DATA_SOURCE), 'min_samples': 4, 'local_epochs': 1,
           'batch_size': 2, 'lr': 1e-3, 'device': 'cpu', 'gate': {'enabled': False}}
    cfg.update(config)
    return {'id': run_id, 'project': 7, 'batch': 1, 'role': role, 'status': 'Standby',
            'cur_seq': 1, 'tasks': [{'seq': 1, 'model': 'BabelBrainFno', 'config': cfg}]}


@skipUnless(HAS_TORCH, 'torch is not installed')
class FnoTaskTestCase(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = os.path.join(self.tmp, 'local')
        self.store = os.path.join(self.tmp, 'store')
        write_synthetic_store(self.store, n_groups=3,
                              per_group=3, val_groups=1)
        for p in (patch('starfish.controller.file.file_utils.base_folder', self.base),
                  patch.dict(os.environ, {STORE_ENV: self.store}),
                  patch.object(BabelBrainFno, 'upload', return_value=True)):
            p.start()
            self.addCleanup(p.stop)

    def trained_site(self, run_id, **config):
        task = BabelBrainFno(make_run(run_id=run_id, **config))
        self.assertTrue(task.prepare_data())
        self.assertTrue(task.training())
        return task

    def deliver(self, coordinator, *sites):
        """What download_mid_artifacts does over the network: every delta into one folder."""
        folder = coordinator._mids_dir()
        os.makedirs(folder, exist_ok=True)
        for site in sites:
            shutil.copy(site._mid_path(), os.path.join(folder, '{}-1-{}-mid-artifacts'.format(
                site.run_id, site._round())))
        return folder

    def aggregate(self, coordinator, n_runs):
        with patch.object(BabelBrainFno, 'fetch_runs', return_value=[{}] * n_runs):
            return coordinator.do_aggregate()

    def global_out(self, coordinator):
        return load_artifact(coordinator._global_out_path())


class OneClientTest(FnoTaskTestCase):

    def test_one_client_one_round_global_equals_local_model(self):
        site = self.trained_site(1)
        base, _ = site.current_global()
        update, meta = load_artifact(site._mid_path())
        self.assertEqual(meta['kind'], 'delta')
        self.assertEqual(meta['n_samples'], 6)
        self.assertTrue(
            any(np.abs(v).max() > 0 for v in update.values()), 'training changed nothing')
        self.deliver(site, site)
        self.assertTrue(self.aggregate(site, 1))
        new_global, gmeta = self.global_out(site)
        local = W.apply_delta(base, update)
        for name in base:
            np.testing.assert_allclose(
                new_global[name], local[name], atol=1e-6)
        self.assertEqual(gmeta['kind'], 'global')
        self.assertEqual(gmeta['model_version'], 'standin-250k-p7-b1-t1-r1')
        self.assertTrue(gmeta['metrics']['accepted'])


class ThreeClientTest(FnoTaskTestCase):

    def test_aggregation_equals_hand_computed_fedavg(self):
        co = self.trained_site(1)
        base, _ = co.current_global()
        folder = self.deliver(co, co)
        # Two more sites' deltas, crafted with known values and sample counts
        _, meta = load_artifact(co._mid_path())
        counts = {1: meta['n_samples'], 2: 30, 3: 60}
        deltas = {1: load_artifact(co._mid_path())[0]}
        rng = np.random.default_rng(0)
        for run_id in (2, 3):
            deltas[run_id] = {k: (0.01 * rng.standard_normal(v.shape)).astype(v.dtype)
                              for k, v in base.items()}
            save_artifact(os.path.join(folder, '{}-1-1-mid-artifacts'.format(run_id)),
                          deltas[run_id], dict(meta, n_samples=counts[run_id],
                                               model_version='x+run{}'.format(run_id)))
        self.assertTrue(self.aggregate(co, 3))
        result, gmeta = self.global_out(co)
        total = sum(counts.values())
        for name in base:
            expected = base[name].astype(np.float64) + sum(
                deltas[r][name].astype(np.float64) * counts[r] / total for r in counts)
            np.testing.assert_allclose(result[name], expected, atol=1e-6)
        self.assertEqual(gmeta['n_samples'], total)


class RejectionTest(FnoTaskTestCase):

    def setUp(self):
        super().setUp()
        self.co = self.trained_site(1)
        self.folder = self.deliver(self.co, self.co)
        self.delta, self.meta = load_artifact(self.co._mid_path())

    def add_delta(self, run_id, tensors=None, **meta):
        save_artifact(os.path.join(self.folder, '{}-1-1-mid-artifacts'.format(run_id)),
                      tensors or self.delta, dict(self.meta, **meta))

    def test_delta_from_another_global_model_is_rejected(self):
        self.add_delta(2, base_digest='0' * 64)
        self.assertFalse(self.aggregate(self.co, 2))
        self.assertFalse(os.path.exists(self.co._global_out_path()))

    def test_delta_for_another_architecture_is_rejected(self):
        self.add_delta(2, arch_hash='different')
        self.assertFalse(self.aggregate(self.co, 2))

    def test_delta_for_another_round_is_rejected(self):
        self.add_delta(2, round=5)
        self.assertFalse(self.aggregate(self.co, 2))

    def test_missing_delta_fails_the_round(self):
        self.assertFalse(self.aggregate(self.co, 2))

    def test_non_finite_delta_is_rejected(self):
        bad = {k: v.copy() for k, v in self.delta.items()}
        next(iter(bad.values())).flat[0] = np.inf
        self.add_delta(2, tensors=bad)
        self.assertFalse(self.aggregate(self.co, 2))

    def test_gate_rejection_keeps_the_previous_global_model(self):
        base, base_meta = self.co.current_global()
        with patch.object(BabelBrainFno, 'accept_candidate', return_value=False):
            self.assertTrue(self.aggregate(self.co, 1))
        result, gmeta = self.global_out(self.co)
        self.assertEqual(gmeta['model_version'], base_meta['model_version'])
        self.assertFalse(gmeta['metrics']['accepted'])
        for name in base:
            np.testing.assert_array_equal(result[name], base[name])


class GlobalModelTest(FnoTaskTestCase):

    def test_seed_init_is_identical_on_every_site(self):
        a = BabelBrainFno(make_run(run_id=1))
        b = BabelBrainFno(make_run(run_id=2, role='participant'))
        self.assertTrue(a.prepare_data() and b.prepare_data())
        self.assertEqual(W.digest(a.current_global()[
                         0]), W.digest(b.current_global()[0]))

    def test_seed_model_from_the_sites_own_folder(self):
        seed_dir = os.path.join(self.tmp, 'seeds')
        torch.manual_seed(3)
        arrays, _ = W.state_to_arrays(standin.build_model(250000).state_dict())
        from safetensors.numpy import save_file
        os.makedirs(seed_dir)
        path = os.path.join(seed_dir, 'tayeb-250k-v1.safetensors')
        save_file(arrays, path)
        from starfish.controller.file.artifact_io import sha256_file
        with patch.dict(os.environ, {SEED_DIR_ENV: seed_dir}):
            task = BabelBrainFno(make_run(seed_model_version='tayeb-250k-v1',
                                          seed_model_sha256=sha256_file(path)))
            self.assertTrue(task.prepare_data())
            self.assertEqual(
                W.digest(task.current_global()[0]), W.digest(arrays))
            wrong = BabelBrainFno(make_run(seed_model_version='tayeb-250k-v1',
                                           seed_model_sha256='0' * 64))
            self.assertFalse(wrong.prepare_data())
            for bad in ('../seeds/tayeb-250k-v1', '/etc/passwd'):
                self.assertFalse(BabelBrainFno(
                    make_run(seed_model_version=bad)).prepare_data())

    def test_validate_fetches_and_checks_the_previous_global_model(self):
        co = self.trained_site(1)
        self.deliver(co, co)
        self.assertTrue(self.aggregate(co, 1))
        published = co._global_out_path()

        def fake_download(run_id, file_type, folder, task_seq=None, round_seq=None, all_runs=False):
            os.makedirs(folder, exist_ok=True)
            target = os.path.join(folder, '1-1-1-artifacts')
            shutil.copy(published, target)
            return [target]

        round2 = BabelBrainFno(
            make_run(run_id=2, role='participant', current_round=2))
        with patch('starfish.controller.tasks.babel_brain_fno.task.transfer.download_all',
                   side_effect=fake_download):
            self.assertTrue(round2.validate())
            self.assertTrue(round2.prepare_data())
            self.assertTrue(round2.training())
        _, meta = load_artifact(round2._mid_path())
        self.assertEqual(meta['base_version'], 'standin-250k-p7-b1-t1-r1')
        self.assertEqual(meta['round'], 2)

        tampered, tmeta = load_artifact(published)
        save_artifact(published, tampered, dict(tmeta, arch_hash='other'))
        again = BabelBrainFno(
            make_run(run_id=3, role='participant', current_round=2))
        with patch('starfish.controller.tasks.babel_brain_fno.task.transfer.download_all',
                   side_effect=fake_download):
            self.assertFalse(again.validate())


class TrainingDetailsTest(FnoTaskTestCase):

    def test_round_log_records_time_and_memory(self):
        site = self.trained_site(1)
        _, meta = load_artifact(site._mid_path())
        self.assertIn('peak_memory_mb', meta['metrics'])
        self.assertEqual(len(meta['metrics']['epoch_seconds']), 1)
        self.assertIn('val_rel_l2', meta['metrics'])
        logs = glob.glob(os.path.join(self.base, '1', '1', '1', 'logs.txt'))
        with open(logs[0]) as f:
            text = f.read()
        self.assertIn('s per epoch, peak memory', text)
        self.assertNotIn(self.tmp, text)

    def test_curriculum_changes_the_loss_weights_by_round(self):
        curriculum = [{'from_round': 1, 'h1_weight': 0.0, 'pde_weight': 0.0},
                      {'from_round': 3, 'h1_weight': 0.5, 'pde_weight': 0.0},
                      {'from_round': 5, 'h1_weight': 0.5, 'pde_weight': 0.1}]
        self.assertEqual(T.stage_weights(curriculum, 2), {
                         'h1_weight': 0.0, 'pde_weight': 0.0})
        self.assertEqual(T.stage_weights(curriculum, 4), {
                         'h1_weight': 0.5, 'pde_weight': 0.0})
        self.assertEqual(T.stage_weights(curriculum, 9), {
                         'h1_weight': 0.5, 'pde_weight': 0.1})
        site = self.trained_site(1, curriculum=[{'from_round': 1, 'h1_weight': 0.5,
                                                 'pde_weight': 0.1}])
        _, meta = load_artifact(site._mid_path())
        self.assertEqual(meta['metrics']['stage'], {
                         'h1_weight': 0.5, 'pde_weight': 0.1})


@skipUnless(HAS_TORCH, 'torch is not installed')
class StandinTest(TestCase):

    def sample(self):
        g = torch.Generator().manual_seed(0)
        shape = (2, 16, 16, 32)
        return {'ct_hu': torch.rand(shape, generator=g) * 1000,
                'water_field': torch.randn((2, 2) + shape[1:], generator=g),
                'skull_field': torch.randn((2, 2) + shape[1:], generator=g),
                'sos': torch.full(shape, 1500.0), 'attenuation': torch.zeros(shape),
                'brain_mask': torch.ones(shape), 'frequency_hz': torch.tensor([250000.0] * 2),
                'spacing_mm': torch.tensor([0.49] * 2)}

    def test_shapes_and_finite_loss_with_every_term(self):
        x, y, aux = standin.inputs(self.sample())
        self.assertEqual(tuple(x.shape), (2, 6, 16, 16, 32))
        model = standin.build_model(250000)
        pred = model(x)
        self.assertEqual(pred.shape, y.shape)
        total, parts = standin.loss(
            pred, y, aux, {'h1_weight': 0.5, 'pde_weight': 0.1})
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(parts), {'rel_l2', 'h1', 'pde'})

    def test_gradient_checkpointing_gives_the_same_output(self):
        x, _, _ = standin.inputs(self.sample())
        torch.manual_seed(0)
        plain = standin.build_model(250000)
        torch.manual_seed(0)
        checkpointed = standin.build_model(250000, grad_checkpointing=True)
        x.requires_grad_(True)
        torch.testing.assert_close(plain(x), checkpointed(x))

    def test_arch_hash_is_stable(self):
        self.assertEqual(len(standin.ARCH_HASH), 16)
        self.assertEqual(standin.ARCH_HASH, standin.ARCH_HASH.lower())
