"""BabelBrainFno: federated fine-tuning of tFUS-FNO on local BabelBrain samples.

Each round, every site trains on its own sample store, see ``store.py``,
starting from the current global model, and sends only the change in its
weights, the delta. The coordinator takes the sample-weighted mean of the
deltas, adds it to the global model, and publishes the result as the next
global model, if the evaluation gate accepts it. Round 1 starts from the
seed model. Artifacts are SF-01 safetensors files moved with SF-02
streamed transfer. Spec: babelbrain-docs/specs/starfish-work-items.md, SF-04.

No LLM calls in the BabelBrain path: agent hooks are never loaded, even if
the task config asks for them. Task logs are uploaded to the router, so
nothing here logs local paths.
"""

import glob
import os
import re
import shutil
import tempfile

from starfish.controller.file import file_utils, transfer
from starfish.controller.file.artifact_io import ArtifactError, load_artifact, save_artifact
from starfish.controller.tasks.abstract_task import AbstractTask
from starfish.controller.tasks.babel_brain_fno import weights as W
from starfish.controller.tasks.babel_brain_fno.store import SampleStore, StoreError
from starfish.controller.tasks.data_source import BABELBRAIN_STORE, validate_data_source

TASK = 'BabelBrainFno'
DEFAULT_MIN_SAMPLES = 20
DEFAULT_MODEL_PACKAGE = 'standin'
DEFAULT_SEED_MODEL = 'init:0'
HASH_CACHE_NAME = os.path.join('babelbrain_fl', 'sha256_cache.json')
SEED_DIR_ENV = 'BABELBRAIN_FL_SEED_DIR'
_SEED_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$')


class TaskError(Exception):
    """A problem that fails this site's step of the round, with a message for the log."""


def load_model_package(name):
    """The model code: the pinned ``tfus_fno`` package, or the stand-in until it exists."""
    if name == 'standin':
        from starfish.controller.tasks.babel_brain_fno import standin
        return standin
    if name == 'tfus_fno':
        import tfus_fno
        return tfus_fno
    raise TaskError('unknown model_package {!r}'.format(name))


class BabelBrainFno(AbstractTask):
    """
    Config Parameters
    -----------------
    data_source : dict, required
        ``{"type": "babelbrain_store", "bucket_hz": 250000}``. The store
        location comes from the site's ``BABELBRAIN_FL_STORE``.
    model_package : str, default "standin"
        ``standin`` or ``tfus_fno``.
    seed_model_version : str, default "init:0"
        ``init:<seed>`` starts every site from the same random init. Any
        other value names ``<value>.safetensors`` in the site's own
        ``BABELBRAIN_FL_SEED_DIR``; ``seed_model_sha256`` pins its hash.
    local_epochs, batch_size, grad_accum, lr, weight_decay, amp,
    grad_checkpointing, clip_norm, device, seed : training settings
    curriculum : list
        ``{"from_round", "h1_weight", "pde_weight"}`` stages; FL rounds
        replace the paper's epochs.
    min_samples : int, default 20
        Refuse to train with fewer local train samples than this.
    """

    agents_allowed = False

    def __init__(self, run):
        super().__init__(run)
        self.train_records = []
        self.val_records = []
        self._global = None

    # ── config and locations ────────────────────────────────────────────────

    def _config(self):
        return self.tasks[self.cur_seq - 1].get('config', {})

    def _round(self):
        return int(self.get_round())

    def bucket_hz(self):
        return int(self._config()['data_source']['bucket_hz'])

    def package_name(self):
        return self._config().get('model_package', DEFAULT_MODEL_PACKAGE)

    def package(self):
        return load_model_package(self.package_name())

    def global_version(self, task_round):
        return '{}-{}k-p{}-b{}-t{}-r{}'.format(
            self.package_name(), self.bucket_hz() // 1000, self.project_id, self.batch_id,
            self.cur_seq, task_round)

    def _start_path(self):
        """This round's starting global model, kept on disk for training and aggregation."""
        return file_utils.gen_url(self.run_id, self.cur_seq, self._round(), 'global-start')

    def _mid_path(self):
        return file_utils.gen_binary_mid_artifacts_url(self.run_id, self.cur_seq, self._round())

    def _global_out_path(self):
        return file_utils.gen_binary_artifacts_url(self.run_id, self.cur_seq, self._round())

    def _mids_dir(self):
        return os.path.join(file_utils.gen_all_mid_artifacts_url(self.project_id, self.batch_id),
                            'bbfno', str(self.cur_seq), str(self._round()))

    # ── metadata ────────────────────────────────────────────────────────────

    def _meta(self, kind, model_version, n_samples, metrics, complex_keys, **extra):
        meta = {
            'task': TASK, 'kind': kind, 'model_version': model_version,
            'n_samples': int(n_samples), 'round': self._round(), 'metrics': metrics,
            'arch_hash': self.package().ARCH_HASH, 'model_package': self.package_name(),
            'bucket_hz': self.bucket_hz(), 'complex_keys': list(complex_keys),
        }
        meta.update(extra)
        return meta

    def _check_meta(self, meta, kind):
        expected = {'task': TASK, 'kind': kind, 'arch_hash': self.package().ARCH_HASH,
                    'model_package': self.package_name(), 'bucket_hz': self.bucket_hz()}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise TaskError('{} artifact has {} {!r}, expected {!r}'.format(
                    kind, key, meta.get(key), value))

    # ── the global model ────────────────────────────────────────────────────

    def _fresh_arrays(self):
        model = self.package().build_model(self.bucket_hz())
        return W.state_to_arrays(model.state_dict())

    def _seed_global(self):
        version = self._config().get('seed_model_version', DEFAULT_SEED_MODEL)
        if version.startswith('init:'):
            import torch
            torch.manual_seed(int(version[len('init:'):]))
            arrays, complex_keys = self._fresh_arrays()
            return arrays, {'model_version': version, 'complex_keys': complex_keys}
        # A pretrained seed from this site's own folder, never a path from config
        if not _SEED_NAME.match(version):
            raise TaskError('seed_model_version must be a plain name')
        seed_dir = os.getenv(SEED_DIR_ENV)
        if not seed_dir:
            raise TaskError('{} is not set on this site'.format(SEED_DIR_ENV))
        try:
            arrays, meta = load_artifact(os.path.join(seed_dir, version + '.safetensors'),
                                         expected_sha256=self._config().get('seed_model_sha256'),
                                         strict=False)
        except ArtifactError as e:
            raise TaskError('seed model {} unusable: {}'.format(
                version, str(e).split(': ', 1)[-1][:200]))
        reference, complex_keys = self._fresh_arrays()
        W.check_compatible(reference, arrays, 'seed model')
        return arrays, {'model_version': version, 'complex_keys': meta.get('complex_keys', complex_keys)}

    def _set_global(self, arrays, meta):
        self._global = (arrays, meta)
        save_artifact(self._start_path(), arrays, self._meta(
            'global', meta['model_version'], meta.get(
                'n_samples', 0), meta.get('metrics', {}),
            meta.get('complex_keys', [])))

    def current_global(self):
        """``(arrays, meta)`` of the global model this round starts from."""
        if self._global is None:
            arrays, meta = load_artifact(self._start_path())
            self._global = (arrays, meta)
        return self._global

    def _load_downloaded_global(self):
        """The previous global model that validate() fetched, read back from disk."""
        seq_no, round_no = self.get_previous_seq_and_round()
        folder = file_utils.downloaded_artifacts_url(
            self.run_id, seq_no, round_no)
        paths = glob.glob(os.path.join(folder, '*-artifacts'))
        if len(paths) != 1:
            raise TaskError('previous global model was not validated')
        arrays, meta = load_artifact(paths[0])
        self._check_meta(meta, 'global')
        return arrays, meta

    def validate(self) -> bool:
        """Round 1 passes; later rounds fetch and check the previous global model."""
        if self.is_first_round():
            return True
        seq_no, round_no = self.get_previous_seq_and_round()
        folder = file_utils.downloaded_artifacts_url(
            self.run_id, seq_no, round_no)
        try:
            shutil.rmtree(folder, ignore_errors=True)
            paths = transfer.download_all(
                self.run_id, 'artifacts', folder, seq_no, round_no)
            if len(paths) != 1:
                raise TaskError('expected one global model for round {}, found {}'.format(
                    round_no, len(paths)))
            arrays, meta = load_artifact(paths[0])
            self._check_meta(meta, 'global')
            W.check_compatible(self._fresh_arrays()[0], arrays, 'global model')
            self._global = (arrays, meta)
        except (transfer.TransferFailed, ArtifactError, TaskError, W.WeightsError) as e:
            self.logger.error('Previous global model rejected: {}'.format(e))
            return False
        self.logger.info('Round {} starts from global model {}'.format(
            self._round(), meta['model_version']))
        return True

    # ── preparing ───────────────────────────────────────────────────────────

    def open_store(self):
        """Open this site's store. The location never comes from task config."""
        cache_path = os.path.join(file_utils.base_folder, HASH_CACHE_NAME)
        return SampleStore.from_env(cache_path=cache_path, logger=self.logger)

    def _prepare_samples(self) -> bool:
        config = self._config()
        data_source = config.get('data_source')
        error = validate_data_source(data_source)
        if error is None and data_source.get('type') != BABELBRAIN_STORE:
            error = 'BabelBrainFno needs a {} data_source'.format(
                BABELBRAIN_STORE)
        if error:
            self.logger.error(error)
            return False
        bucket_hz = data_source['bucket_hz']
        min_samples = int(config.get('min_samples', DEFAULT_MIN_SAMPLES))
        try:
            store = self.open_store()
            self.train_records = store.samples(bucket_hz, 'train')
            self.val_records = store.samples(bucket_hz, 'val')
        except StoreError as e:
            self.logger.error('Sample store unavailable: {}'.format(e))
            return False
        self.logger.info('Sample store at {} Hz: {} train, {} val'.format(
            bucket_hz, len(self.train_records), len(self.val_records)))
        if len(self.train_records) < min_samples:
            self.logger.error('Only {} train samples, fewer than min_samples {}'.format(
                len(self.train_records), min_samples))
            return False
        return True

    def prepare_data(self) -> bool:
        if not self._prepare_samples():
            return False
        try:
            if self.is_first_round():
                arrays, meta = self._seed_global()
                self.logger.info(
                    'Round 1 starts from seed model {}'.format(meta['model_version']))
            else:
                if self._global is None:
                    self._global = self._load_downloaded_global()
                arrays, meta = self._global
            self._set_global(arrays, meta)
        except (TaskError, ArtifactError, W.WeightsError, ImportError) as e:
            self.logger.error('Global model unavailable: {}'.format(e))
            return False
        return True

    # ── running ─────────────────────────────────────────────────────────────

    def training(self) -> bool:
        try:
            return self._train()
        except Exception as e:
            self.logger.error('Training failed: {}: {}'.format(
                e.__class__.__name__, e))
            return False

    def _train(self):
        import torch
        from torch.utils.data import DataLoader

        from starfish.controller.tasks.babel_brain_fno import training as T
        from starfish.controller.tasks.babel_brain_fno.sample_io import torch_dataset

        config, pkg, task_round = self._config(), self.package(), self._round()
        base, base_meta = self.current_global()
        device = T.pick_device(config.get('device', 'auto'))
        seed = int(config.get('seed', 0)) + 1000 * \
            task_round + int(self.run_id)
        torch.manual_seed(seed)

        model = pkg.build_model(self.bucket_hz(),
                                grad_checkpointing=bool(config.get('grad_checkpointing', False)))
        model.load_state_dict(W.arrays_to_state(
            base, base_meta.get('complex_keys', [])))
        model.to(device)
        batch_size = int(config.get('batch_size', 1))
        train_loader = DataLoader(torch_dataset(self.train_records), batch_size=batch_size,
                                  shuffle=True, generator=torch.Generator().manual_seed(seed))
        stage = T.stage_weights(config.get('curriculum'), task_round)
        metrics = T.train_round(model, train_loader, pkg,
                                device, config, stage, self.logger)
        if self.val_records:
            val_loader = DataLoader(torch_dataset(
                self.val_records), batch_size=batch_size)
            metrics.update(T.evaluate(model, val_loader, pkg, device))
        metrics['stage'] = stage

        local, complex_keys = W.state_to_arrays(model.state_dict())
        update = W.delta(local, base)
        save_artifact(self._mid_path(), update, self._meta(
            'delta', '{}+run{}'.format(
                base_meta['model_version'], self.run_id),
            len(self.train_records), metrics, complex_keys,
            base_version=base_meta['model_version'], base_digest=W.digest(base)))
        self.logger.info(
            'Round {}: {} train samples, loss {:.5f}, val rel l2 {}, {} s per epoch, '
            'peak memory {:.0f} MB on {}'.format(
                task_round, len(self.train_records), metrics['train_loss'],
                '{:.4f}'.format(metrics['val_rel_l2']
                                ) if 'val_rel_l2' in metrics else 'n/a',
                metrics['epoch_seconds'], metrics['peak_memory_mb'], metrics['device']))
        return True

    # ── transfer ────────────────────────────────────────────────────────────

    def upload(self, is_artifact: bool) -> bool:
        """Stream this round's files to the router through SF-02."""
        task_round = self._round()
        try:
            if is_artifact:
                transfer.upload_file(self._global_out_path(), self.run_id, self.cur_seq,
                                     task_round, 'artifacts', name='artifacts')
                return True
            if os.path.exists(self._mid_path()):
                transfer.upload_file(self._mid_path(), self.run_id, self.cur_seq, task_round,
                                     'mid_artifacts', name='mid-artifacts')
            self._upload_logs(task_round)
            return True
        except (transfer.TransferFailed, OSError) as e:
            self.logger.error('Upload failed: {}'.format(e))
            return False

    def _upload_logs(self, task_round):
        logs = file_utils.gen_logs_url(self.run_id, self.cur_seq, task_round)
        if not logs or not os.path.exists(logs):
            return
        # The log grows while it uploads, so send a snapshot
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = os.path.join(tmp, 'logs.txt')
            shutil.copyfile(logs, snapshot)
            transfer.upload_file(snapshot, self.run_id, self.cur_seq, task_round, 'logs',
                                 name='logs.txt')

    def download_mid_artifacts(self) -> bool:
        """Coordinator: fetch every site's delta for this round."""
        folder = self._mids_dir()
        shutil.rmtree(folder, ignore_errors=True)
        try:
            paths = transfer.download_all(self.run_id, 'mid_artifacts', folder, self.cur_seq,
                                          self._round(), all_runs=True)
        except transfer.TransferFailed as e:
            self.logger.error('Downloading deltas failed: {}'.format(e))
            return False
        self.logger.info('Downloaded {} deltas'.format(len(paths)))
        return bool(paths)

    # ── aggregating ─────────────────────────────────────────────────────────

    def accept_candidate(self, candidate, metrics) -> bool:
        """Evaluation gate. SF-05 replaces this with a check on held-out subjects."""
        return True

    def do_aggregate(self) -> bool:
        try:
            return self._aggregate()
        except (TaskError, ArtifactError, W.WeightsError) as e:
            self.logger.error('Aggregation failed: {}'.format(e))
            return False

    def _aggregate(self):
        task_round = self._round()
        base, base_meta = self.current_global()
        base_digest = W.digest(base)
        paths = sorted(glob.glob(os.path.join(
            self._mids_dir(), '*-mid-artifacts')))
        runs = self.fetch_runs() or []
        if runs and len(paths) != len(runs):
            raise TaskError('expected {} deltas, found {}'.format(
                len(runs), len(paths)))
        if not paths:
            raise TaskError('no deltas to aggregate')

        updates, sites = [], []
        for path in paths:
            tensors, meta = load_artifact(path)
            self._check_meta(meta, 'delta')
            if meta['round'] != task_round:
                raise TaskError('a delta is for round {}, not {}'.format(
                    meta['round'], task_round))
            if meta.get('base_digest') != base_digest:
                raise TaskError(
                    'a delta was trained from a different global model')
            updates.append((tensors, meta['n_samples']))
            sites.append(meta)
        candidate = W.fedavg(base, updates)

        metrics = {'sites': len(sites), 'train_samples': sum(
            m['n_samples'] for m in sites)}
        val = [(m['metrics'].get('val_rel_l2'), m['metrics'].get('val_ssim'),
                m['metrics'].get('val_samples', 0)) for m in sites]
        val = [v for v in val if v[0] is not None and v[2]]
        if val:
            n = sum(v[2] for v in val)
            metrics['site_val_rel_l2'] = sum(v[0] * v[2] for v in val) / n
            metrics['site_val_ssim'] = sum(v[1] * v[2] for v in val) / n

        accepted = self.accept_candidate(candidate, metrics)
        metrics['accepted'] = bool(accepted)
        if accepted:
            arrays, version = candidate, self.global_version(task_round)
        else:
            # The previous global model stays current
            arrays, version = base, base_meta['model_version']
        save_artifact(self._global_out_path(), arrays, self._meta(
            'global', version, metrics['train_samples'], metrics,
            base_meta.get('complex_keys', []), base_version=base_meta['model_version']))
        self.logger.info('Round {}: aggregated {} deltas, {} samples; global model {} {}'.format(
            task_round, len(sites), metrics['train_samples'], version,
            'accepted' if accepted else 'rejected by the gate, previous model kept'))
        return self.upload(True)
