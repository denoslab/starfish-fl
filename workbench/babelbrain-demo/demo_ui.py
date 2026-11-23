"""
Live page for the BabelBrain FL demo, the coordinator's console at site A.

    python demo_ui.py [--port 8090]

Runs in the controller image through ``make babelbrain-demo-ui``, next to
the demo stack, and serves http://localhost:8090. Like ``run_demo.py`` it
talks only to the router's API from inside the compose network. The
browser thus needs neither the router password nor CORS. The page starts a
run and follows it with each site's state per round, the gate's verdict
and scores, the sites' training losses, field slices from each global
model on site A's held-out eval samples, and the registry, where the
coordinator approves a pending model.

The field slices read site A's eval store, mounted read-only at
``BABELBRAIN_FL_EVAL_STORE``, with the model, training, metric and store
code of the task in the image. Importing the ``starfish`` package would set
up the controller's Celery app. This module therefore loads those files by
path.

For comparison the page also trains two baselines with the same model,
settings and rounds, from the three site stores under ``DEMO_SITE_STORES``.
The centralized baseline trains on the pooled train samples. The local-only
baseline trains one model per site on its own samples without aggregation,
and its scores are the simple average over the sites. Each model is scored
after each round the way the gate scores a candidate. With all three stores
on one machine, only this demo can train them.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_demo as D  # noqa: E402

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_ui.html')
CODE_DIR = os.environ.get('STARFISH_CONTROLLER_CODE', '/app/starfish/controller')
EVAL_STORE = os.environ.get('BABELBRAIN_FL_EVAL_STORE', '/babelbrain-eval-store')
# The folder holding site-a, site-b and site-c, pooled only for the centralized baseline
SITE_STORES = os.environ.get('DEMO_SITE_STORES', '/demo-stores')
PROJECT_PREFIX = 'BabelBrain FL demo'
POLL_S = 2.0
TERMINAL = ('Success', 'Failed')
LOCAL_HOSTS = ('localhost', '127.0.0.1', '::1')
MAX_BODY = 4096
METRICS = ('rel_l2_pct', 'peak_distance_mm', 'ssim', 'psnr_db', 'peak_amplitude_error_pct',
           'focal_centroid_distance_mm')
# Mirrors DEFAULT_GATE in tasks/babel_brain_fno/gate.py, for the page before the first report
GATE_DEFAULTS = {'enabled': True, 'max_rel_l2_increase': 0.0,
                 'max_peak_distance_increase_mm': None}
LOSS = re.compile(r'Epoch \d+ of \d+: loss ([\d.]+)')
ERROR_LINE = re.compile(r'^\S+ \S+ \[ERROR\] (.+)$', re.M)
SAMPLE_DATASETS = ('ct_hu', 'water_field', 'skull_field', 'sos', 'attenuation', 'brain_mask')


def clean(value):
    """A JSON-safe copy, with every non-finite float as None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def router_message(response):
    """The router's own words for a failed call, shortened."""
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, dict):
        body = body.get('detail') or body.get('error') or json.dumps(body)
    return 'the router answered {}, {}'.format(response.status_code, str(body).strip()[:200])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── what the router's files say ─────────────────────────────────────────────

def means(scores):
    """The mean of each metric in a gate summary, or None without one."""
    if not scores:
        return None
    return {name: (scores.get(name) or {}).get('mean') for name in METRICS}


def gate_summary(report):
    """The parts of a round's eval_report.json that the page shows."""
    regions = {}
    for region, row in (report.get('region_breakdown') or {}).items():
        row = row or {}
        either = row.get('candidate') or row.get('current') or {}
        regions[region] = {'n': either.get('n'), 'current': means(row.get('current')),
                           'candidate': means(row.get('candidate'))}
    return {'round': report.get('round'), 'accepted': bool(report.get('accepted')),
            'store_digest': report.get('store_digest'),
            'reasons': report.get('reasons') or [], 'eval_samples': report.get('eval_samples'),
            'current_model': report.get('current_model'), 'margins': report.get('margins') or {},
            'current': means(report.get('current')), 'candidate': means(report.get('candidate')),
            'regions': regions or None}


def round_info(files, rnd):
    """One site's round as its log tells it, plus the gate's report at site A."""
    log = files.get('logs.txt', '')
    trained = re.search(r'Round {}: (\d+) train samples'.format(rnd), log)
    start = re.search(r'Round {} starts from (?:global|seed) model (\S+)'.format(rnd), log)
    errors = ERROR_LINE.findall(log)
    info = {'losses': [float(x) for x in LOSS.findall(log)],
            'train_samples': int(trained.group(1)) if trained else None,
            'start_model': start.group(1) if start else None,
            'error': errors[-1].strip()[:300] if errors else None,
            'gate': None}
    if 'eval_report.json' in files:
        try:
            info['gate'] = gate_summary(json.loads(files['eval_report.json']))
        except (ValueError, AttributeError):
            pass
    return info


# ── narration ───────────────────────────────────────────────────────────────

def who(keys):
    keys = sorted(keys)
    if len(keys) == 1:
        return 'Site ' + keys[0]
    return 'Sites {} and {}'.format(', '.join(keys[:-1]), keys[-1])


def phrase(status, rnd, keys):
    """One line for sites that reached the same state in the same poll, or None for a quiet step."""
    many = len(keys) > 1
    s, its = ('', 'their') if many else ('s', 'its')
    subject = who(keys)
    if status == 'Standby':
        return '{} wait{} to start round {}'.format(subject, s, rnd)
    if status == 'Preparing':
        return '{} check{} {} own sample store for round {}'.format(subject, s, its, rnd)
    if status == 'Running':
        return '{} train{} round {} on {} own samples'.format(subject, s, rnd, its)
    if status == 'Pending Success':
        return '{} finished training round {} and sent {} weight update{}'.format(
            subject, rnd, its, 's' if many else '')
    if status == 'Pending Aggregating':
        return 'Site A collects the weight updates of round {}'.format(rnd) if 'A' in keys else None
    if status == 'Aggregating':
        return ('Site A averages the updates of round {} and scores the result on its held-out '
                'samples'.format(rnd)) if 'A' in keys else None
    if status == 'Success':
        return '{} finished the run'.format(subject)
    if status == 'Failed':
        return '{} failed in round {}'.format(subject, rnd)
    if status == 'Pending Failed':
        return '{} report{} a failure in round {}'.format(subject, s, rnd)
    if status == 'Sitting Out':
        return '{} sit{} out round {}'.format(subject, s, rnd)
    return '{} in state {}, round {}'.format(subject, status, rnd)


def gate_phrase(rnd, gate):
    old = (gate.get('current') or {}).get('rel_l2_pct')
    new = (gate.get('candidate') or {}).get('rel_l2_pct')
    scores = ''
    if old is not None and new is not None:
        scores = ', relative l2 error {:.1f} % against {:.1f} % for the current model'.format(new, old)
    if gate['accepted']:
        return 'Round {} gate accepted the new model{}'.format(rnd, scores)
    return 'Round {} gate rejected the new model{}. The previous model stays'.format(rnd, scores)


# ── the task's code and site A's eval samples ───────────────────────────────

class Lab:
    """The task's own model, training, metric and store code, and site A's eval samples.

    Loaded once and shared by the field slices and the centralized baseline.
    """

    MODULES = ('standin', 'weights', 'metrics', 'store', 'training')

    def __init__(self, eval_root):
        self.eval_root = eval_root
        self.lock = threading.Lock()
        # torch's random state is global, so every model build takes this lock
        self.rng_lock = threading.Lock()
        self.mods = self.torch = None
        self.evals, self.digests = {}, {}

    def load(self, package):
        """The task's modules, with torch imported, or a RuntimeError saying why not."""
        if package != 'standin':
            raise RuntimeError('the page runs the stand-in model only, '
                               'this project uses {}'.format(package))
        with self.lock:
            if self.mods is None:
                folder = os.path.join(CODE_DIR, 'tasks', 'babel_brain_fno')
                mods = {name: load_module('demo_ui_' + name, os.path.join(folder, name + '.py'))
                        for name in self.MODULES}
                mods['artifact_io'] = load_module(
                    'demo_ui_artifact_io', os.path.join(CODE_DIR, 'file', 'artifact_io.py'))
                import torch
                torch.set_num_threads(2)
                self.mods, self.torch = mods, torch
        return self.mods

    def build(self, bucket, seed=None):
        """A new stand-in model, built after ``torch.manual_seed(seed)`` when a seed is given."""
        with self.rng_lock:
            if seed is not None:
                self.torch.manual_seed(seed)
            return self.mods['standin'].build_model(bucket)

    def records(self, root, bucket, split=None):
        """A store's usable samples, checked the way the sites check them."""
        store = self.mods['store']
        if not os.path.isdir(root):
            raise RuntimeError('no store at {}'.format(root))
        try:
            return store.SampleStore(root).samples(bucket, split)
        except store.StoreError as e:
            raise RuntimeError('the store at {} is unusable, {}'.format(root, e))

    def read(self, record):
        """One sample as the task's dataset yields it, with its arrays and region."""
        import h5py
        import numpy as np
        with h5py.File(record.path, 'r') as f:
            arrays = {name: f[name][()] for name in SAMPLE_DATASETS}
            spacing = float(f.attrs['spacing_mm'])
            frequency = float(f.attrs['frequency_hz'])
        tensors = {k: self.torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
        tensors.update(frequency_hz=frequency, spacing_mm=spacing)
        return {'region': record.region or 'unlabelled', 'arrays': arrays, 'tensors': tensors,
                'spacing_mm': spacing, 'scale': float(np.abs(arrays['water_field']).max())}

    def eval_samples(self, bucket):
        """Site A's eval samples at a bucket, in the store's order, which the gate uses too."""
        with self.lock:
            cached = self.evals.get(bucket)
        if cached is not None:
            return cached
        records = self.records(self.eval_root, bucket)
        if not records:
            raise RuntimeError('the eval store has no samples at {} Hz'.format(bucket))
        samples = [self.read(r) for r in records]
        # The digest the gate writes into its report, which shows whether a run saw these samples
        digest = hashlib.sha256()
        for r in sorted(records, key=lambda r: r.sample_id):
            digest.update('{}:{}\n'.format(r.sample_id, r.sha256).encode())
        with self.lock:
            self.digests.setdefault(bucket, digest.hexdigest())
            return self.evals.setdefault(bucket, samples)

    def score(self, model, samples):
        """Predictions and per-sample metrics of a model, computed the way the gate does."""
        standin, metrics = self.mods['standin'], self.mods['metrics']
        model.eval()
        rows, preds = [], []
        with self.torch.no_grad():
            for s in samples:
                x, y, aux = standin.inputs(s['tensors'])
                pred = model(x)[0].float().numpy()
                rows.append(metrics.sample_metrics(pred, y[0].numpy(), s['arrays']['brain_mask'],
                                                   s['spacing_mm']))
                preds.append(pred)
        return rows, preds


# ── field slices on site A's held-out samples ───────────────────────────────

class Fields:
    """Field slices and per-sample scores of each model on site A's eval samples.

    A background thread computes each model once and keeps the result in
    memory. The page shows one sample per target region.
    """

    def __init__(self, lab):
        self.lab = lab
        self.baseline = None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.wanted, self.bucket, self.package = [], None, 'standin'
        self.done, self.failed = {}, {}
        self.samples, self.shown = None, []
        self.error = None
        self.rev = 0
        threading.Thread(target=self._work, name='fields', daemon=True).start()

    def want(self, wanted, bucket, package):
        """Ask for these models, each a dict with its version and the source of its weights."""
        with self.lock:
            if (wanted, bucket, package) == (self.wanted, self.bucket, self.package):
                return
            if bucket != self.bucket:
                self.samples, self.shown, self.error = None, [], None
            self.wanted, self.bucket, self.package = wanted, bucket, package
        self.wake.set()

    def status(self):
        with self.lock:
            ready = [w['version'] for w in self.wanted if w['version'] in self.done]
            computing = [w['version'] for w in self.wanted
                         if w['version'] not in self.done and w['version'] not in self.failed]
            failed = {w['version']: self.failed[w['version']] for w in self.wanted
                      if w['version'] in self.failed}
            return {'rev': self.rev, 'error': self.error, 'ready': ready,
                    'computing': [] if self.error else computing, 'failed': failed,
                    'n_eval': len(self.samples or []),
                    'eval_digest': self.lab.digests.get(self.bucket)}

    def payload(self):
        with self.lock:
            return {'rev': self.rev, 'error': self.error, 'n_eval': len(self.samples or []),
                    'samples': self.shown,
                    'models': {w['version']: self.done[w['version']] for w in self.wanted
                               if w['version'] in self.done}}

    def _work(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            while True:
                with self.lock:
                    todo = [w for w in self.wanted
                            if w['version'] not in self.done and w['version'] not in self.failed]
                    bucket, package = self.bucket, self.package
                if not todo:
                    break
                try:
                    self._prepare(bucket, package)
                except Exception as e:
                    with self.lock:
                        self.error = str(e)[:300]
                        self.rev += 1
                    print('Field slices unavailable: {}'.format(e), flush=True)
                    break
                item = todo[0]
                try:
                    result = self._compute(item, bucket)
                    with self.lock:
                        self.done[item['version']] = result
                        self.rev += 1
                except Exception as e:
                    traceback.print_exc()
                    with self.lock:
                        self.failed[item['version']] = '{}: {}'.format(
                            e.__class__.__name__, e)[:300]
                        self.rev += 1

    def _prepare(self, bucket, package):
        """The eval samples and the slices the page shows of them, once per bucket."""
        self.lab.load(package)
        if self.samples is not None:
            return
        import numpy as np
        metrics, regions = self.lab.mods['metrics'], self.lab.mods['store'].REGIONS
        samples = self.lab.eval_samples(bucket)
        # One sample per target region, in the order of the sample contract
        order = list(regions) + ['unlabelled']
        firsts = {}
        for index, sample in enumerate(samples):
            firsts.setdefault(sample['region'], index)
        shown = []
        for region in sorted(firsts, key=lambda r: order.index(r) if r in order else len(order)):
            index = firsts[region]
            s = samples[index]
            truth = metrics.amplitude(s['arrays']['skull_field'])
            mask = s['arrays']['brain_mask']
            masked = np.where(mask > 0, truth, -np.inf)
            y = int(np.unravel_index(int(np.argmax(masked)), truth.shape)[1])
            shown.append({
                'index': index, 'region': s['region'], 'y': y, 'shape': list(truth.shape),
                'spacing_mm': s['spacing_mm'],
                'ct': np.take(s['arrays']['ct_hu'], y, 1).round(0).astype(int).tolist(),
                'water': np.take(metrics.amplitude(s['arrays']['water_field']), y, 1)
                .round(0).astype(int).tolist(),
                'skull': np.take(truth, y, 1).round(0).astype(int).tolist(),
                'mask': np.take(mask, y, 1).astype(int).tolist()})
        with self.lock:
            self.samples, self.shown, self.error = samples, shown, None
            self.rev += 1

    def _model(self, item, bucket):
        mods = self.lab.mods
        if 'seed' in item:
            # The round 1 start, built the way the task builds init:<seed>
            return self.lab.build(bucket, item['seed'])
        if 'baseline' in item:
            return self.baseline.model(item['baseline'], bucket, item.get('site'))
        response = D.api('GET', '/registry/file/', params={'version': item['version']})
        if not response.ok:
            raise RuntimeError('downloading the model failed, {}'.format(router_message(response)))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.safetensors')
            with open(path, 'wb') as f:
                f.write(response.content)
            arrays, meta = mods['artifact_io'].load_artifact(path, expected_sha256=item['sha256'])
        model = self.lab.build(bucket)
        model.load_state_dict(mods['weights'].arrays_to_state(
            arrays, meta.get('complex_keys', [])))
        return model

    def _compute(self, item, bucket):
        import numpy as np
        metrics = self.lab.mods['metrics']
        scores, preds = self.lab.score(self._model(item, bucket), self.samples)
        rows = [{'region': s['region'], **{k: round(float(r[k]), 4)
                                           for k in ('rel_l2_pct', 'peak_distance_mm', 'ssim')}}
                for s, r in zip(self.samples, scores)]
        slices = []
        for shown in self.shown:
            s, pred = self.samples[shown['index']], preds[shown['index']]
            amp = metrics.amplitude(pred) * s['scale']
            slices.append(np.take(amp, shown['y'], 1).round(0).astype(int).tolist())
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ('rel_l2_pct', 'peak_distance_mm', 'ssim')}
        return {'rows': rows, 'mean': mean, 'slices': slices}


# ── centralized baseline ────────────────────────────────────────────────────

class EpochLog:
    """Takes the place of the task's logger and keeps the loss of each epoch."""

    def __init__(self):
        self.losses = []

    def info(self, message, *args):
        found = LOSS.search(str(message))
        if found:
            self.losses.append(float(found.group(1)))

    warning = error = debug = info


def pooled_dataset(samples):
    """The pooled samples as a torch Dataset whose items match the task's dataset."""
    from torch.utils.data import Dataset

    class Pooled(Dataset):

        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]['tensors']

    return Pooled()


class Baselines:
    """The centralized and the local-only baseline of the run on the page.

    The centralized baseline trains one model on the pooled train samples of
    sites A, B and C. The local-only baseline trains one model per site on
    that site's own train samples and never aggregates, and its scores are
    the simple average over the sites. With all three stores on one machine,
    only this demo can train either. Both mirror the run's config. They use
    the task's own model, start, optimizer, loss and batch size through
    ``training.train_round``, and the same rounds of local epochs. Every
    sample is thus seen as often as in the federated run. After every round
    each model is scored on site A's eval samples with the gate's metrics.
    Results stay in memory, one per kind and config.
    """

    KINDS = ('centralized', 'local')
    NAMES = {'centralized': 'The centralized baseline', 'local': 'The local-only baseline'}

    def __init__(self, lab, stores):
        self.lab = lab
        self.stores = stores
        self.note = lambda text, kind='info': print(text, flush=True)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.jobs, self.queue, self.weights = {}, [], {}
        threading.Thread(target=self._work, name='baselines', daemon=True).start()

    def want(self, config):
        """The key of each baseline for a run's config, queued for training when new."""
        spec = {'bucket_hz': config['bucket_hz'], 'rounds': config['total_round'],
                'epochs': config['local_epochs'], 'package': config['model_package'],
                'seed': str(config['seed_model_version']), 'train': config['train']}
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
        keys = {}
        with self.lock:
            for kind in self.KINDS:
                key = '{}-{}'.format(kind, digest)
                keys[kind] = key
                if key not in self.jobs:
                    self.jobs[key] = {'key': key, 'kind': kind, 'status': 'queued',
                                      'rounds': spec['rounds'], 'epochs': spec['epochs'],
                                      'round': 0, 'scores': [], 'losses': [], 'per_site': {},
                                      'n_train': None, 'seconds': None, 'error': None}
                    self.queue.append((key, dict(spec, kind=kind)))
                    self.wake.set()
        return keys

    def status(self, key):
        with self.lock:
            job = self.jobs.get(key)
            return copy.deepcopy(job) if job else None

    def model(self, key, bucket, site=None):
        """A copy of a baseline's final model, for local-only the one of ``site``."""
        with self.lock:
            weights = self.weights.get((key, site))
        if weights is None:
            raise RuntimeError('the baseline has not finished')
        model = self.lab.build(bucket)
        model.load_state_dict(self.lab.mods['weights'].arrays_to_state(*weights))
        return model

    def _update(self, key, **changes):
        with self.lock:
            self.jobs[key].update(changes)

    def _work(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    key, spec = self.queue.pop(0)
                try:
                    self._train(key, spec)
                except Exception as e:
                    traceback.print_exc()
                    message = '{}: {}'.format(e.__class__.__name__, e)[:300]
                    self._update(key, status='failed', error=message)
                    self.note('{} failed, {}'.format(self.NAMES[spec['kind']], message), 'bad')

    def _means(self, model, samples):
        rows, _ = self.lab.score(model, samples)
        summary = self.lab.mods['metrics'].summarize(rows)
        return {name: summary[name]['mean'] for name in METRICS}

    @staticmethod
    def _average(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def _publish(self, key, scores, losses, rnd):
        """Scores and losses so far, averaged over the models with equal weight."""
        groups = list(scores)
        changes = {'round': rnd,
                   'scores': [{m: self._average([scores[g][i][m] for g in groups]) for m in METRICS}
                              for i in range(len(scores[groups[0]]))],
                   'losses': [[self._average([losses[g][r][e] for g in groups])
                               for e in range(min(len(losses[g][r]) for g in groups))]
                              for r in range(len(losses[groups[0]]))]}
        if groups != [None]:
            changes['site_scores'] = {g: [dict(s) for s in v] for g, v in scores.items()}
            changes['site_losses'] = {g: [list(x) for x in v] for g, v in losses.items()}
        self._update(key, **changes)

    def _train(self, key, spec):
        kind = spec['kind']
        mods = self.lab.load(spec['package'])
        torch, standin, training = self.lab.torch, mods['standin'], mods['training']
        seed = spec['seed']
        if not (seed.startswith('init:') and seed[5:].isdigit()):
            raise RuntimeError('the baselines start from init seeds only, '
                               'this project starts from {}'.format(seed))
        bucket = spec['bucket_hz']
        evals = self.lab.eval_samples(bucket)
        data = {site: [self.lab.read(r) for r in self.lab.records(root, bucket, 'train')]
                for site, root in self.stores.items()}
        per_site = {site: len(samples) for site, samples in data.items()}
        # One model on the pooled samples, or one model per site on its own samples
        if kind == 'centralized':
            groups = {None: [s for samples in data.values() for s in samples]}
        else:
            groups = {site: samples for site, samples in data.items() if samples}
        if not any(groups.values()):
            raise RuntimeError('the site stores hold no train samples at {} Hz'.format(bucket))
        started = time.time()
        self._update(key, status='training', per_site=per_site, n_train=sum(per_site.values()))
        if kind == 'centralized':
            self.note('The centralized baseline starts, {} rounds of {} epochs on the {} train '
                      'samples of sites {} pooled'.format(spec['rounds'], spec['epochs'],
                                                         sum(per_site.values()),
                                                         who(per_site)[len('Sites '):]))
        else:
            sites = who(groups)
            self.note('The local-only baseline starts, {} each training {} rounds of {} epochs on '
                      'their own samples without aggregation'.format(
                          sites[0].lower() + sites[1:], spec['rounds'], spec['epochs']))
        from torch.utils.data import DataLoader
        models = {g: self.lab.build(bucket, int(seed[5:])) for g in groups}
        scores = {g: [self._means(model, evals)] for g, model in models.items()}
        losses = {g: [] for g in groups}
        self._publish(key, scores, losses, 0)
        # FedProx only makes sense across sites, so the baselines leave it out
        cfg = dict(spec['train'], local_epochs=spec['epochs'], fedprox_mu=0.0)
        for rnd in range(1, spec['rounds'] + 1):
            for index, (g, samples) in enumerate(groups.items()):
                # Shuffled with its own generator, seeded per round as the sites seed theirs
                shuffle = torch.Generator().manual_seed(
                    int(cfg.get('seed', 0)) + 1000 * rnd + index)
                loader = DataLoader(pooled_dataset(samples),
                                    batch_size=int(cfg.get('batch_size', 1)),
                                    shuffle=True, generator=shuffle)
                log = EpochLog()
                training.train_round(models[g], loader, standin, torch.device('cpu'), cfg,
                                     training.stage_weights(cfg.get('curriculum'), rnd), log)
                scores[g].append(self._means(models[g], evals))
                losses[g].append(log.losses)
            self._publish(key, scores, losses, rnd)
        with self.lock:
            for g, model in models.items():
                self.weights[(key, g)] = mods['weights'].state_to_arrays(model.state_dict())
        seconds = time.time() - started
        self._update(key, status='done', seconds=round(seconds, 1))
        last = self.status(key)['scores'][-1]['rel_l2_pct']
        self.note('{} finished in {:.0f} s, relative l2 error {:.1f} % on site A\'s eval samples{}'
                  .format(self.NAMES[kind], seconds, last,
                          '' if kind == 'centralized' else ', averaged over the sites'))


# ── the router, polled ──────────────────────────────────────────────────────

class Monitor:
    """Polls the router and keeps the page's state, with a narration of what changed."""

    def __init__(self, fields, baselines):
        self.fields, self.baselines = fields, baselines
        self.lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.wake = threading.Event()
        self.snapshot = {'ready': False}
        self.events = deque(maxlen=150)
        self.infos = {}
        self.project_id = None
        self.seen_sites, self.seen_gates, self.seen_models = {}, set(), {}
        self.quiet = True
        self.starting = False

    def start_polling(self):
        threading.Thread(target=self._loop, name='monitor', daemon=True).start()

    def note(self, text, kind='info'):
        with self.lock:
            self.events.appendleft({'t': time.time(), 'text': text, 'kind': kind})
        print('[{}] {}'.format(time.strftime('%H:%M:%S'), text), flush=True)

    def state(self):
        with self.lock:
            snapshot = dict(self.snapshot)
            snapshot['events'] = list(self.events)
        snapshot['fields'] = self.fields.status()
        return snapshot

    def _loop(self):
        while True:
            error = None
            try:
                snapshot = self._poll()
            except requests.RequestException as e:
                error = 'The router is not reachable ({})'.format(e.__class__.__name__)
            except Exception as e:
                traceback.print_exc()
                error = 'Reading the run failed, {}: {}'.format(e.__class__.__name__, e)[:300]
            with self.lock:
                if error:
                    snapshot = dict(self.snapshot)
                snapshot.update(ready=True, error=error, polled_at=time.time())
                self.snapshot = snapshot
            self.wake.wait(POLL_S)
            self.wake.clear()

    def _poll(self):
        site_status = {s['uid']: s.get('status') for s in D.list_all('/sites/')}
        projects = [p for p in D.list_all('/projects/') if p['name'].startswith(PROJECT_PREFIX)]
        if not projects:
            return {'project': None, 'config': {}, 'sites': [], 'rounds': [], 'registry': [],
                    'active': False}
        project = max(projects, key=lambda p: p['id'])
        if project['id'] != self.project_id:
            self._switch(project)
        runs = sorted(D.runs_of(project['id'], project['batch']),
                      key=D.site_of) if project['batch'] else []
        config = self._config(project, runs)
        sites = []
        for run in runs:
            key = D.site_of(run)
            sites.append({'key': key, 'name': D.SITE_NAMES[key], 'role': run['role'],
                          'run': run['id'], 'status': run['status'], 'round': D.round_of(run),
                          'connected': site_status.get(run['site_uid']),
                          'created_at': run['created_at'], 'updated_at': run['updated_at']})
        rounds = self._rounds(runs, sites, config['total_round'])
        registry = self._registry(project['id'], config['bucket_hz'])
        self._narrate(sites, rounds, registry)
        keys = self.baselines.want(config) if config['total_round'] else {}
        baseline, local = (self.baselines.status(keys[kind]) if keys else None
                           for kind in Baselines.KINDS)
        self._want_fields(config, registry, baseline, local)
        return {'project': {'id': project['id'], 'name': project['name'],
                            'batch': project['batch'], 'created_at': project.get('created_at')},
                'config': config, 'sites': sites, 'rounds': rounds, 'registry': registry,
                'baseline': baseline, 'local': local,
                'site_status': {key: site_status.get(uid) for key, uid in D.SITES.items()},
                'active': any(s['status'] not in TERMINAL for s in sites)}

    def _switch(self, project):
        self.project_id = project['id']
        self.seen_sites, self.seen_gates, self.seen_models = {}, set(), {}
        # A run started from this page is narrated from its first state
        self.quiet, self.starting = not self.starting, False
        if self.quiet:
            self.note('Showing the latest demo project, {}'.format(project['name']))

    @staticmethod
    def _config(project, runs):
        tasks = (runs[0].get('tasks') if runs else None) or project.get('tasks') or [{}]
        cfg = tasks[0].get('config') or {}
        gate = dict(GATE_DEFAULTS)
        gate.update(cfg.get('gate') or {})
        return {'total_round': int(cfg.get('total_round') or 0),
                'local_epochs': int(cfg.get('local_epochs') or 1),
                'bucket_hz': int((cfg.get('data_source') or {}).get('bucket_hz') or 250000),
                'model_package': cfg.get('model_package', 'standin'),
                'seed_model_version': cfg.get('seed_model_version', 'init:0'),
                'min_samples': cfg.get('min_samples'), 'gate': gate,
                # The training settings a centralized baseline has to share with the sites
                'train': {k: cfg[k] for k in ('lr', 'weight_decay', 'batch_size', 'grad_accum',
                                              'clip_norm', 'amp', 'curriculum', 'seed')
                          if cfg.get(k) is not None}}

    def _rounds(self, runs, sites, total):
        rounds = [{'round': r, 'sites': {}, 'gate': None} for r in range(1, total + 1)]
        for run, site in zip(runs, sites):
            current = min(site['round'] or 0, total)
            for rnd in range(1, current + 1):
                key = (run['id'], rnd)
                info = self.infos.get(key)
                if info is None:
                    info = round_info(D.run_files(run['id'], rnd), rnd)
                    # A round's files stop changing once the site has moved on or ended
                    if rnd < (site['round'] or 0) or site['status'] in TERMINAL:
                        self.infos[key] = info
                info = dict(info)
                gate = info.pop('gate')
                if gate:
                    rounds[rnd - 1]['gate'] = gate
                rounds[rnd - 1]['sites'][site['key']] = info
        return rounds

    @staticmethod
    def _registry(project_id, bucket):
        response = D.api('GET', '/registry/', params={'bucket_hz': bucket, 'status': 'any'})
        response.raise_for_status()
        models = []
        for m in response.json():
            if m.get('project') != project_id:
                continue
            models.append({'version': m['version'], 'round': m['round_seq'],
                           'parent': m.get('parent'), 'status': m['status'],
                           'sha256': m['sha256'], 'created_at': m.get('created_at'),
                           'reviewed_by': m.get('reviewed_by'), 'reviewed_at': m.get('reviewed_at'),
                           'review_note': m.get('review_note'),
                           'gate': means((m.get('eval_report') or {}).get('candidate'))})
        return sorted(models, key=lambda m: m['round'])

    def _narrate(self, sites, rounds, registry):
        quiet, self.quiet = self.quiet, False
        groups = {}
        for site in sites:
            state = (site['status'], site['round'])
            if self.seen_sites.get(site['key']) != state:
                self.seen_sites[site['key']] = state
                groups.setdefault(state, []).append(site['key'])
        for (status, rnd), keys in sorted(groups.items(), key=lambda g: sorted(g[1])):
            text = phrase(status, rnd, keys)
            if quiet or not text:
                continue
            if status == 'Failed':
                errors = [((rounds[rnd - 1]['sites'].get(k) or {}).get('error'))
                          for k in keys if rnd and rnd <= len(rounds)]
                errors = [e for e in errors if e]
                if errors:
                    text += ', ' + errors[0]
            self.note(text, 'bad' if status in ('Failed', 'Pending Failed') else 'info')
        for entry in rounds:
            gate = entry['gate']
            if gate and entry['round'] not in self.seen_gates:
                self.seen_gates.add(entry['round'])
                if not quiet:
                    self.note(gate_phrase(entry['round'], gate),
                              'good' if gate['accepted'] else 'warn')
        for model in registry:
            before = self.seen_models.get(model['version'])
            self.seen_models[model['version']] = model['status']
            if quiet or before == model['status']:
                continue
            if before is None:
                self.note('{} from round {} entered the registry and waits for the coordinator'
                          .format(model['version'], model['round']))
            else:
                self.note('{} is {} in the registry'.format(model['version'], model['status']),
                          'good' if model['status'] == 'approved' else 'warn')

    def _want_fields(self, config, registry, baseline, local):
        wanted = []
        seed = str(config['seed_model_version'])
        if seed.startswith('init:') and seed[5:].isdigit():
            wanted.append({'version': seed, 'seed': int(seed[5:])})
        wanted += [{'version': m['version'], 'sha256': m['sha256']} for m in registry]
        if baseline and baseline['status'] == 'done':
            wanted.append({'version': baseline['key'], 'baseline': baseline['key']})
        if local and local['status'] == 'done':
            wanted += [{'version': '{}-{}'.format(local['key'], site), 'baseline': local['key'],
                        'site': site} for site in sorted(local.get('site_scores') or {})]
        self.fields.want(wanted, config['bucket_hz'], config['model_package'])

    # ── actions from the page ───────────────────────────────────────────────

    @staticmethod
    def _latest_active():
        """The newest demo project if one of its runs is still going, read fresh from the router."""
        projects = [p for p in D.list_all('/projects/') if p['name'].startswith(PROJECT_PREFIX)]
        if not projects:
            return None
        project = max(projects, key=lambda p: p['id'])
        runs = D.runs_of(project['id'], project['batch']) if project['batch'] else []
        return project if any(r['status'] not in TERMINAL for r in runs) else None

    def start_run(self, body):
        try:
            rounds, epochs = int(body.get('rounds', 3)), int(body.get('epochs', 4))
            margin = body.get('peak_margin_mm')
            margin = None if margin in (None, '') else float(margin)
        except (TypeError, ValueError):
            return 400, {'error': 'Rounds and epochs take whole numbers and the margin a number.'}
        if not (1 <= rounds <= 10 and 1 <= epochs <= 20) or (
                margin is not None and not 0 <= margin <= 100):
            return 400, {'error': 'Use 1 to 10 rounds, 1 to 20 epochs and a margin from 0 to 100 mm.'}
        with self.action_lock:
            try:
                if self._latest_active():
                    return 409, {'error': 'A run is still going. Start the next one when every '
                                          'site has finished.'}
                self.starting = True
                project = D.create_project(rounds, epochs, margin)
                if project is None:
                    raise RuntimeError('the new project is missing from the router')
                D.start_run(project)
            except requests.HTTPError as e:
                self.starting = False
                message = 'Starting a run failed, {}.'.format(router_message(e.response))
                self.note(message, 'bad')
                return 502, {'error': message}
            except (requests.RequestException, RuntimeError) as e:
                self.starting = False
                return 502, {'error': 'Starting a run failed, {}.'.format(
                    e if isinstance(e, RuntimeError) else 'the router is not reachable')}
            finally:
                self.wake.set()
        margin_text = '' if margin is None else ', peak-distance margin {} mm'.format(margin)
        self.note('Run started from this page, {} rounds of {} local epochs{}'.format(
            rounds, epochs, margin_text))
        return 200, {'project': project['name']}

    def approve(self, body):
        version = str(body.get('version') or '')
        with self.action_lock:
            model = next((m for m in self.state().get('registry') or []
                          if m['version'] == version), None)
            if model is None:
                return 404, {'error': 'No model {} in this project.'.format(version)}
            if model['status'] != 'pending':
                return 409, {'error': '{} is already {}.'.format(version, model['status'])}
            try:
                response = D.api('POST', '/registry/approve/', json={
                    'version': version, 'note': 'approved on the demo page'})
            except requests.RequestException as e:
                return 502, {'error': 'The router is not reachable ({}).'.format(
                    e.__class__.__name__)}
            finally:
                self.wake.set()
            if not response.ok:
                return 502, {'error': 'Approving {} failed, {}.'.format(
                    version, router_message(response))}
        self.note('The coordinator approved {}. Labs can download it from the registry'.format(
            version), 'good')
        return 200, response.json()


# ── HTTP ────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    """The page, its state as JSON, and the two actions. Only localhost may call it."""
    server_version = 'BabelBrainDemoUI/1'
    monitor = None
    fields = None

    def log_message(self, fmt, *args):
        # The page polls every two seconds, so the log keeps only the narration
        pass

    def _host_ok(self):
        host = urlsplit('//' + (self.headers.get('Host') or '')).hostname or ''
        return host in LOCAL_HOSTS

    def _send(self, code, body, content_type='application/json'):
        if isinstance(body, bytes):
            data = body
        else:
            data = json.dumps(clean(body), allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {'error': 'Open the page at http://localhost:8090.'})
        path = urlsplit(self.path).path
        if path in ('/', '/index.html'):
            with open(PAGE, 'rb') as f:
                return self._send(200, f.read(), 'text/html; charset=utf-8')
        if path == '/api/state':
            return self._send(200, self.monitor.state())
        if path == '/api/fields':
            return self._send(200, self.fields.payload())
        if path == '/healthz':
            return self._send(200, {'ok': True})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        # Same-origin JSON only, so another site open in the browser cannot press the buttons
        origin = self.headers.get('Origin')
        if not self._host_ok() or (origin and urlsplit(origin).netloc != self.headers.get('Host')):
            return self._send(403, {'error': 'Use the page at http://localhost:8090.'})
        if (self.headers.get('Content-Type') or '').split(';')[0].strip() != 'application/json':
            return self._send(415, {'error': 'Send JSON.'})
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = MAX_BODY + 1
        if length > MAX_BODY:
            return self._send(413, {'error': 'Request too large.'})
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            return self._send(400, {'error': 'Send JSON.'})
        if not isinstance(body, dict):
            return self._send(400, {'error': 'Send a JSON object.'})
        path = urlsplit(self.path).path
        if path == '/api/start':
            return self._send(*self.monitor.start_run(body))
        if path == '/api/approve':
            return self._send(*self.monitor.approve(body))
        self._send(404, {'error': 'not found'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--host', default='0.0.0.0',
                        help='address to listen on inside the container')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args(argv)
    lab = Lab(EVAL_STORE)
    fields = Fields(lab)
    baselines = Baselines(lab, {key: os.path.join(SITE_STORES, 'site-' + key.lower())
                                for key in sorted(D.SITES)})
    monitor = Monitor(fields, baselines)
    fields.baseline, baselines.note = baselines, monitor.note
    monitor.start_polling()
    Handler.monitor, Handler.fields = monitor, fields
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    # docker stop sends SIGTERM, which Python as PID 1 would otherwise ignore
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print('Demo page at http://localhost:{}, router {}'.format(args.port, D.ROUTER), flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
