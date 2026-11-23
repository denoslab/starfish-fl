"""
``starfish-site check``, which tells a practitioner whether this site is ready.

It reads the settings, asks the router who this site is, opens the sample
stores, inspects torch and the GPU, and trains the stand-in model for one
epoch on a few synthetic samples on the device this site will use. The
synthetic samples go to a temporary folder and are deleted afterwards.
"""

import os
import re
import shutil
import tempfile
import time
import uuid
from urllib.parse import urlparse

DEVICE_PATTERN = re.compile(r'^(cpu|mps|cuda(:\d+)?)$')
LOCAL_HOSTS = ('localhost', '127.0.0.1', '::1')


class Report:
    """Collects check results and prints each line as it comes."""

    def __init__(self, write=print):
        self.write = write
        self.failed = 0
        self.warned = 0

    def section(self, title):
        self.write('\n' + title)

    def ok(self, text):
        self.write('  ok    ' + text)

    def warn(self, text, hint=None):
        self.warned += 1
        self.write('  warn  ' + text)
        if hint:
            self.write('        ' + hint)

    def fail(self, text, hint=None):
        self.failed += 1
        self.write('  FAIL  ' + text)
        if hint:
            self.write('        ' + hint)


def check_settings(values, report):
    report.section('Settings')
    uid = values.get('SITE_UID')
    try:
        uuid.UUID(uid or '')
        report.ok('SITE_UID {}'.format(uid))
    except ValueError:
        report.fail('SITE_UID is missing or not a UUID', 'Enrol first, see starfish-site enrol --help')
    url = values.get('ROUTER_URL')
    if not url:
        report.fail('ROUTER_URL is missing', 'Enrol first, see starfish-site enrol --help')
    elif url.startswith('https://'):
        report.ok('ROUTER_URL {}'.format(url))
    elif urlparse(url).hostname in LOCAL_HOSTS:
        report.ok('ROUTER_URL {}, plain HTTP on this machine'.format(url))
    else:
        report.warn('ROUTER_URL {} is plain HTTP'.format(url),
                    'The token travels unencrypted. Ask the router\'s admin for its https URL.')
    if values.get('ROUTER_TOKEN'):
        report.ok('ROUTER_TOKEN is set')
    elif values.get('ROUTER_USERNAME') and values.get('ROUTER_PASSWORD'):
        report.warn('The router password is used in place of a site token',
                    'Enrol with a code from the router\'s admin to get a token.')
    else:
        report.fail('No ROUTER_TOKEN', 'Enrol first, see starfish-site enrol --help')
    bundle = values.get('REQUESTS_CA_BUNDLE')
    if bundle:
        if os.path.isfile(bundle):
            report.ok('REQUESTS_CA_BUNDLE {}'.format(bundle))
        else:
            report.fail('REQUESTS_CA_BUNDLE {} does not exist'.format(bundle))
    for key, needed in (('BABELBRAIN_FL_STORE', True), ('BABELBRAIN_FL_EVAL_STORE', False)):
        path = values.get(key)
        if not path:
            if needed:
                report.warn('{} is not set'.format(key),
                            'A site needs its sample store to train. Add it to site.env.')
        elif os.path.isdir(path):
            report.ok('{} {}'.format(key, path))
        else:
            report.fail('{} {} is not a folder'.format(key, path))
    device = values.get('BABELBRAIN_FL_DEVICE')
    if device and not DEVICE_PATTERN.match(device.strip()):
        report.fail('BABELBRAIN_FL_DEVICE {} is not cpu, mps, cuda or cuda:N'.format(device))
    elif device:
        report.ok('BABELBRAIN_FL_DEVICE {}'.format(device))


def check_router(values, report):
    import requests
    from starfish.controller.router_auth import router_auth

    report.section('Router')
    url, uid = values.get('ROUTER_URL'), values.get('SITE_UID')
    if not url or not uid:
        report.fail('Skipped, the settings are incomplete')
        return
    started = time.monotonic()
    try:
        response = requests.get('{}/sites/lookup/'.format(url), params={'uid': uid},
                                auth=router_auth(), timeout=15)
    except requests.exceptions.SSLError as e:
        report.fail('The router\'s HTTPS certificate is not trusted: {}'.format(e),
                    'Enrol with --ca-cert, or set REQUESTS_CA_BUNDLE to the CA file from the admin.')
        return
    except requests.RequestException as e:
        report.fail('The router cannot be reached: {}'.format(e.__class__.__name__),
                    'Check the URL, that this device is on the same network, and that the router runs.')
        return
    ms = 1000 * (time.monotonic() - started)
    if response.status_code in (401, 403):
        report.fail('The router refused this site\'s credentials ({})'.format(response.status_code),
                    'The token may be revoked. Ask the admin for a new enrolment code.')
    elif response.status_code == 404:
        report.fail('The router does not know site {}'.format(uid),
                    'Enrol again with a new code from the admin.')
    elif not response.ok:
        report.fail('The router answered {}'.format(response.status_code))
    else:
        site = response.json()
        report.ok('Reached in {:.0f} ms, this is site "{}", id {}'.format(
            ms, site.get('name'), site.get('id')))


def describe_store(root):
    """Usable samples of a store per bucket and split, and the reasons any were rejected."""
    from starfish.controller.tasks.babel_brain_fno.store import SampleStore

    store = SampleStore(root)
    counts = {}
    for record in store.samples():
        counts.setdefault(record.bucket_hz, {}).setdefault(record.split, 0)
        counts[record.bucket_hz][record.split] += 1
    return counts, dict(store.rejections)


def check_stores(values, report):
    report.section('Sample stores')
    found = False
    for key in ('BABELBRAIN_FL_STORE', 'BABELBRAIN_FL_EVAL_STORE'):
        root = values.get(key)
        if not root or not os.path.isdir(root):
            continue
        found = True
        started = time.monotonic()
        try:
            counts, rejected = describe_store(root)
        except Exception as e:
            report.fail('{} cannot be read: {}: {}'.format(key, e.__class__.__name__, e))
            continue
        if not counts:
            report.fail('{} holds no usable sample'.format(key))
            continue
        parts = ['{} kHz {}'.format(bucket // 1000, ', '.join(
            '{} {}'.format(n, split) for split, n in sorted(splits.items())))
            for bucket, splits in sorted(counts.items())]
        report.ok('{} has {}, checked in {:.1f} s'.format(
            key, '; '.join(parts), time.monotonic() - started))
        if rejected:
            report.warn('{} rejected samples: {}'.format(key, ', '.join(
                '{} {}'.format(n, reason) for reason, n in sorted(rejected.items()))))
    if not found:
        report.warn('No store to check')


def arch_supported(capability, arches):
    """
    Whether a torch build with ``arches`` runs on a GPU of ``capability``.

    Kernels for ``sm_86`` run on any 8.x GPU from 8.6 on, such as the 8.9 of
    Ada cards, and PTX for ``compute_XY`` compiles for any later GPU.
    Kernels for an ``a`` or ``f`` variant such as ``sm_90a`` run on that GPU only.
    """
    for arch in arches:
        kind, _, version = arch.partition('_')
        found = re.match(r'(\d+)(\w*)$', version)
        if not found:
            continue
        built = divmod(int(found.group(1)), 10)
        if found.group(2):
            if built == tuple(capability):
                return True
        elif kind == 'sm' and built[0] == capability[0] and built[1] <= capability[1]:
            return True
        elif kind == 'compute' and built <= tuple(capability):
            return True
    return False


def check_torch(report):
    """Torch, the GPU and the device this site trains on. Return that device or None."""
    report.section('Torch and device')
    try:
        import torch
    except ImportError:
        report.fail('torch is not installed', 'Run install.sh again.')
        return None
    from starfish.controller.tasks.babel_brain_fno import training as T

    report.ok('torch {}, built for CUDA {}'.format(torch.__version__, torch.version.cuda or 'none'))
    if torch.cuda.is_available():
        arches = torch.cuda.get_arch_list()
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            arch = 'sm_{}{}'.format(*capability)
            name = torch.cuda.get_device_name(index)
            memory = torch.cuda.get_device_properties(index).total_memory / 2 ** 30
            if arch_supported(capability, arches):
                report.ok('cuda:{} {}, {}, {:.0f} GB'.format(index, name, arch, memory))
            else:
                report.fail('cuda:{} {} is {}, which this torch build has no kernels for'.format(
                    index, name, arch), 'Run install.sh again to get a torch build for CUDA 12.8.')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        report.ok('Apple MPS is available')
    try:
        device = T.pick_device('auto')
    except Exception as e:
        report.fail('No usable device: {}'.format(e))
        return None
    report.ok('This site trains on {}{}'.format(
        device, ', set by BABELBRAIN_FL_DEVICE' if os.getenv(T.DEVICE_ENV) else ''))
    return device


def training_check(device, samples=4, epochs=1):
    """
    Train the stand-in for ``epochs`` on synthetic samples on ``device``.

    Returns the seconds taken, the last loss, and the largest difference
    between the model's output on ``device`` and on the CPU, relative to the
    output's largest value.
    """
    import torch
    from torch.utils.data import DataLoader

    from starfish.controller.tasks.babel_brain_fno import standin, training as T
    from starfish.controller.tasks.babel_brain_fno.sample_io import torch_dataset
    from starfish.controller.tasks.babel_brain_fno.store import SampleStore
    from starfish.controller.tasks.babel_brain_fno.synthetic import write_synthetic_store

    folder = tempfile.mkdtemp(prefix='starfish-site-check-')
    try:
        write_synthetic_store(folder, n_groups=samples, per_group=1, val_groups=0, seed=1)
        records = SampleStore(folder).samples()
        loader = DataLoader(torch_dataset(records), batch_size=1, shuffle=False)
        torch.manual_seed(0)
        model = standin.build_model(250000)
        x, _, _ = standin.inputs(next(iter(loader)))
        model.eval()
        with torch.no_grad():
            on_cpu = model(x)
            model.to(device)
            on_device = model(x.to(device)).float().cpu()
        difference = float((on_device - on_cpu).abs().max() / on_cpu.abs().max().clamp_min(1e-12))
        started = time.monotonic()
        metrics = T.train_round(model, loader, standin, device, {'local_epochs': epochs},
                                T.stage_weights(None, 1), _Quiet())
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.monotonic() - started, metrics['train_loss'], difference
    finally:
        shutil.rmtree(folder, ignore_errors=True)


class _Quiet:

    def info(self, *args, **kwargs):
        pass


def check_training(device, report):
    report.section('Training on this device')
    if device is None:
        report.fail('Skipped, no device')
        return
    try:
        seconds, loss, difference = training_check(device)
    except Exception as e:
        hint = None
        if device.type == 'mps':
            hint = ('Put BABELBRAIN_FL_DEVICE=\'cpu\' in site.env to train on the CPU, '
                    'or PYTORCH_ENABLE_MPS_FALLBACK=\'1\' to run what MPS lacks on the CPU.')
        report.fail('Training the stand-in on {} failed: {}: {}'.format(
            device, e.__class__.__name__, e), hint)
        return
    if difference > 1e-3:
        report.fail('The model\'s output on {} differs from the CPU by {:.1e} of its peak'.format(
            device, difference))
    else:
        report.ok('Output on {} matches the CPU to {:.1e} of its peak'.format(device, difference))
    report.ok('One epoch on 4 synthetic samples took {:.2f} s, loss {:.4f}'.format(seconds, loss))


def run_checks(values, quick=False, write=print):
    """Run every check and return the report. Without settings only torch and training are checked."""
    report = Report(write)
    if values:
        check_settings(values, report)
        check_router(values, report)
        check_stores(values, report)
    device = check_torch(report)
    if not quick:
        check_training(device, report)
    report.section('{} failed, {} warnings'.format(report.failed, report.warned))
    return report
