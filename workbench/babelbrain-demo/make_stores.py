"""
Fill the demo's sample stores with BabelBrain's own exporter and backfill tool.

    python make_stores.py [--root /demo-data] [--reset]

Runs in the controller image through ``make babelbrain-demo-stores``, with
the BabelBrain fork's ``BabelBrain/BabelBrain`` folder on the Python path.
Every run is a toy, see ``toy_physics.py``; only the crop is a stand-in.
Eligibility, the store layout, group IDs, splits, counters, region labels
and the manifest are BabelBrain's code, unchanged.

    <root>/runs/      toy Step 2 outputs, laid out like lab study folders
    <root>/stores/    site-a, site-b, site-c and eval-a, one store each
"""

import argparse
import glob
import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.environ.get('BABELBRAIN_PKG', '/babelbrain'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FederatedLearning import backfill  # noqa: E402
from FederatedLearning import synthetic as bb_synthetic  # noqa: E402
from FederatedLearning.exporter import export_from_step2  # noqa: E402
from FederatedLearning.store import SampleStore  # noqa: E402

import toy_physics as toy  # noqa: E402

# What a lab states for a whole folder of past runs, question S11
STATED = SimpleNamespace(babelbrain_version='0.8.2', real_ct=True, default_options=True,
                         eval_regions=False)
LAB_TARGETS = ('LeftVIM_12_Aug_2026_194643',
               'RightSTN', 'LeftM1', 'RightAmygdala')
HARD_REGIONS = ('P7', 'P8', 'PO7', 'TP7', 'TP8')
STORES = ('site-a', 'site-b', 'site-c', 'eval-a')


def heading(text):
    print('\n' + text)
    print('-' * len(text))


def lab_runs(folder, lab, n_subjects, rng, extra=()):
    """Two targets per toy subject, plus ``extra`` (name suffix, subject) runs; return the paths."""
    runs = []
    for i in range(1, n_subjects + 1):
        subject = 'demo-{}-sub{:02d}'.format(lab, i)
        for target in rng.choice(LAB_TARGETS, 2, replace=False):
            prefix = '{}_{}_Single_250kHz_9PPW_'.format(subject, target)
            runs.append(toy.write_run(os.path.join(folder, subject), prefix, rng,
                                      rng.uniform(1.5, 3.0), rng.uniform(-0.3, 0.3)))
    for tail, subject in extra:
        prefix = '{}_{}'.format(subject, tail)
        runs.append(toy.write_run(os.path.join(
            folder, subject), prefix, rng, 2.0, 0.0))
    return runs


def show_store(root):
    summary = SampleStore(root).summary()
    counts = ', '.join('{} {}'.format(n, split)
                       for (_, split), n in sorted(summary['counts'].items()))
    outcomes = ', '.join('{} {}'.format(k.replace('_', ' '), v)
                         for k, v in sorted(summary['outcomes'].items()))
    print('Store now holds {} at 250 kHz, {:.1f} MB. Outcomes so far: {}'.format(
        counts, summary['bytes'] / 1e6, outcomes))


def site_a(root, rng):
    heading('Site A, NeuroFUS, the coordinator: back-fill 14 past subjects')
    folder = os.path.join(root, 'runs', 'site-a')
    lab_runs(folder, 'a', 14, rng)
    store = os.path.join(root, 'stores', 'site-a')
    print('$ python -m FederatedLearning.backfill <studies> --dry-run '
          '--babelbrain-version 0.8.2 --real-ct --default-options')
    backfill.run(folder, store_root=store, dry_run=True, args=STATED)
    print('$ ... the same without --dry-run')
    backfill.run(folder, store_root=store, args=STATED, crop=toy.demo_crop)
    show_store(store)


def site_b(root, rng):
    heading('Site B: 12 subjects exported live, as the Step 2 hook does after each run')
    folder = os.path.join(root, 'runs', 'site-b')
    store = os.path.join(root, 'stores', 'site-b')
    runs = lab_runs(folder, 'b', 12, rng)
    # Two runs the rules turn away: a ZTE pseudo-CT and a homogeneous medium
    zte = lab_runs(folder, 'b-zte', 0, rng,
                   [('RightSTN_Single_250kHz_9PPW_', 'demo-b-sub13')])
    homog = lab_runs(folder, 'b-hom', 0, rng,
                     [('LeftM1_Single_250kHz_9PPW_', 'demo-b-sub14')])
    outcomes = {}
    for full, info in ([(r, {}) for r in runs] + [(zte[0], {'CTType': 2})] +
                       [(homog[0], {'options': dict(bb_synthetic.run_info('')['options'],
                                                    bForceHomogenousMedium=True)})]):
        water = full.replace('DataForSim.h5', 'Water_DataForSim.h5')
        result = export_from_step2(full, water,
                                   bb_synthetic.run_info(
                                       os.path.dirname(full), **info),
                                   store_root=store, source='live', crop=toy.demo_crop)
        outcomes[result.reason] = outcomes.get(result.reason, 0) + 1
    print('Step 2 runs finished: {}'.format(sum(outcomes.values())))
    for reason, n in sorted(outcomes.items()):
        print('  {}: {}'.format('exported' if reason ==
              'ok' else reason.replace('_', ' '), n))
    show_store(store)


def site_c(root, rng):
    heading(
        'Site C: back-fill 12 past subjects, some runs not eligible, then run it again')
    folder = os.path.join(root, 'runs', 'site-c')
    lab_runs(folder, 'c', 12, rng, [('LeftVIM_Single_400kHz_9PPW_', 'demo-c-sub13'),
                                    ('RightSTN_CTX_500_500kHz_6PPW_', 'demo-c-sub14')])
    store = os.path.join(root, 'stores', 'site-c')
    print('$ python -m FederatedLearning.backfill <studies> --dry-run ...')
    backfill.run(folder, store_root=store, dry_run=True, args=STATED)
    print('$ ... without --dry-run')
    backfill.run(folder, store_root=store, args=STATED, crop=toy.demo_crop)
    print('$ ... and once more: nothing is added twice')
    backfill.run(folder, store_root=store, args=STATED, crop=toy.demo_crop)
    show_store(store)


def eval_store(root, rng):
    heading('NeuroFUS test set: 12 held-out subjects with region labels, schema 1.1')
    folder = os.path.join(root, 'runs', 'neurofus-test')
    for i, target in enumerate(HARD_REGIONS * 2 + ('LeftVIM', 'RightSTN'), start=1):
        subject = 'demo-test-sub{:02d}'.format(i)
        hard = target in HARD_REGIONS
        # In the toy data, the hard targets sit behind a thicker, more oblique skull
        skull_mm = rng.uniform(2.5, 3.5) if hard else rng.uniform(1.5, 3.0)
        tilt = rng.choice([-1, 1]) * rng.uniform(0.4,
                                                 0.6) if hard else rng.uniform(-0.3, 0.3)
        toy.write_run(os.path.join(folder, subject),
                      '{}_{}_Single_250kHz_9PPW_'.format(subject, target), rng, skull_mm, tilt)
    store = os.path.join(root, 'stores', 'eval-a')
    args = SimpleNamespace(**dict(vars(STATED), eval_regions=True))
    print('$ python -m FederatedLearning.backfill <test subjects> --dry-run ... --eval-regions')
    backfill.run(folder, store_root=store, dry_run=True, args=args)
    print('$ ... without --dry-run')
    backfill.run(folder, store_root=store, args=args, crop=toy.demo_crop)
    show_store(store)


def privacy_check(root):
    heading('Privacy check')
    needles = ['demo-a-sub', 'demo-b-sub', 'demo-c-sub', 'demo-test-sub', 'LeftVIM', 'Aug_2026',
               'DataForSim', '/runs/', root]
    found = []
    for manifest in glob.glob(os.path.join(root, 'stores', '*', 'v1', 'manifest.jsonl')):
        with open(manifest) as f:
            text = f.read()
        found += [n for n in needles if n in text]
    lines = sum(1 for m in glob.glob(os.path.join(root, 'stores', '*', 'v1', 'manifest.jsonl'))
                for _ in open(m))
    if found:
        raise SystemExit('Found {} in a manifest'.format(sorted(set(found))))
    print('{} manifest lines checked: no subject ID, target name, file name or path.'.format(lines))
    with open(glob.glob(os.path.join(root, 'stores', 'eval-a', 'v1', 'manifest.jsonl'))[0]) as f:
        entry = json.loads(f.readline())
    print('One test-set line, as Starfish sees it:')
    print('  ' + json.dumps({k: entry[k] for k in ('sample_id', 'schema_version', 'bucket_hz',
                                                   'tx_system', 'ct_type', 'group_id', 'split',
                                                   'source', 'region')}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--root', default='/demo-data')
    parser.add_argument('--reset', action='store_true',
                        help='delete the demo data and start again')
    args = parser.parse_args(argv)
    manifests = [os.path.join(args.root, 'stores', s,
                              'v1', 'manifest.jsonl') for s in STORES]
    if args.reset:
        for sub in ('runs', 'stores'):
            shutil.rmtree(os.path.join(args.root, sub), ignore_errors=True)
    elif all(os.path.isfile(m) for m in manifests):
        print('Demo stores exist, keeping them. make babelbrain-demo-stores writes them again.')
        return 0
    rng = np.random.default_rng(2026)
    site_a(args.root, rng)
    site_b(args.root, rng)
    site_c(args.root, rng)
    eval_store(args.root, rng)
    privacy_check(args.root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
