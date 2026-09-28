"""
Run one federated BabelBrainFno training on the demo stores and narrate it.

    python run_demo.py [--rounds 3]

Runs in the controller image through ``make babelbrain-demo-run``, next to
the demo stack. It talks to the router's API only, as the sites do: it
creates a project that sites A, B and C join, starts a run, prints each
site's state as it changes, and at the end shows per round what each site
trained and what the evaluation gate decided, the region breakdown of the
last round and the approved models in the registry.
"""

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile

import requests

ROUTER = os.environ.get('DEMO_ROUTER', 'http://router:8000/starfish/api/v1')
AUTH = ('admin', '1234')
SITES = {
    'A': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
    'B': 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
    'C': 'cccccccc-cccc-cccc-cccc-cccccccccccc',
}
SITE_NAMES = {'A': 'NeuroFUS, coordinator', 'B': 'lab B', 'C': 'lab C'}
REGIONS = ('P7', 'P8', 'PO7', 'TP7', 'TP8', 'other')
TIMEOUT_S = 900


def api(method, path, **kwargs):
    return requests.request(method, ROUTER + path, auth=AUTH, timeout=60, **kwargs)


def list_all(path):
    items, url = [], ROUTER + path
    while url:
        response = requests.get(url, auth=AUTH, timeout=30)
        response.raise_for_status()
        page = response.json()
        if isinstance(page, list):
            return page
        items.extend(page['results'])
        url = page['next']
    return items


def ensure_site(name, uid):
    for site in list_all('/sites/'):
        if site['uid'] == uid:
            return site['id']
    response = api('POST', '/sites/', json={
        'uid': uid, 'name': 'demo-site-' + name.lower(), 'description': SITE_NAMES[name]})
    response.raise_for_status()
    return ensure_site(name, uid)


def project_by_name(name):
    return next((p for p in list_all('/projects/') if p['name'] == name), None)


def runs_of(project_id, batch):
    response = api('GET', '/runs/detail/', params={
        'batch': batch, 'project': project_id, 'site_uid': SITES['A']})
    response.raise_for_status()
    return response.json()['runs']


def site_of(run):
    return next(n for n, uid in SITES.items() if run['site_uid'] == uid)


def round_of(run):
    tasks = run.get('tasks') or []
    try:
        return tasks[run['cur_seq'] - 1]['config']['current_round']
    except (IndexError, KeyError, TypeError):
        return None


def run_files(run_id, round_seq):
    """The logs.txt and eval_report.json a run uploaded for a round, as text.

    The router stores them as <run>-<task>-<round>-logs.txt and so on.
    """
    response = api('GET', '/runs-action/download/', params={
        'run': run_id, 'type': 'logs', 'task_seq': 1, 'round_seq': round_seq})
    if response.status_code != 200:
        return {}
    files = {}
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for name in zf.namelist():
            for kind in ('logs.txt', 'eval_report.json'):
                if name.endswith(kind):
                    files[kind] = zf.read(name).decode('utf-8', 'replace')
    return files


def watch(project_id, batch, rounds):
    start, last = time.time(), None
    while time.time() - start < TIMEOUT_S:
        runs = sorted(runs_of(project_id, batch), key=site_of)
        state = ', '.join('{} {} (round {})'.format(site_of(r), r['status'], round_of(r))
                          for r in runs)
        if state != last:
            print('[{:>4.0f} s] {}'.format(
                time.time() - start, state), flush=True)
            last = state
        if all(r['status'] in ('Success', 'Failed') for r in runs):
            return runs
        time.sleep(2)
    raise SystemExit('The run did not finish in {} s'.format(TIMEOUT_S))


def fmt(scores, metric):
    value = ((scores or {}).get(metric) or {}).get('mean')
    return '   n/a' if value is None else '{:6.2f}'.format(value)


def report(runs, rounds, project_id):
    by_site = {site_of(r): r for r in runs}
    last_report = None
    for rnd in range(1, rounds + 1):
        print('\nRound {}'.format(rnd))
        for name in sorted(by_site):
            files = run_files(by_site[name]['id'], rnd)
            log = files.get('logs.txt', '')
            start = re.search(
                r'Round {} starts from (?:global|seed) model (\S+)'.format(rnd), log)
            if name == 'A' and start:
                print('  every site starts from global model {}'.format(
                    start.group(1)))
            losses = re.findall(r'Epoch \d+ of \d+: loss ([\d.]+)', log)
            trained = re.search(
                r'Round {}: (\d+) train samples'.format(rnd), log)
            if trained and losses:
                print('  site {}: trained on its {} samples, loss {} to {} over {} epochs'.format(
                    name, trained.group(1), losses[0], losses[-1], len(losses)))
            if 'eval_report.json' in files:
                gate = json.loads(files['eval_report.json'])
                last_report = gate
                verdict = 'ACCEPTED, becomes the new global model' if gate['accepted'] else \
                    'REJECTED, the previous global model stays: ' + \
                    '; '.join(gate['reasons'])
                print('  gate at site A, on {} held-out samples: {}'.format(
                    gate['eval_samples'], verdict))
                for metric, label in (('rel_l2_pct', 'relative l2 error, %'),
                                      ('peak_distance_mm', 'peak distance, mm'),
                                      ('ssim', 'SSIM')):
                    print('    {:<22} current {}  candidate {}'.format(
                        label, fmt(gate['current'], metric), fmt(gate['candidate'], metric)))
    if last_report and last_report.get('region_breakdown'):
        print('\nRegion breakdown, last round, from the test set\'s schema 1.1 labels')
        print('  {:<6} {:>3}  {:>18}  {:>18}'.format('region', 'n', 'rel l2 %, cur > new',
                                                     'peak mm, cur > new'))
        for region in REGIONS:
            row = last_report['region_breakdown'].get(region)
            if not row:
                continue
            print('  {:<6} {:>3}  {} > {}      {} > {}'.format(
                region, row['candidate']['n'],
                fmt(row['current'], 'rel_l2_pct'), fmt(
                    row['candidate'], 'rel_l2_pct'),
                fmt(row['current'], 'peak_distance_mm'), fmt(row['candidate'], 'peak_distance_mm')))
    listing = api('GET', '/registry/', params={'bucket_hz': 250000})
    models = [m for m in listing.json() if m.get(
        'project') == project_id] if listing.ok else []
    if models:
        print('\nModel registry: the approved 250 kHz models of this run, newest first')
        for model in models:
            print('  {}  from round {}, parent {}, sha256 {}...'.format(
                model['version'], model['round_seq'], model.get(
                    'parent') or 'none',
                model['sha256'][:12]))
        print('  A lab downloads the newest one from the registry; rejected rounds never get there.')
    else:
        print('\nModel registry: nothing new; the gate rejected every round of this run')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=4,
                        help='local epochs per round')
    parser.add_argument('--peak-margin-mm', type=float, default=None,
                        help='gate margin on the mean peak distance; default 0, answer T8')
    args = parser.parse_args(argv)
    try:
        api('GET', '/sites/').raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(
            'Router not reachable, run make babelbrain-demo-up first: {}'.format(e))

    site_ids = {name: ensure_site(name, uid) for name, uid in SITES.items()}
    name = 'BabelBrain FL demo {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'))
    tasks = [{'seq': 1, 'model': 'BabelBrainFno', 'config': {
        'total_round': args.rounds, 'current_round': 1,
        'data_source': {'type': 'babelbrain_store', 'bucket_hz': 250000},
        'min_samples': 12, 'local_epochs': args.epochs}}]
    if args.peak_margin_mm is not None:
        tasks[0]['config']['gate'] = {'enabled': True,
                                      'max_peak_distance_increase_mm': args.peak_margin_mm}
    for site in ('A', 'B', 'C'):
        response = api('POST', '/projects/', json={
            'name': name, 'description': 'BabelBrain FL demo on toy data',
            'site': site_ids[site], 'tasks': tasks})
        response.raise_for_status()
    project = project_by_name(name)
    print('Project "{}": site A coordinates, B and C take part. Model: the stand-in FNO, '
          '{} rounds of {} local epochs at 250 kHz.'.format(name, args.rounds, args.epochs))
    margin = 0.0 if args.peak_margin_mm is None else args.peak_margin_mm
    print('Gate: a new model must not raise the mean relative l2 error at all, or the mean peak '
          'distance by more than {} mm.'.format(margin))
    api('POST', '/runs', json={'project': project['id']}).raise_for_status()
    batch = project_by_name(name)['batch']
    print('Run started. No site uploads data; each reads its own BabelBrain sample store.\n')
    runs = watch(project['id'], batch, args.rounds)
    report(runs, args.rounds, project['id'])
    print('\nOnly model weights moved: each site sent its update to the model, never a sample.')
    print('Site pages: http://localhost:8001/controller/ for A, 8002 for B, 8003 for C.')
    return 0 if all(r['status'] == 'Success' for r in runs) else 1


if __name__ == '__main__':
    sys.exit(main())
