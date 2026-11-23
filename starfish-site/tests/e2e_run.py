"""
Create a project for the sites named on the command line, start a run and follow it.

    python3 e2e_run.py API ROUNDS EPOCHS COORDINATOR [PARTICIPANT ...]

Uses the workbench superuser and the standard library only. Exits 0 when
every run of the batch ends in Success.
"""
import base64
import json
import sys
import time
import urllib.request

API, ROUNDS, EPOCHS, SITES = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
AUTH = 'Basic ' + base64.b64encode(b'admin:1234').decode()
TIMEOUT_S = 1200


def call(method, path, body=None, query=''):
    url = path if path.startswith('http') else API + path + query
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={
        'Authorization': AUTH, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read()
    return json.loads(text) if text else None


def all_of(path):
    items, url = [], API + path
    while url:
        page = call('GET', url)
        items += page['results']
        url = page['next']
    return items


def main():
    sites = {s['name']: s for s in all_of('/sites/')}
    name = 'starfish-site e2e {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'))
    tasks = [{'seq': 1, 'model': 'BabelBrainFno', 'config': {
        'total_round': ROUNDS, 'current_round': 1,
        'data_source': {'type': 'babelbrain_store', 'bucket_hz': 250000},
        'min_samples': 12, 'local_epochs': EPOCHS}}]
    for site in SITES:
        call('POST', '/projects/', {'name': name, 'description': 'starfish-site e2e',
                                    'site': sites[site]['id'], 'tasks': tasks})
    project = next(p for p in all_of('/projects/') if p['name'] == name)
    call('POST', '/runs', {'project': project['id']})
    batch = next(p for p in all_of('/projects/') if p['id'] == project['id'])['batch']
    query = '?project={}&batch={}&site_uid={}'.format(project['id'], batch, sites[SITES[0]]['uid'])
    started, last, runs = time.time(), None, []
    while time.time() - started < TIMEOUT_S:
        runs = call('GET', '/runs/detail/', query=query).get('runs', [])
        state = sorted({(r['status'], r['tasks'][0]['config'].get('current_round')) for r in runs})
        if state != last:
            print('{:5.0f} s  {}'.format(time.time() - started, state), flush=True)
            last = state
        if runs and all(r['status'] in ('Success', 'Failed') for r in runs):
            break
        time.sleep(3)
    ok = bool(runs) and all(r['status'] == 'Success' for r in runs)
    print('project {} batch {} {} after {:.0f} s'.format(
        project['id'], batch, 'succeeded' if ok else 'did not succeed', time.time() - started))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
