"""
End-to-end test of starfish-site without Docker, for Windows, macOS and Linux.

    python e2e_without_docker.py --router-python PY --work DIR [--rounds 2] [--epochs 1]

Run it with the Python of a starfish-site install. It runs the router of this
checkout with PY, a Python that has the router's packages, against the
PostgreSQL that DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD,
DATABASE_HOST and DATABASE_PORT name. Then it writes synthetic stores,
enrols three sites, checks site A with a training run on its device, starts
the sites, runs a project through e2e_run.py and stops each site with
starfish-site stop. Site C trains on the CPU. Everything goes into DIR, and
the exit status is 0 when every run succeeds and every site stops cleanly.
"""

import argparse
import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTER_DIR = HERE.parent.parent / 'router'
PORT = 18100
API = 'http://127.0.0.1:{}/starfish/api/v1'.format(PORT)
AUTH = 'Basic ' + base64.b64encode(b'admin:1234').decode()
SITES = (('a', 'e2e A'), ('b', 'e2e B'), ('c', 'e2e C'))


def say(text):
    print('[e2e] ' + text, flush=True)


def site_command(*args):
    return [sys.executable, '-m', 'starfish_site'] + [str(a) for a in args]


def run(command, **kwargs):
    say(' '.join(str(c) for c in command))
    return subprocess.run(command, **kwargs)


def wait_for_port(port, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return True
        time.sleep(1)
    return False


def new_code(note):
    request = urllib.request.Request(API + '/enrolment-codes/', method='POST',
                                     data=json.dumps({'note': note}).encode(),
                                     headers={'Authorization': AUTH, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)['code']


def show_tail(path, lines=40):
    try:
        text = Path(path).read_text(errors='replace').splitlines()
    except FileNotFoundError:
        return
    say('last lines of {}'.format(path))
    for line in text[-lines:]:
        print('    ' + line, flush=True)


def start_router(router_python, work):
    env = dict(os.environ, SECRET_KEY=secrets.token_hex(24), DEBUG='True',
               DJANGO_SUPERUSER_PASSWORD='1234', STARFISH_DISABLE_AGENTS='1',
               MEDIA_ROOT=str(work / 'router-files'))
    run([router_python, 'manage.py', 'migrate', '--no-input'], cwd=ROUTER_DIR, env=env, check=True)
    # Fails harmlessly when the account is there from an earlier run
    run([router_python, 'manage.py', 'createsuperuser', '--noinput', '--username', 'admin',
         '--email', 'admin@e2e.local'], cwd=ROUTER_DIR, env=env)
    log = open(work / 'router.log', 'wb')
    router = subprocess.Popen([router_python, 'manage.py', 'runserver', '127.0.0.1:{}'.format(PORT),
                               '--noreload'], cwd=ROUTER_DIR, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    if not wait_for_port(PORT, 120):
        router.kill()
        show_tail(work / 'router.log')
        sys.exit('The router did not start')
    return router, log


def write_stores(stores):
    module = 'starfish.controller.tasks.babel_brain_fno.synthetic'
    for seed, (x, _) in enumerate(SITES, start=1):
        run([sys.executable, '-m', module, stores / ('lab-' + x), '--groups', '5', '--per-group', '4',
             '--val-groups', '1', '--seed', str(seed)], check=True)
    run([sys.executable, '-m', module, stores / 'test-set', '--groups', '6', '--per-group', '2',
         '--val-groups', '0', '--seed', '4', '--regions'], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--router-python', required=True, help='Python with the router\'s packages')
    p.add_argument('--work', required=True, help='folder for stores, site folders and logs')
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--epochs', type=int, default=1)
    args = p.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    stores = work / 'stores'
    homes = {x: work / ('site-' + x) for x, _ in SITES}
    problems = []

    router, router_log = start_router(args.router_python, work)
    sites, logs = {}, []
    try:
        write_stores(stores)
        for x, name in SITES:
            extra = []
            if x == 'a':
                extra = ['--eval-store', stores / 'test-set']
            elif x == 'c':
                extra = ['--device', 'cpu']
            run(site_command('--home', homes[x], 'enrol', '--router', API, '--code', new_code(name),
                             '--name', name, '--store', stores / ('lab-' + x), *extra), check=True)
        # A full check trains on site A's own device, the GPU of a Mac where MPS is there
        if run(site_command('--home', homes['a'], 'check')).returncode != 0:
            problems.append('check of site A failed')
        for x, _ in SITES:
            log = open(work / ('site-' + x + '.log'), 'wb')
            logs.append(log)
            sites[x] = subprocess.Popen(site_command('--home', homes[x], 'start'), stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        result = run([sys.executable, HERE / 'e2e_run.py', API, str(args.rounds), str(args.epochs)]
                     + [name for _, name in SITES])
        if result.returncode != 0:
            problems.append('the federated run did not succeed')
        for x, process in sites.items():
            if process.poll() is not None:
                problems.append('site {} ended early with status {}'.format(x, process.returncode))
                continue
            if run(site_command('--home', homes[x], 'stop', '--wait', '60')).returncode != 0:
                problems.append('starfish-site stop did not stop site {}'.format(x))
            try:
                status = process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                problems.append('site {} kept running after stop'.format(x))
                continue
            if status != 0:
                problems.append('site {} ended with status {}'.format(x, status))
    finally:
        for process in sites.values():
            if process.poll() is None:
                process.kill()
        for log in logs:
            log.close()
        router.terminate()
        try:
            router.wait(20)
        except subprocess.TimeoutExpired:
            router.kill()
        router_log.close()

    if problems:
        for x, _ in SITES:
            show_tail(work / ('site-' + x + '.log'))
        show_tail(work / 'router.log')
        for problem in problems:
            say('FAILED, ' + problem)
        return 1
    say('passed, {} rounds of {} epochs on {} with three sites'.format(args.rounds, args.epochs, sys.platform))
    return 0


if __name__ == '__main__':
    sys.exit(main())
