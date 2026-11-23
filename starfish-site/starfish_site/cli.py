"""
The ``starfish-site`` command.

    starfish-site enrol --router URL --code CODE --name NAME [--store PATH] ...
    starfish-site check [--quick]
    starfish-site start [--verbose]
    starfish-site stop [--now]

``--home`` before the command picks the site folder, ``~/.starfish-site`` by
default. Each site on one machine needs its own folder.
"""

import argparse
import atexit
import fcntl
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import time
import uuid

from pathlib import Path

from starfish_site import __version__
from starfish_site import config as C

logger = logging.getLogger('starfish_site')


def _site_status():
    from starfish.controller import site_status_task
    return site_status_task


def _quiet_exit():
    """Keep a command that does not run the site from reporting it offline at exit."""
    atexit.unregister(_site_status().report_exit)


def load(home, require=('SITE_UID', 'ROUTER_URL')):
    """Read site.env into the environment and return it, or exit with a message."""
    values = C.read_config(home)
    missing = [k for k in require if not values.get(k)]
    if missing:
        sys.exit('{} is not set in {}. Enrol first, see starfish-site enrol --help.'.format(
            ' and '.join(missing), C.config_path(home)))
    C.apply_config(values)
    return values


# enrol

def enrol(args):
    import requests

    home = C.site_home(args.home)
    values = C.read_config(home)
    if values.get('ROUTER_TOKEN') and not args.force:
        sys.exit('This site is enrolled already, as {}. Pass --force to enrol it again '
                 'with a new code.'.format(values.get('SITE_UID')))
    try:
        url = C.api_url(args.router)
    except ValueError as e:
        sys.exit(str(e))
    home.mkdir(parents=True, exist_ok=True)
    verify = True
    if args.ca_cert:
        ca = home / C.CA_NAME
        if os.path.abspath(args.ca_cert) != str(ca):
            shutil.copyfile(args.ca_cert, ca)
        verify = str(ca)
    elif values.get('REQUESTS_CA_BUNDLE'):
        verify = values['REQUESTS_CA_BUNDLE']
    uid = values.get('SITE_UID') or str(uuid.uuid4())
    body = {'code': args.code, 'uid': uid, 'name': args.name,
            'description': args.description or ''}
    try:
        response = requests.post(url + '/sites/enrol/', json=body, timeout=30, verify=verify)
    except requests.exceptions.SSLError as e:
        sys.exit('The router\'s HTTPS certificate is not trusted: {}\n'
                 'Pass the CA file from the router\'s admin with --ca-cert.'.format(e))
    except requests.RequestException as e:
        sys.exit('The router at {} cannot be reached: {}'.format(url, e))
    if response.status_code == 403:
        sys.exit('The router refused the code. It may be mistyped, used or expired.')
    if response.status_code != 201:
        sys.exit('Enrolment failed with {}: {}'.format(response.status_code, response.text[:300]))
    answer = response.json()
    values.update({'SITE_UID': answer['uid'], 'ROUTER_URL': url, 'ROUTER_TOKEN': answer['token'],
                   'STARFISH_DISABLE_AGENTS': values.get('STARFISH_DISABLE_AGENTS') or '1'})
    values.pop('ROUTER_USERNAME', None)
    values.pop('ROUTER_PASSWORD', None)
    if verify is not True:
        values['REQUESTS_CA_BUNDLE'] = verify
    for key, value in (('BABELBRAIN_FL_STORE', args.store), ('BABELBRAIN_FL_EVAL_STORE', args.eval_store),
                       ('BABELBRAIN_FL_DEVICE', args.device)):
        if value:
            values[key] = os.path.abspath(value) if key.endswith('STORE') else value
    path = C.write_config(home, values)
    print('Enrolled as site "{}", {}, router id {}.'.format(args.name, answer['uid'], answer['site']))
    if answer.get('project'):
        print('The code also added this site to project {}.'.format(answer['project']))
    print('Settings are in {}. Next run starfish-site check.'.format(path))


# check

def check(args):
    from starfish_site.checks import run_checks

    home = C.site_home(args.home)
    values = load(home, require=())
    print('starfish-site {} checks {}'.format(__version__, C.config_path(home)))
    if not values:
        print('\nNo settings found. The check covers torch and training only until this site '
              'is enrolled, see starfish-site enrol --help.')
    _quiet_exit()
    # The check reports a plain HTTP router itself
    logging.getLogger('starfish.controller.router_auth').setLevel(logging.ERROR)
    report = run_checks(values, quick=args.quick)
    return 1 if report.failed else 0


# start and stop

PID_NAME = 'starfish-site.pid'


def hold_site(home):
    """
    Lock the site folder for this process and write its PID there.

    Two processes on one site would each act on its runs, so a second
    start exits. The lock goes with the process, even when it is killed.
    """
    handle = open(Path(home) / PID_NAME, 'a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        other = handle.read().strip()
        handle.close()
        sys.exit('This site runs already{}. Stop it first with starfish-site stop.'.format(
            ', as process ' + other if other else ''))
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def running_pid(home):
    """The PID of the process that runs this site, or None when none does."""
    path = Path(home) / PID_NAME
    if not path.is_file():
        return None
    with open(path) as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            text = handle.read().strip()
            return int(text) if text.isdigit() else None
        fcntl.flock(handle, fcntl.LOCK_UN)
        return None


def stop(args):
    home = C.site_home(args.home)
    pid = running_pid(home)
    if pid is None:
        print('This site is not running.')
        return 0
    os.kill(pid, signal.SIGTERM)
    if args.now:
        time.sleep(0.5)
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        if running_pid(home) is None:
            print('Stopped process {}.'.format(pid))
            return 0
        time.sleep(0.5)
    print('Process {} is still finishing the step in progress and stops when that ends. '
          'starfish-site stop --now cuts it short.'.format(pid))
    return 1


def setup_logging(home, verbose):
    folder = C.log_folder(home)
    folder.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(name)s: %(message)s')
    to_file = logging.handlers.RotatingFileHandler(folder / 'starfish-site.log',
                                                   maxBytes=10 * 2 ** 20, backupCount=5)
    to_screen = logging.StreamHandler(sys.stdout)
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    for handler in (to_file, to_screen):
        handler.setFormatter(formatter)
        # A task logger keeps its own DEBUG level, and its records skip the root's level
        handler.setLevel(level)
        root.addHandler(handler)
    for noisy in ('urllib3', 'matplotlib', 'h5py'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return folder / 'starfish-site.log'


def start(args):
    home = C.site_home(args.home)
    values = load(home)
    if not (values.get('ROUTER_TOKEN') or values.get('ROUTER_PASSWORD')):
        sys.exit('No ROUTER_TOKEN in {}. Enrol first.'.format(C.config_path(home)))
    lock = hold_site(home)
    log_file = setup_logging(home, args.verbose)

    # The controller code reads its settings when imported, so import it now
    from starfish.controller import run_dispatch
    from starfish.controller.file import file_utils
    from starfish_site.runner import SiteRunner

    local = C.local_folder(home)
    local.mkdir(parents=True, exist_ok=True)
    # Run files go to the site folder in place of the container's /starfish-controller/local
    file_utils.base_folder = str(local)
    status = _site_status()

    logger.info('starfish-site %s, site %s, router %s', __version__, values['SITE_UID'],
                values['ROUTER_URL'])
    logger.info('Store %s, device %s, run files in %s, log in %s',
                values.get('BABELBRAIN_FL_STORE', 'not set'),
                values.get('BABELBRAIN_FL_DEVICE', 'auto'), local, log_file)
    runner = SiteRunner(values['SITE_UID'], run_dispatch, status.report_alive)

    def on_signal(signum, frame):
        if runner.stopping.is_set():
            logger.warning('Stopping now, the step in progress is cut short')
            try:
                status.report_exit()
            finally:
                os._exit(130)
        if runner.current:
            logger.info('Stopping after the step in progress, run %s %s. '
                        'Press Ctrl-C again to stop at once.', *runner.current[:2])
        else:
            logger.info('Stopping')
        runner.stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    runner.start()
    logger.info('Polling the router. Press Ctrl-C to stop.')
    while not runner.join(timeout=1.0):
        pass
    logger.info('Stopped')
    lock.close()
    return 0


def parser():
    p = argparse.ArgumentParser(prog='starfish-site', description='A native Starfish-FL site.')
    p.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    p.add_argument('--home', help='site folder, default ~/.starfish-site or $' + C.HOME_ENV)
    sub = p.add_subparsers(dest='command', required=True)

    e = sub.add_parser('enrol', help='join the router with a code from its admin')
    e.add_argument('--router', required=True, help='router URL, for example https://10.0.0.5:8443')
    e.add_argument('--code', required=True, help='single-use enrolment code from the admin')
    e.add_argument('--name', required=True, help='name of this site, for example "Lab B"')
    e.add_argument('--description', help='a line about this site')
    e.add_argument('--store', help='this site\'s BabelBrain sample store')
    e.add_argument('--eval-store', help='coordinator only, the held-out test set')
    e.add_argument('--device', help='cuda, mps or cpu, to override the automatic choice')
    e.add_argument('--ca-cert', help='CA file that signed the router\'s certificate')
    e.add_argument('--force', action='store_true', help='enrol again, with a new code')
    e.set_defaults(func=enrol)

    c = sub.add_parser('check', help='check settings, router, stores, torch and training')
    c.add_argument('--quick', action='store_true', help='skip the training check')
    c.set_defaults(func=check)

    s = sub.add_parser('start', help='run this site until Ctrl-C')
    s.add_argument('--verbose', action='store_true', help='log debug messages too')
    s.set_defaults(func=start)

    t = sub.add_parser('stop', help='stop the running site after its step in progress')
    t.add_argument('--now', action='store_true', help='cut the step in progress short')
    t.add_argument('--wait', type=float, default=30, help='seconds to wait, default 30')
    t.set_defaults(func=stop)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
