"""
The ``starfish-site`` command.

    starfish-site enrol --router URL --code CODE --name NAME [--store PATH] ...
    starfish-site check [--quick]
    starfish-site start [--verbose]
    starfish-site stop [--now]
    starfish-site invite --router URL --user ADMIN --out FILE ...

``--home`` before the command picks the site folder, ``~/.starfish-site`` by
default. Each site on one machine needs its own folder. ``invite`` is for the
router's admin and writes the invitation file that Starfish Site opens.
"""

import argparse
import atexit
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

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

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

class EnrolError(Exception):
    """
    Enrolment did not happen.

    ``kind`` is ``enrolled``, ``url``, ``certificate``, ``unreachable``,
    ``refused`` or ``failed``, and the message is the command line's.
    """

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def enrol_site(home, router, code, name, description='', store=None, eval_store=None, device=None,
               ca_cert=None, force=False):
    """
    Enrol the site in ``home`` with a code from the router's admin and write its site.env.

    The command and Starfish Site both enrol through here. Returns the
    router's answer, with the site's ``uid``, its router id ``site`` and the
    ``project`` the code added it to, and the path of site.env.
    """
    import requests

    home = Path(home)
    values = C.read_config(home)
    if values.get('ROUTER_TOKEN') and not force:
        raise EnrolError('enrolled', 'This site is enrolled already, as {}. Pass --force to enrol it '
                         'again with a new code.'.format(values.get('SITE_UID')))
    try:
        url = C.api_url(router)
    except ValueError as e:
        raise EnrolError('url', str(e))
    home.mkdir(parents=True, exist_ok=True)
    verify = True
    if ca_cert:
        ca = home / C.CA_NAME
        if os.path.abspath(ca_cert) != os.path.abspath(ca):
            shutil.copyfile(ca_cert, ca)
        verify = str(ca)
    elif values.get('REQUESTS_CA_BUNDLE'):
        verify = values['REQUESTS_CA_BUNDLE']
    uid = values.get('SITE_UID') or str(uuid.uuid4())
    body = {'code': code, 'uid': uid, 'name': name, 'description': description or ''}
    try:
        response = requests.post(url + '/sites/enrol/', json=body, timeout=30, verify=verify)
    except requests.exceptions.SSLError as e:
        raise EnrolError('certificate', 'The router\'s HTTPS certificate is not trusted: {}\n'
                         'Pass the CA file from the router\'s admin with --ca-cert.'.format(e))
    except requests.RequestException as e:
        raise EnrolError('unreachable', 'The router at {} cannot be reached: {}'.format(url, e))
    if response.status_code == 403:
        raise EnrolError('refused', 'The router refused the code. It may be mistyped, used or expired.')
    if response.status_code != 201:
        raise EnrolError('failed', 'Enrolment failed with {}: {}'.format(
            response.status_code, response.text[:300]))
    answer = response.json()
    values.update({'SITE_UID': answer['uid'], 'ROUTER_URL': url, 'ROUTER_TOKEN': answer['token'],
                   'STARFISH_DISABLE_AGENTS': values.get('STARFISH_DISABLE_AGENTS') or '1'})
    values.pop('ROUTER_USERNAME', None)
    values.pop('ROUTER_PASSWORD', None)
    if verify is not True:
        values['REQUESTS_CA_BUNDLE'] = verify
    for key, value in (('BABELBRAIN_FL_STORE', store), ('BABELBRAIN_FL_EVAL_STORE', eval_store),
                       ('BABELBRAIN_FL_DEVICE', device)):
        if value:
            values[key] = os.path.abspath(value) if key.endswith('STORE') else value
    return answer, C.write_config(home, values)


def enrol(args):
    home = C.site_home(args.home)
    try:
        answer, path = enrol_site(home, args.router, args.code, args.name, args.description, args.store,
                                  args.eval_store, args.device, args.ca_cert, args.force)
    except EnrolError as e:
        sys.exit(str(e))
    print('Enrolled as site "{}", {}, router id {}.'.format(args.name, answer['uid'], answer['site']))
    if answer.get('project'):
        print('The code also added this site to project {}.'.format(answer['project']))
    print('Settings are in {}. Next run starfish-site check.'.format(path))


# invite

def invite(args):
    import getpass

    from starfish_site import invitation as I

    password = os.environ.get('ROUTER_PASSWORD') or getpass.getpass(
        'Password of {} on the router: '.format(args.user))
    try:
        found = I.create(args.router, args.user, password, project=args.project, note=args.note,
                         valid_hours=args.valid_hours, testbed=args.testbed, ca_cert=args.ca_cert)
        path = I.write(args.out, found)
    except (I.InvitationError, OSError) as e:
        sys.exit(str(e))
    print('Wrote {}, valid until {}.'.format(path, I.describe_time(found.expires_at)))
    print('Send it to the site, where Starfish Site opens it. Its code works once, so keep the file '
          'as private as a password until the site has used it.')


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
STOP_NAME = 'starfish-site.stop'
# Windows locks byte ranges and refuses to read a locked one, and the lock
# therefore sits past the PID to keep the PID readable
_LOCK_AT = 64


def _try_lock(handle, shared=False):
    """Lock an open PID file without waiting. False when another process holds it."""
    if os.name == 'nt':
        handle.seek(_LOCK_AT)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        finally:
            handle.seek(0)
        return True
    try:
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(handle):
    if os.name == 'nt':
        handle.seek(_LOCK_AT)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.seek(0)
    else:
        fcntl.flock(handle, fcntl.LOCK_UN)


def _read_pid(handle):
    handle.seek(0)
    text = handle.read(32).decode('ascii', 'replace').strip()
    return int(text) if text.isdigit() else None


def hold_site(home):
    """
    Lock the site folder for this process and write its PID there.

    Two processes on one site would each act on its runs, so a second
    start exits. The lock goes with the process, even when it is killed.
    """
    path = Path(home) / PID_NAME
    path.touch()
    # Unbuffered, so reads, writes and the lock all act on the file itself
    handle = open(path, 'r+b', buffering=0)
    if not _try_lock(handle):
        other = _read_pid(handle)
        handle.close()
        sys.exit('This site runs already{}. Stop it first with starfish-site stop.'.format(
            ', as process {}'.format(other) if other else ''))
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()).encode('ascii'))
    return handle


def running_pid(home):
    """The PID of the process that runs this site, or None when none does."""
    path = Path(home) / PID_NAME
    if not path.is_file():
        return None
    with open(path, 'rb', buffering=0) as handle:
        if not _try_lock(handle, shared=True):
            return _read_pid(handle)
        _unlock(handle)
        return None


def request_stop(home, now=False):
    """
    Ask the site that runs from ``home`` to stop.

    The site looks for the request file every second. Windows has no signal
    that another process can send to a Python handler, and the file works the
    same on every system.
    """
    (Path(home) / STOP_NAME).write_text('now' if now else 'step')


def stop_request(home):
    """``'now'``, ``'step'`` for a stop after the step in progress, or None."""
    try:
        text = (Path(home) / STOP_NAME).read_text().strip()
    except FileNotFoundError:
        return None
    except OSError:
        # Being written at this moment, the next look gets it
        return None
    return 'now' if text == 'now' else 'step'


def clear_stop_request(home):
    try:
        (Path(home) / STOP_NAME).unlink()
    except FileNotFoundError:
        pass


def stop(args):
    home = C.site_home(args.home)
    pid = running_pid(home)
    if pid is None:
        print('This site is not running.')
        return 0
    request_stop(home, now=args.now)
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
    to_file = logging.handlers.RotatingFileHandler(folder / 'starfish-site.log', encoding='utf-8',
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
    # A request left by a site that was killed before it read it
    clear_stop_request(home)
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
    if hasattr(signal, 'SIGBREAK'):
        # Ctrl-Break in a Windows console
        signal.signal(signal.SIGBREAK, on_signal)
    runner.start()
    logger.info('Polling the router. Press Ctrl-C to stop.')
    handled = None
    while not runner.join(timeout=1.0):
        request = stop_request(home)
        if request and request != handled:
            handled = request
            if not runner.stopping.is_set():
                on_signal(None, None)
            if request == 'now':
                on_signal(None, None)
    logger.info('Stopped')
    clear_stop_request(home)
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

    i = sub.add_parser('invite', help='for the router\'s admin, write an invitation file for a new site')
    i.add_argument('--router', required=True, help='router URL as the sites reach it')
    i.add_argument('--user', required=True, help='an admin user of the router, whose password '
                   'comes from $ROUTER_PASSWORD or a prompt')
    i.add_argument('--out', required=True,
                   help='the file to write, for example "Hospital A.starfish-invitation"')
    i.add_argument('--project', type=int, help='a project the new site joins on enrolment')
    i.add_argument('--testbed', default='', help='a name for the testbed, shown to the site')
    i.add_argument('--note', default='', help='a line for the site, such as who to contact')
    i.add_argument('--valid-hours', type=float, default=72, help='hours the code stays valid, default 72')
    i.add_argument('--ca-cert', help='CA file that signed the router\'s certificate, sent along')
    i.set_defaults(func=invite)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
