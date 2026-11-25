"""Site folder, site.env and enrolment."""

import contextlib
import io
import logging
import os
import shutil
import stat
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from starfish_site import cli
from starfish_site import config as C


class ConfigTest(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def test_settings_survive_a_round_trip_and_stay_private(self):
        values = {'ROUTER_TOKEN': 'secret-token', 'SITE_UID': 'u', 'EXTRA': 'x y',
                  'BABELBRAIN_FL_STORE': '/data/My Store', 'EMPTY': ''}
        path = C.write_config(self.home, values)
        if os.name != 'nt':
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        read = C.read_config(self.home)
        self.assertEqual(read, {k: v for k, v in values.items() if v})
        keys = [line.split('=')[0] for line in path.read_text().splitlines()
                if line and not line.startswith('#')]
        self.assertEqual(keys, ['SITE_UID', 'ROUTER_TOKEN', 'BABELBRAIN_FL_STORE', 'EXTRA'])

    def test_a_path_with_letters_beyond_ascii_survives(self):
        # Windows writes its own code page unless told UTF-8, and python-dotenv reads UTF-8
        values = {'SITE_UID': 'u', 'BABELBRAIN_FL_STORE': '/data/José Müller/Ergebnisse 東京'}
        C.write_config(self.home, values)
        self.assertEqual(C.read_config(self.home), values)

    def test_a_value_with_a_quote_is_refused(self):
        with self.assertRaises(ValueError):
            C.write_config(self.home, {'SITE_UID': "it's"})

    def test_a_missing_file_reads_as_no_settings(self):
        self.assertEqual(C.read_config(self.home), {})

    def test_the_api_path_is_added_to_a_bare_host(self):
        self.assertEqual(C.api_url('https://10.0.0.5:8443/'), 'https://10.0.0.5:8443/starfish/api/v1')
        self.assertEqual(C.api_url('http://host:8000/starfish/api/v1'),
                         'http://host:8000/starfish/api/v1')
        with self.assertRaises(ValueError):
            C.api_url('10.0.0.5:8443')

    def test_the_home_comes_from_the_option_then_the_environment(self):
        # Paths, which Windows writes with backslashes
        with patch.dict(os.environ, {C.HOME_ENV: '/from/env'}):
            self.assertEqual(C.site_home('/from/option'), C.Path('/from/option'))
            self.assertEqual(C.site_home(), C.Path('/from/env'))
        with patch.dict(os.environ, {}, clear=True), patch('pathlib.Path.home', return_value=C.Path('/h')):
            self.assertEqual(C.site_home(), C.Path('/h') / '.starfish-site')

    def test_apply_config_skips_empty_values(self):
        environ = {}
        C.apply_config({'A': '1', 'B': ''}, environ)
        self.assertEqual(environ, {'A': '1'})


def answer(status, body=None):
    response = MagicMock(status_code=status, text=str(body))
    response.json.return_value = body
    return response


class EnrolTest(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.store = tempfile.mkdtemp()
        self.ca = os.path.join(self.store, 'ca.pem')
        with open(self.ca, 'w') as f:
            f.write('-----BEGIN CERTIFICATE-----\n')

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)
        shutil.rmtree(self.store, ignore_errors=True)

    def enrol(self, *extra, status=201, body=None):
        body = body or {'site': 7, 'uid': 'from-router', 'token': 'tok', 'project': 3}
        argv = ['--home', self.home, 'enrol', '--router', 'https://router.lan:8443', '--code', 'C0DE',
                '--name', 'Lab B', '--store', self.store, '--ca-cert', self.ca, *extra]
        out = io.StringIO()
        with patch('requests.post', return_value=answer(status, body)) as post, \
                contextlib.redirect_stdout(out):
            code = cli.main(argv)
        return code, post, out.getvalue()

    def test_enrolment_writes_the_token_and_the_settings(self):
        code, post, out = self.enrol('--device', 'cpu')
        self.assertEqual(code, 0)
        self.assertEqual(post.call_args.args[0], 'https://router.lan:8443/starfish/api/v1/sites/enrol/')
        sent = post.call_args.kwargs['json']
        self.assertEqual((sent['code'], sent['name']), ('C0DE', 'Lab B'))
        ca = os.path.join(self.home, C.CA_NAME)
        self.assertEqual(post.call_args.kwargs['verify'], ca)
        values = C.read_config(self.home)
        self.assertEqual(values['ROUTER_TOKEN'], 'tok')
        self.assertEqual(values['SITE_UID'], 'from-router')
        self.assertEqual(values['ROUTER_URL'], 'https://router.lan:8443/starfish/api/v1')
        self.assertEqual(values['REQUESTS_CA_BUNDLE'], ca)
        self.assertEqual(values['BABELBRAIN_FL_STORE'], os.path.abspath(self.store))
        self.assertEqual(values['BABELBRAIN_FL_DEVICE'], 'cpu')
        self.assertEqual(values['STARFISH_DISABLE_AGENTS'], '1')
        self.assertTrue(os.path.isfile(ca))
        self.assertIn('project 3', out)

    def test_an_enrolled_site_needs_force_to_enrol_again(self):
        self.enrol()
        with self.assertRaises(SystemExit) as stop:
            self.enrol()
        self.assertIn('--force', str(stop.exception))
        code, post, _ = self.enrol('--force')
        self.assertEqual(code, 0)
        self.assertEqual(post.call_args.kwargs['json']['uid'], 'from-router')

    def test_a_refused_code_says_why(self):
        with self.assertRaises(SystemExit) as stop:
            self.enrol(status=403, body='Invalid')
        self.assertIn('refused the code', str(stop.exception))
        self.assertEqual(C.read_config(self.home), {})


class CheckWithoutSettingsTest(unittest.TestCase):

    def test_check_covers_torch_only_before_enrolment(self):
        home = tempfile.mkdtemp()
        out = io.StringIO()
        needs_settings = MagicMock(side_effect=AssertionError('needs settings'))
        try:
            with contextlib.redirect_stdout(out), \
                    patch('starfish_site.checks.check_settings', needs_settings), \
                    patch('starfish_site.checks.check_router', needs_settings), \
                    patch('starfish_site.checks.check_stores', needs_settings), \
                    patch('starfish_site.checks.check_torch', return_value=None) as torch_check, \
                    patch('starfish_site.checks.check_training') as training_check:
                code = cli.main(['--home', home, 'check'])
        finally:
            shutil.rmtree(home, ignore_errors=True)
        self.assertEqual(code, 0)
        torch_check.assert_called_once()
        training_check.assert_called_once()
        self.assertIn('torch and training only until this site is enrolled', out.getvalue())


class LockTest(unittest.TestCase):

    def test_one_site_folder_runs_one_process(self):
        home = tempfile.mkdtemp()
        try:
            self.assertIsNone(cli.running_pid(home))
            lock = cli.hold_site(home)
            self.assertEqual(cli.running_pid(home), os.getpid())
            with self.assertRaises(SystemExit) as stop:
                cli.hold_site(home)
            self.assertIn('runs already, as process {}'.format(os.getpid()), str(stop.exception))
            lock.close()
            self.assertIsNone(cli.running_pid(home))
        finally:
            shutil.rmtree(home, ignore_errors=True)


class StopTest(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)

    def stop(self, now=False, wait=5.0):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(['--home', self.home, 'stop', '--wait', str(wait)] + (['--now'] if now else []))
        return code, out.getvalue()

    def test_a_site_that_is_not_running_needs_no_stop(self):
        self.assertEqual(self.stop(), (0, 'This site is not running.\n'))
        self.assertIsNone(cli.stop_request(self.home))

    def test_stop_asks_through_the_request_file_and_waits_for_the_lock(self):
        lock = cli.hold_site(self.home)
        seen = []

        def site():
            # The running site's loop, which looks for a request every second
            while cli.stop_request(self.home) is None:
                time.sleep(0.05)
            seen.append(cli.stop_request(self.home))
            cli.clear_stop_request(self.home)
            lock.close()

        thread = threading.Thread(target=site)
        thread.start()
        code, out = self.stop()
        thread.join()
        self.assertEqual(code, 0)
        self.assertEqual(out, 'Stopped process {}.\n'.format(os.getpid()))
        self.assertEqual(seen, ['step'])

    def test_now_asks_to_cut_the_step_short(self):
        lock = cli.hold_site(self.home)
        self.addCleanup(lock.close)
        code, out = self.stop(now=True, wait=0.6)
        self.assertEqual(code, 1)
        self.assertIn('still finishing the step in progress', out)
        self.assertEqual(cli.stop_request(self.home), 'now')
        cli.clear_stop_request(self.home)
        self.assertIsNone(cli.stop_request(self.home))


def messages(lines):
    return [line.split(': ', 1)[1] for line in lines]


class LoggingTest(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()
        root = logging.getLogger()
        self.before = list(root.handlers)
        self.addCleanup(shutil.rmtree, self.home, True)
        self.addCleanup(root.setLevel, root.level)

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if handler not in self.before:
                root.removeHandler(handler)
                handler.close()

    def task_lines(self, verbose):
        screen = io.StringIO()
        with contextlib.redirect_stdout(screen):
            log_file = cli.setup_logging(self.home, verbose)
        # A task logger keeps its own level, as the controller's tasks set it
        task = logging.getLogger('logger-test-{}'.format(verbose))
        task.setLevel(logging.DEBUG)
        task.debug('Init logger for run 1')
        task.info('Round 1 starts')
        return messages(screen.getvalue().splitlines()), messages(log_file.read_text().splitlines())

    def test_task_debug_lines_stay_out_without_verbose(self):
        screen, log_file = self.task_lines(verbose=False)
        self.assertEqual(screen, ['Round 1 starts'])
        self.assertEqual(log_file, ['Round 1 starts'])

    def test_verbose_shows_task_debug_lines(self):
        screen, log_file = self.task_lines(verbose=True)
        self.assertEqual(screen, ['Init logger for run 1', 'Round 1 starts'])
        self.assertEqual(log_file, ['Init logger for run 1', 'Round 1 starts'])


if __name__ == '__main__':
    unittest.main()
