"""
Starfish Site, the desktop app.

The tests of the site's process, of the router's answers and of starting at
log on need no window. The tests of the windows run offscreen and are skipped
where PySide6 is not installed.
"""

import datetime
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from starfish_site import config as C
from starfish_site import invitation as I
from starfish_site.app import engine, status as S

try:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

UID = '0b7c3f8e-56a3-4c67-9a1b-2f5e8d1c4a90'


def enrolled_home(router='http://127.0.0.1:9/starfish/api/v1'):
    home = tempfile.mkdtemp()
    C.write_config(home, {'SITE_UID': UID, 'ROUTER_URL': router, 'ROUTER_TOKEN': 'tok',
                          'BABELBRAIN_FL_STORE': home, 'STARFISH_ADMIN_NOTE': 'Room 2.14'})
    return home


class EngineTest(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def test_commands_from_source_and_from_the_installed_app(self):
        self.assertEqual(engine.site_command(self.home, 'start'),
                         [sys.executable, '-m', 'starfish_site', '--home', self.home, 'start'])
        exe = os.path.join(self.home, 'Starfish Site.exe' if os.name == 'nt' else 'Starfish Site')
        with patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'executable', exe):
            command = engine.site_command(self.home, 'check')
            self.assertEqual(os.path.basename(command[0]),
                             'starfish-site.exe' if os.name == 'nt' else 'starfish-site')
            self.assertEqual(os.path.dirname(command[0]), self.home)
            self.assertEqual(engine.app_command('--background'), [exe, '--background'])

    def test_the_site_starts_in_a_process_of_its_own(self):
        site = engine.Site(self.home)
        popen = MagicMock()
        site.start(popen=popen)
        args, kwargs = popen.call_args
        self.assertEqual(args[0][-1], 'start')
        self.assertIs(kwargs['stdout'], subprocess.DEVNULL)
        self.assertEqual(kwargs['cwd'], self.home)
        self.assertEqual(kwargs['env']['PYINSTALLER_RESET_ENVIRONMENT'], '1')
        if os.name == 'nt':
            self.assertTrue(kwargs['creationflags'] & subprocess.CREATE_NO_WINDOW)
        else:
            self.assertTrue(kwargs['start_new_session'])
        self.assertTrue(site.stderr_path().is_file())

    def test_a_relative_site_folder_becomes_absolute(self):
        # The site's process starts in the folder and gets it as --home
        self.assertTrue(engine.Site('relative-folder').home.is_absolute())

    def test_settings_change_and_an_empty_value_drops_one(self):
        site = engine.Site(self.home)
        self.assertFalse(site.enrolled())
        site.update(SITE_UID=UID, ROUTER_URL='https://h/x', ROUTER_TOKEN='t', BABELBRAIN_FL_DEVICE='cpu')
        self.assertTrue(site.enrolled())
        site.update(BABELBRAIN_FL_DEVICE=None)
        self.assertNotIn('BABELBRAIN_FL_DEVICE', site.settings())

    def test_the_log_is_read_in_whole_lines_and_again_after_rotation(self):
        path = os.path.join(self.home, 'site.log')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('2026-11-26 11:01:45,120 INFO    starfish_site: Starfish one\npartial')
        lines, offset = engine.read_log(path)
        self.assertEqual(len(lines), 1)
        self.assertEqual(engine.parse_log_line(lines[0]),
                         ('2026-11-26', '11:01:45', 'INFO', 'starfish_site', 'Starfish one'))
        self.assertIsNone(engine.parse_log_line('Traceback (most recent call last):'))
        with open(path, 'a', encoding='utf-8') as f:
            f.write(' line\n')
        lines, offset = engine.read_log(path, offset)
        self.assertEqual(lines, ['partial line'])
        with open(path, 'w', encoding='utf-8') as f:
            f.write('new\n')
        self.assertEqual(engine.read_log(path, offset)[0], ['new'])
        self.assertEqual(engine.read_log(os.path.join(self.home, 'none.log')), ([], 0))

    def test_starting_at_log_on_on_linux_and_macos(self):
        command = ['/opt/Starfish Site/Starfish Site', '--background']
        for platform in ('linux', 'darwin'):
            with self.subTest(platform=platform):
                self.assertFalse(engine.autostart_enabled(platform, self.home))
                engine.set_autostart(True, command, platform, self.home)
                self.assertTrue(engine.autostart_enabled(platform, self.home))
                if platform == 'darwin':
                    with open(engine._mac_agent_path(self.home), 'rb') as f:
                        agent = plistlib.load(f)
                    self.assertEqual(agent['ProgramArguments'], command)
                    self.assertEqual(agent['Label'], engine.BUNDLE_ID)
                    self.assertTrue(agent['RunAtLoad'])
                else:
                    text = engine._linux_autostart_path(self.home).read_text()
                    self.assertIn('Exec="/opt/Starfish Site/Starfish Site" --background', text)
                engine.set_autostart(False, command, platform, self.home)
                self.assertFalse(engine.autostart_enabled(platform, self.home))
                engine.set_autostart(False, command, platform, self.home)

    @unittest.skipUnless(sys.platform == 'win32' and os.environ.get('CI'), 'changes the registry of this user')
    def test_starting_at_log_on_on_windows(self):
        engine.set_autostart(True, ['C:\\Program Files\\Starfish Site\\Starfish Site.exe', '--background'])
        try:
            self.assertTrue(engine.autostart_enabled())
        finally:
            engine.set_autostart(False)
        self.assertFalse(engine.autostart_enabled())


def response(status, body=None):
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = body
    if not r.ok:
        r.raise_for_status.side_effect = requests.HTTPError(str(status))
    return r


RUN = {'id': 12, 'project': 4, 'status': 'Running', 'role': 'PA', 'site_uid': UID, 'cur_seq': 1,
       'tasks': [{'seq': 1, 'config': {'current_round': 2, 'total_round': 3}}]}


class StatusTest(unittest.TestCase):

    values = {'ROUTER_URL': 'https://router/x', 'SITE_UID': UID, 'ROUTER_TOKEN': 'tok'}

    def fake_get(self, answers):
        def get(url, **kwargs):
            self.assertEqual(kwargs['auth'].token, 'tok')
            for part, answer in answers.items():
                if part in url:
                    if isinstance(answer, Exception):
                        raise answer
                    return answer
            raise AssertionError(url)
        return get

    def test_the_router_says_who_this_site_is_and_what_it_does(self):
        router = S.fetch(self.values, get=self.fake_get({
            'sites/lookup': response(200, {'id': 3, 'name': 'Hospital A'}),
            'projects/lookup': response(200, [{'role': 'participant', 'project': {'id': 4, 'name': 'P'}}]),
            'runs/active': response(200, [RUN, dict(RUN, id=13, site_uid='other')])}))
        self.assertTrue(router.reached)
        self.assertEqual((router.site_name, router.site_id), ('Hospital A', 3))
        self.assertEqual([(p.name, p.role) for p in router.projects], [('P', 'Participant')])
        self.assertEqual(len(router.runs), 1)
        run = router.runs[0]
        self.assertEqual((run.state_text, run.round_text, run.role), ('Training on this computer', '2 of 3',
                                                                      'Participant'))
        self.assertEqual(S.headline(True, None, router),
                         ('Training on this computer, round 2 of 3',
                          'Project P. You can keep using this computer as usual.'))

    def test_failures_become_sentences(self):
        for answer, text in ((response(401), 'refused'), (response(404), 'does not know'),
                             (requests.ConnectionError('x'), 'cannot be reached'),
                             (requests.exceptions.SSLError('x'), 'not trusted')):
            with self.subTest(text=text):
                router = S.fetch(self.values, get=self.fake_get({'sites/lookup': answer}))
                self.assertFalse(router.reached)
                self.assertIn(text, router.error)
        self.assertIn('not set up', S.fetch({}).error)

    def test_headlines(self):
        self.assertEqual(S.headline(False, None, None)[0], 'Starfish is stopped')
        self.assertEqual(S.headline(True, 'step', None)[0], 'Starfish is stopping')
        waiting = S.headline(True, None, S.RouterStatus(reached=True))
        self.assertIn('waits for the admin', waiting[1])
        self.assertIn('tries to reach', S.headline(True, None, S.RouterStatus(reached=False))[1])


@unittest.skipUnless(HAVE_QT, 'PySide6 is not installed')
class WindowTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from starfish_site.app import classic, main as M
        cls.app = QApplication.instance() or QApplication([sys.argv[0]])
        classic.apply(cls.app)
        M.keep_site_online()

    def setUp(self):
        self.homes = []

    def tearDown(self):
        for home in self.homes:
            shutil.rmtree(home, ignore_errors=True)

    def home(self, enrolled=False):
        home = enrolled_home() if enrolled else tempfile.mkdtemp()
        self.homes.append(home)
        return home

    def test_the_look_is_classic(self):
        self.assertEqual(self.app.style().name().lower(), 'windows')
        from PySide6.QtGui import QPalette
        self.assertEqual(self.app.palette().color(QPalette.Window).name(), '#d4d0c8')

    def test_log_lines_in_plain_words(self):
        from starfish_site.app.window import activity_text
        self.assertEqual(activity_text('starfish_site', 'INFO', 'Run 12: Running, round 2 of 3'),
                         'Training on this computer, round 2 of 3.')
        self.assertEqual(activity_text('starfish_site', 'INFO', 'Run 12: Preparing'), 'Getting the cases ready.')
        self.assertIsNone(activity_text('starfish_site', 'INFO',
                                        'Run 12: Pending Aggregating, round 1 of 3, again while it waits'))
        self.assertEqual(activity_text('starfish_site', 'INFO', 'Stopped'), 'Starfish stopped.')
        self.assertIn('cannot be reached', activity_text('starfish_site', 'WARNING',
                                                         'heartbeat failed: ConnectionError: x'))
        self.assertIsNone(activity_text('starfish.controller.tasks.x', 'INFO', 'epoch 1'))
        self.assertTrue(activity_text('starfish.controller.tasks.x', 'ERROR', 'boom').startswith('Error.'))

    def test_check_lines(self):
        from starfish_site.app.widgets import parse_check_line
        self.assertEqual(parse_check_line('Router'), ('section', 'Router'))
        self.assertEqual(parse_check_line('  ok    Reached in 11 ms'), ('ok', 'Reached in 11 ms'))
        self.assertEqual(parse_check_line('  FAIL  No device'), ('fail', 'No device'))
        self.assertEqual(parse_check_line('        Enrol first'), ('hint', 'Enrol first'))
        self.assertEqual(parse_check_line('0 failed, 1 warnings')[0], 'summary')
        self.assertIsNone(parse_check_line('/x/torch/cuda.py:1: UserWarning: y')[0])

    def run_check(self, script):
        import time

        from starfish_site.app import widgets
        panel = widgets.CheckPanel(engine.Site(self.home()))
        with patch('starfish_site.app.engine.site_command', lambda home, *args: [sys.executable, '-c', script]):
            panel.start()
            end = time.monotonic() + 30
            while not panel.ran and time.monotonic() < end:
                self.app.processEvents()
                time.sleep(0.02)
        self.assertTrue(panel.ran)
        return panel

    def test_the_check_lists_its_results(self):
        panel = self.run_check('print("Router"); print("  ok    Reached in 3 ms"); print("  warn  Plain HTTP");'
                               'print("        Use https"); print("\\n0 failed, 1 warnings")')
        self.assertEqual((panel.failed, panel.warned), (0, 1))
        self.assertEqual(panel.tree.topLevelItem(0).text(0), 'Main device')
        self.assertEqual(panel.tree.topLevelItem(0).childCount(), 2)
        self.assertIn('yellow line', panel.summary.text())

    def test_a_check_that_crashes_counts_as_a_problem(self):
        panel = self.run_check('import sys; print("Settings"); print("  ok    fine"); '
                               'print("ModuleNotFoundError: boom"); sys.exit(1)')
        self.assertEqual(panel.failed, 1)
        crash = panel.tree.topLevelItem(1)
        self.assertEqual(crash.text(0), 'The check stopped with an error')
        self.assertEqual(crash.child(0).text(0), 'ModuleNotFoundError: boom')
        self.assertIn('found 1 problem', panel.summary.text())

    def test_the_join_page_needs_a_name_and_an_invitation_or_a_code(self):
        from starfish_site.app.wizard import PAGE_JOIN, SetupWizard, settings_text
        wizard = SetupWizard(engine.Site(self.home()))
        wizard.setStartId(PAGE_JOIN)
        wizard.restart()
        join = wizard.join
        self.assertFalse(join.isComplete())
        join.by_code.setChecked(True)
        join.address.setText('https://10.0.0.5:8443')
        join.code.setText('C0DE')
        self.assertFalse(join.isComplete())
        join.name.setText('Hospital A')
        self.assertTrue(join.isComplete())
        folder = self.home()
        path = I.write(os.path.join(folder, 'inv'), I.Invitation(
            'https://10.0.0.5:8443/starfish/api/v1', 'C0DE', note="the lab's admin",
            expires_at=datetime.datetime(2030, 1, 1, tzinfo=datetime.timezone.utc)))
        self.assertTrue(join.load(str(path)))
        self.assertTrue(join.by_file.isChecked())
        self.assertEqual(join.device.text(), 'https://10.0.0.5:8443/starfish/api/v1')
        self.assertTrue(join.isComplete())
        self.assertEqual(settings_text("the lab's\nadmin"), 'the lab\u2019s admin')
        wizard.close()

    def test_the_main_window_shows_a_stopped_site_and_an_unreachable_router(self):
        from starfish_site.app.window import MainWindow
        window = MainWindow(engine.Site(self.home(enrolled=True)))
        window.show_router(S.RouterStatus(error='The main device cannot be reached (ConnectionError)'))
        self.assertEqual(window.headline.text(), 'Starfish is stopped')
        self.assertTrue(window.start_button.isEnabled())
        self.assertFalse(window.stop_button.isEnabled())
        self.assertIn('cannot be reached', window.connection.text())
        window.quitting = True
        window.close()

    def test_screenshots_of_a_build(self):
        from starfish_site.app import main as M
        folder = self.home()
        with patch('starfish_site.app.widgets.CheckPanel.start', lambda panel: setattr(panel, 'ran', True)):
            code = M.save_screenshots(folder, home=self.home(), check_seconds=1)
        self.assertEqual(code, 0)
        names = sorted(os.listdir(folder))
        self.assertIn('main-window.png', names)
        self.assertIn('wizard-1.png', names)
        self.assertIn('summary.txt', names)


if __name__ == '__main__':
    unittest.main()
