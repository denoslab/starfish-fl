"""
Starfish Site, the desktop app of a site, and how it starts.

    Starfish Site [--background] [--home DIR] [INVITATION]

Without settings the app opens its setup wizard, and with them its main
window. ``--background`` starts it at log on with only the starfish icon in
view, and starts the site's process unless it runs already. An invitation
file given on the command line, as a double-click on one does, opens in the
wizard. One copy of the app runs per user and site folder, and a second start
hands its request to the first and ends.
"""

import argparse
import atexit
import hashlib
import os
import sys

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox

from starfish_site import __version__
from starfish_site.app import classic, engine

APP_USER_MODEL_ID = 'Denoslab.StarfishSite'
# Lets the Windows installer see a running copy and ask to close it
WINDOWS_MUTEX = 'StarfishSiteApp'
_handles = []


def keep_site_online():
    """
    Leave the router's view of this site to the site's own process.

    Importing the controller's code registers a report at exit that the site
    went offline, meant for the process that runs the site. The app imports
    that code too, and its exit must not take the site offline.
    """
    try:
        from starfish.controller import site_status_task
    except Exception:
        return
    atexit.unregister(site_status_task.report_exit)


def windows_setup():
    if sys.platform != 'win32':
        return
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        _handles.append(ctypes.windll.kernel32.CreateMutexW(None, False, WINDOWS_MUTEX))
    except (AttributeError, OSError):
        pass


def server_name(home):
    who = os.environ.get('USERNAME') or os.environ.get('USER') or ''
    digest = hashlib.sha1('{}|{}'.format(who, os.path.abspath(str(home))).encode('utf-8')).hexdigest()
    return 'starfish-site-' + digest[:16]


def hand_over(name, message):
    """Give ``message`` to a copy that runs already. True when one took it."""
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(500):
        return False
    socket.write(message.encode('utf-8'))
    socket.flush()
    socket.waitForBytesWritten(1000)
    socket.disconnectFromServer()
    return True


class FileOpener(QObject):
    """Catches the invitation files that macOS hands over when one is double-clicked."""

    def __init__(self, on_file):
        super().__init__()
        self.on_file = on_file

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FileOpen:
            self.on_file(event.file())
            return True
        return False


class Controller:
    """Shows the setup wizard or the main window, and answers other copies of the app."""

    def __init__(self, app, site, background=False):
        self.app = app
        self.site = site
        self.background = background
        self.wizard = None
        self.window = None

    def set_up(self):
        return self.site.enrolled() and bool(self.site.settings().get('BABELBRAIN_FL_STORE'))

    def begin(self, invitation=None):
        if invitation or not self.set_up():
            if self.background and not invitation:
                return False
            self.open_wizard(invitation)
            return True
        self.open_window(show=not self.background)
        if self.background and self.site.pid() is None:
            self.window.start_site()
        return True

    def open_wizard(self, invitation=None):
        from starfish_site.app.wizard import SetupWizard

        if self.wizard is not None:
            if invitation:
                self.wizard.open_invitation(invitation)
            self.wizard.show()
            self.wizard.raise_()
            self.wizard.activateWindow()
            return
        self.wizard = SetupWizard(self.site, invitation)
        self.wizard.finished.connect(self.wizard_done)
        self.wizard.show()
        self.wizard.raise_()
        self.wizard.activateWindow()

    def wizard_done(self, result):
        wizard, self.wizard = self.wizard, None
        wizard.deleteLater()
        if result != wizard.Accepted:
            if self.window is None:
                self.app.quit()
            return
        try:
            engine.set_autostart(wizard.at_logon)
        except OSError as e:
            QMessageBox.warning(None, 'Starfish Site', 'Starting at log on could not be set up.\n\n{}'.format(e))
        if self.window is None:
            self.open_window(show=True)
        else:
            self.window.settings_changed()
            self.window.show_window()
        if wizard.start_now and self.site.pid() is None:
            self.window.start_site()

    def open_window(self, show=True):
        from starfish_site.app.window import MainWindow

        if self.window is None:
            self.window = MainWindow(self.site, on_setup_again=self.open_wizard)
        if show or self.window.tray is None:
            self.window.show_window()

    def message(self, text):
        """A request from another copy of the app, ``show`` or ``open <invitation path>``."""
        if text.startswith('open '):
            path = text[5:]
            if self.window is not None and self.set_up():
                self.window.show_window()
                answer = QMessageBox.question(
                    self.window, 'Starfish Site', 'This computer is set up already. Do you want to set it up '
                    'again with the invitation {}?'.format(os.path.basename(path)),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:
                    return
            self.open_wizard(path)
            return
        if self.wizard is not None:
            self.open_wizard()
        elif self.window is not None:
            self.window.show_window()
        else:
            self.begin()


def save_screenshots(folder, home=None, check_seconds=600):
    """
    Save pictures of the wizard's pages and the main window to ``folder`` and quit, for tests of a build.

    The site folder is a new empty one unless ``home`` names another. The
    check page runs the real check through ``starfish-site``, which tests the
    command that comes with the app as well. Writes ``summary.txt`` and
    returns 0 when the check found no problem.
    """
    import tempfile
    import time

    from starfish_site.app.window import MainWindow
    from starfish_site.app.wizard import PAGE_CHECK, PAGE_FINISH, SetupWizard

    os.makedirs(folder, exist_ok=True)
    site = engine.Site(home or tempfile.mkdtemp(prefix='starfish-site-screens-'))
    app = QApplication.instance()
    lines = ['Starfish Site {}, style {}, font {}'.format(
        __version__, app.style().name(), app.font().family())]

    def settle(seconds=0.3):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.02)

    wizard = SetupWizard(site)
    wizard.show()
    for page in wizard.pageIds():
        wizard.setStartId(page)
        wizard.restart()
        settle()
        if page == PAGE_CHECK:
            end = time.monotonic() + check_seconds
            while not wizard.check.panel.ran and time.monotonic() < end:
                settle(0.2)
            panel = wizard.check.panel
            lines.append('check ran {}, failed {}, warned {}'.format(panel.ran, panel.failed, panel.warned))
            lines.append(panel.summary.text())
        wizard.grab().save(os.path.join(folder, 'wizard-{}.png'.format(page + 1)))
    wizard.hide()
    window = MainWindow(site)
    window.show()
    settle(1.0)
    window.grab().save(os.path.join(folder, 'main-window.png'))
    window.quitting = True
    window.close()
    with open(os.path.join(folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    panel = wizard.check.panel
    return 0 if panel.ran and not panel.failed and PAGE_FINISH in wizard.pageIds() else 1


def parser():
    p = argparse.ArgumentParser(prog='Starfish Site', description='The desktop app of a Starfish-FL site.')
    p.add_argument('--version', action='version', version='Starfish Site ' + __version__)
    p.add_argument('--background', action='store_true', help='start with only the starfish icon in view')
    p.add_argument('--home', help='site folder, default ~/.starfish-site or $STARFISH_SITE_HOME')
    p.add_argument('--screenshots', metavar='FOLDER',
                   help='save pictures of the windows to FOLDER and quit, for tests of a build')
    p.add_argument('invitation', nargs='?', help='an invitation file to open in the setup wizard')
    return p


def main(argv=None):
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    keep_site_online()
    windows_setup()
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName('Starfish Site')
    app.setApplicationDisplayName('Starfish Site')
    app.setApplicationVersion(__version__)
    app.setOrganizationName('Starfish')
    app.setQuitOnLastWindowClosed(False)
    classic.apply(app)
    app.setWindowIcon(classic.app_icon())
    if args.screenshots:
        return save_screenshots(args.screenshots, args.home)

    site = engine.Site(args.home)
    invitation = os.path.abspath(args.invitation) if args.invitation else None
    name = server_name(site.home)
    if hand_over(name, 'open ' + invitation if invitation else 'show'):
        return 0
    server = QLocalServer()
    QLocalServer.removeServer(name)
    controller = Controller(app, site, background=args.background)

    def incoming():
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            socket.waitForReadyRead(1000)
            text = bytes(socket.readAll()).decode('utf-8', 'replace')
            socket.disconnectFromServer()
            QTimer.singleShot(0, lambda text=text: controller.message(text))

    server.newConnection.connect(incoming)
    server.listen(name)
    opener = FileOpener(lambda path: controller.message('open ' + path))
    app.installEventFilter(opener)
    if not controller.begin(invitation):
        return 0
    return app.exec()
