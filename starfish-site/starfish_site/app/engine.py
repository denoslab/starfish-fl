"""
The site's own process, which Starfish Site starts, stops and watches.

The app runs the site as ``starfish-site start`` in a process of its own. The
site keeps training when the window closes, and a crash in training leaves
the window alone. The app stops the site through the request file of
``starfish-site stop`` and learns whether it runs from the lock on the site
folder, the same way the command line does.

Starting at log on goes through the system's own list of programs to start,
the ``Run`` key of the registry on Windows, a launch agent on macOS and an
autostart entry on Linux. None of them needs admin rights.
"""

import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

from starfish_site import cli
from starfish_site import config as C

APP_NAME = 'Starfish Site'
BUNDLE_ID = 'com.denoslab.starfish-site'
LOG_NAME = 'starfish-site.log'
# What the site's process writes outside its log, such as a crash before the log opens
STDERR_NAME = 'starfish-site-stderr.log'
RUN_KEY = r'Software\Microsoft\Windows\CurrentVersion\Run'
# The log lines that tell a user what happened, from starfish_site.cli and starfish_site.runner
LOG_LINE = re.compile(r'^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d),\d+ (\w+)\s+([\w.]+): (.*)$')


def frozen():
    """True inside the installed app, which PyInstaller built."""
    return bool(getattr(sys, 'frozen', False))


def site_command(home, *args):
    """The command line of ``starfish-site`` for the site in ``home``."""
    if frozen():
        name = 'starfish-site.exe' if os.name == 'nt' else 'starfish-site'
        return [str(Path(sys.executable).with_name(name)), '--home', str(home), *args]
    return [sys.executable, '-m', 'starfish_site', '--home', str(home), *args]


def app_command(*args):
    """The command line that opens Starfish Site itself."""
    if frozen():
        return [sys.executable, *args]
    return [sys.executable, '-m', 'starfish_site.app', *args]


def detached():
    """Arguments for Popen that let a process outlive the app and show no window of its own."""
    if os.name == 'nt':
        return {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'start_new_session': True}


def child_environment():
    env = dict(os.environ)
    # A program frozen by PyInstaller that starts another needs a fresh start for the child
    env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
    return env


class Site:
    """The site in one folder, ``~/.starfish-site`` unless another is named."""

    def __init__(self, home=None):
        # Absolute, since the site's process starts in this folder and gets it as --home
        self.home = C.site_home(home).absolute()

    def settings(self):
        return C.read_config(self.home)

    def enrolled(self, values=None):
        values = self.settings() if values is None else values
        return bool(values.get('SITE_UID') and values.get('ROUTER_URL') and values.get('ROUTER_TOKEN'))

    def update(self, **changes):
        """Change settings in site.env, dropping those set to None or an empty string."""
        values = self.settings()
        for key, value in changes.items():
            if value in (None, ''):
                values.pop(key, None)
            else:
                values[key] = str(value)
        C.write_config(self.home, values)
        return values

    def pid(self):
        """The process that runs this site, or None."""
        return cli.running_pid(self.home)

    def stopping(self):
        return cli.stop_request(self.home)

    def log_path(self):
        return C.log_folder(self.home) / LOG_NAME

    def stderr_path(self):
        return C.log_folder(self.home) / STDERR_NAME

    def start(self, popen=subprocess.Popen):
        """Start the site in a process of its own and return it."""
        folder = C.log_folder(self.home)
        folder.mkdir(parents=True, exist_ok=True)
        with open(self.stderr_path(), 'wb') as stderr:
            return popen(site_command(self.home, 'start'), stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=stderr, cwd=str(self.home),
                         env=child_environment(), close_fds=True, **detached())

    def stop(self, now=False):
        """Ask the running site to stop, after the step in progress or at once."""
        cli.request_stop(self.home, now=now)

    def last_error(self, limit=2000):
        """The end of what the site's process wrote outside its log, or an empty string."""
        try:
            text = self.stderr_path().read_bytes()[-limit:].decode('utf-8', 'replace')
        except OSError:
            return ''
        return text.strip()


def read_log(path, start=0, limit=2 ** 20):
    """
    Lines of the site's log from byte ``start`` on, and the offset to read from next.

    A log that shrank was rotated, and reading starts again from its top.
    Only whole lines are returned, and a line still being written waits for
    the next read.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], 0
    if size < start:
        start = 0
    with open(path, 'rb') as f:
        f.seek(max(start, size - limit))
        data = f.read()
    end = data.rfind(b'\n') + 1
    lines = data[:end].decode('utf-8', 'replace').splitlines()
    return lines, max(start, size - limit) + end


def parse_log_line(line):
    """(date, time, level, logger, message) of a log line, or None for a line of a traceback."""
    match = LOG_LINE.match(line)
    return match.groups() if match else None


# Starting at log on

def autostart_command():
    return app_command('--background')


def _desktop_quote(arg):
    """One argument of an Exec line, quoted as the desktop entry specification wants it."""
    if arg and not re.search(r'[\s"\'\\`$<>~|&;*?#()]', arg):
        return arg
    return '"' + re.sub(r'(["`$\\])', r'\\\1', arg) + '"'


def _linux_autostart_path(home):
    return Path(home) / '.config' / 'autostart' / 'starfish-site.desktop'


def _mac_agent_path(home):
    return Path(home) / 'Library' / 'LaunchAgents' / (BUNDLE_ID + '.plist')


def autostart_enabled(platform=sys.platform, home=None):
    home = home or Path.home()
    if platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
                winreg.QueryValueEx(key, APP_NAME)
            return True
        except OSError:
            return False
    if platform == 'darwin':
        return _mac_agent_path(home).is_file()
    return _linux_autostart_path(home).is_file()


def set_autostart(enabled, command=None, platform=sys.platform, home=None):
    """Start Starfish Site at log on, or no longer, for this user only."""
    command = command or autostart_command()
    home = home or Path.home()
    if platform == 'win32':
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, subprocess.list2cmdline(command))
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
        return
    path = _mac_agent_path(home) if platform == 'darwin' else _linux_autostart_path(home)
    if not enabled:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if platform == 'darwin':
        agent = {'Label': BUNDLE_ID, 'ProgramArguments': list(command), 'RunAtLoad': True,
                 'ProcessType': 'Interactive'}
        with open(path, 'wb') as f:
            plistlib.dump(agent, f)
    else:
        entry = ['[Desktop Entry]', 'Type=Application', 'Name=' + APP_NAME,
                 'Comment=Starts Starfish Site in the background at log on',
                 'Exec=' + ' '.join(_desktop_quote(part) for part in command),
                 'X-GNOME-Autostart-enabled=true', '']
        path.write_text('\n'.join(entry), encoding='utf-8')
