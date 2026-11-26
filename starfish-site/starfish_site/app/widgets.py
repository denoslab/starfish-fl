"""
Pieces that the setup wizard and the main window share.

``CheckPanel`` runs ``starfish-site check`` in a process of its own and lists
what it finds as it goes. ``CasesBox`` counts the cases in a folder in the
background. ``in_background`` runs a slow function off the window's thread
and hands its result back on it.
"""

import logging
import os
import subprocess
import sys
import threading

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QStyle, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from starfish_site.app import engine

# The check's sections, as the check prints them, in words for the people at the site
SECTION_TEXT = {
    'Settings': 'Settings',
    'Router': 'Main device',
    'Sample stores': 'Cases',
    'Torch and device': 'Processor and graphics card',
    'Training on this device': 'Practice training',
}
_keep = []


class _Bridge(QObject):
    done = Signal(object, object)


class _Job(QRunnable):

    def __init__(self, fn, bridge):
        super().__init__()
        self.fn, self.bridge = fn, bridge

    def run(self):
        try:
            result, error = self.fn(), None
        except Exception as e:
            result, error = None, e
        self.bridge.done.emit(result, error)


def in_background(fn, on_done):
    """Run ``fn()`` on a worker thread and call ``on_done(result, error)`` on the window's thread."""
    bridge = _Bridge()
    _keep.append(bridge)

    def finish(result, error):
        _keep.remove(bridge)
        on_done(result, error)

    bridge.done.connect(finish, Qt.QueuedConnection)
    QThreadPool.globalInstance().start(_Job(fn, bridge))


def standard_icon(name):
    return QApplication.style().standardIcon(getattr(QStyle, name))


def level_icon(level):
    return standard_icon({'ok': 'SP_DialogApplyButton', 'warn': 'SP_MessageBoxWarning',
                          'fail': 'SP_MessageBoxCritical'}.get(level, 'SP_MessageBoxInformation'))


def parse_check_line(line):
    """
    One printed line of ``starfish-site check`` as (kind, text).

    ``kind`` is ``section``, ``ok``, ``warn``, ``fail``, ``hint``, ``summary``
    or None for a line to skip.
    """
    if not line.strip():
        return None, ''
    for prefix, kind in (('  ok    ', 'ok'), ('  warn  ', 'warn'), ('  FAIL  ', 'fail'), ('        ', 'hint')):
        if line.startswith(prefix):
            return kind, line[len(prefix):].strip()
    text = line.strip()
    if line.startswith(' '):
        return None, text
    if text.endswith('warnings') and ' failed, ' in text:
        return 'summary', text
    if text in SECTION_TEXT:
        return 'section', text
    # The first line, a note before enrolment, and anything a library prints
    return None, text


class _Lines(QObject):
    """Hands each line a process prints, and its exit code, to the window's thread."""

    line = Signal(str)
    done = Signal(int)

    def follow(self, process):
        for raw in iter(process.stdout.readline, b''):
            self.line.emit(raw.decode('utf-8', 'replace').rstrip('\r\n'))
        process.stdout.close()
        self.done.emit(process.wait())


class CheckPanel(QWidget):
    """Runs the check of this computer and lists each result under its section."""

    finished = Signal(int, int)

    def __init__(self, site, parent=None):
        super().__init__(parent)
        self.site = site
        self.process = None
        self.failed = self.warned = 0
        self.ran = False
        self._reader = None
        self._section = None
        self._last = None
        self._other = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.button = QPushButton('&Start the Check')
        self.button.clicked.connect(self.start)
        top.addWidget(self.button)
        self.busy = QProgressBar()
        self.busy.setRange(0, 1)
        self.busy.setTextVisible(False)
        top.addWidget(self.busy, 1)
        layout.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['What was checked'])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setWordWrap(True)
        layout.addWidget(self.tree, 1)
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def start(self):
        if self.process is not None:
            return
        self.tree.clear()
        self.failed = self.warned = 0
        self._section = self._last = None
        self._other = []
        self.summary.setText('Checking. The practice training can take a minute.')
        self.button.setEnabled(False)
        self.busy.setRange(0, 0)
        env = engine.child_environment()
        env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        # No console window of its own on Windows
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        try:
            self.process = subprocess.Popen(engine.site_command(self.site.home, 'check'),
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, env=env, **flags)
        except OSError as e:
            self.process = None
            self._other = [str(e)]
            self._done(-1, started=False)
            return
        reader = _Lines()
        reader.line.connect(self.add_line, Qt.QueuedConnection)
        reader.done.connect(self._done, Qt.QueuedConnection)
        self._reader = reader
        threading.Thread(target=reader.follow, args=(self.process,), daemon=True).start()

    def add_line(self, line):
        kind, text = parse_check_line(line)
        if kind is None and text:
            # Kept for a check that ends in an error before it reports one
            self._other.append(text)
            del self._other[:-12]
        if kind == 'section':
            self._section = QTreeWidgetItem(self.tree, [SECTION_TEXT.get(text, text)])
            font = self._section.font(0)
            font.setBold(True)
            self._section.setFont(0, font)
            self._section.setExpanded(True)
        elif kind in ('ok', 'warn', 'fail'):
            if self._section is None:
                self._section = QTreeWidgetItem(self.tree, ['Check'])
                self._section.setExpanded(True)
            self._last = QTreeWidgetItem(self._section, [text])
            self._last.setIcon(0, level_icon(kind))
            self._last.setToolTip(0, text)
            self.failed += kind == 'fail'
            self.warned += kind == 'warn'
            # A section shows the worst of its results
            rank = {'ok': 0, 'warn': 1, 'fail': 2}
            worst = self._section.data(0, Qt.UserRole) or 'ok'
            if rank[kind] >= rank[worst]:
                worst = kind
            self._section.setData(0, Qt.UserRole, worst)
            self._section.setIcon(0, level_icon(worst))
            self.tree.scrollToItem(self._last)
        elif kind == 'hint' and self._last is not None:
            hint = QTreeWidgetItem(self._last, [text])
            hint.setToolTip(0, text)
            self._last.setExpanded(True)

    def _done(self, code, started=True):
        if code != 0 and not self.failed:
            # The check stopped before it could report what went wrong
            self.failed = 1
            item = QTreeWidgetItem(self.tree, ['The check stopped with an error' if started else
                                               'The check could not start'])
            item.setIcon(0, level_icon('fail'))
            for text in self._other or ['No details']:
                QTreeWidgetItem(item, [text]).setToolTip(0, text)
            item.setExpanded(True)
        self._finish(code)

    def _finish(self, code):
        self.process = None
        self._reader = None
        self.ran = True
        self.busy.setRange(0, 1)
        self.busy.setValue(1)
        self.button.setText('&Check Again')
        self.button.setEnabled(True)
        if self.failed:
            self.summary.setText('The check found {} problem{}. Each red line says what is wrong and what '
                                 'to do about it.'.format(self.failed, '' if self.failed == 1 else 's'))
        elif self.warned:
            self.summary.setText('This computer can take part. Look at the {} yellow line{} for things '
                                 'to know.'.format(self.warned, '' if self.warned == 1 else 's'))
        else:
            self.summary.setText('All set. This computer can take part.')
        self.finished.emit(self.failed, self.warned)


def count_cases(folder):
    """The usable cases of a folder per frequency and split, and the reasons any were left out."""
    from starfish_site.checks import describe_store

    quiet = logging.getLogger('starfish_site.app.cases')
    quiet.propagate = False
    quiet.addHandler(logging.NullHandler())
    return describe_store(folder, logger=quiet)


def describe_cases(counts, rejected):
    """Lines that say how many cases are ready, per frequency, and how many were left out."""
    total = sum(sum(splits.values()) for splits in counts.values())
    lines = ['{} case{} ready for training.'.format(total, ' is' if total == 1 else 's are')]
    for bucket, splits in sorted(counts.items()):
        lines.append('{} kHz, {}.'.format(bucket // 1000, ', '.join(
            '{} {}'.format(n, {'train': 'for training', 'val': 'for checking',
                               'test': 'for the final test'}.get(split, split))
            for split, n in sorted(splits.items()))))
    left = sum(rejected.values())
    if left:
        lines.append('{} case{} left out, {}.'.format(left, ' is' if left == 1 else 's are', ', '.join(
            '{} {}'.format(n, reason.replace('_', ' ')) for reason, n in sorted(rejected.items()))))
    return total, lines


class CasesBox(QGroupBox):
    """Says how many cases a folder holds, counted in the background."""

    counted = Signal(int)

    def __init__(self, title='What Starfish found', parent=None):
        super().__init__(title, parent)
        layout = QHBoxLayout(self)
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignTop)
        layout.addWidget(self.icon)
        self.text = QLabel('No folder is chosen.')
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.text, 1)
        self.total = None
        self._folder = None

    def count(self, folder):
        self._folder = folder
        self.total = None
        if not folder or not os.path.isdir(folder):
            self.icon.setPixmap(standard_icon('SP_MessageBoxWarning').pixmap(32, 32))
            self.text.setText('The folder {} does not exist.'.format(folder) if folder else 'No folder is chosen.')
            self.counted.emit(-1)
            return
        self.icon.setPixmap(standard_icon('SP_FileDialogContentsView').pixmap(32, 32))
        self.text.setText('Looking at the cases in this folder...')

        def done(result, error, folder=folder):
            if folder != self._folder:
                return
            if error is not None:
                self.icon.setPixmap(standard_icon('SP_MessageBoxCritical').pixmap(32, 32))
                self.text.setText('The folder cannot be read. {}'.format(error))
                self.total = 0
                self.counted.emit(0)
                return
            total, lines = describe_cases(*result)
            self.total = total
            self.icon.setPixmap(standard_icon('SP_DialogApplyButton' if total else
                                              'SP_MessageBoxWarning').pixmap(32, 32))
            if not total:
                lines.append('Starfish trains only on cases saved by BabelBrain 0.8.8 in the form '
                             'Starfish reads. Ask the admin which folder to choose.')
            self.text.setText('\n'.join(lines))
            self.counted.emit(total)

        in_background(lambda: count_cases(folder), done)


def is_mac():
    return sys.platform == 'darwin'


def tray_place():
    """Where the starfish icon sits on this system, for the texts that point to it."""
    if is_mac():
        return 'in the menu bar at the top of the screen'
    if sys.platform == 'win32':
        return 'in the notification area of the taskbar'
    return 'in the panel of your desktop'
