"""
The main window of Starfish Site and its icon in the notification area.

The window has a menu bar, a toolbar, five tabs and a status bar, like the
usual Windows software of 2000 to 2010. Status shows what Starfish does on
this computer, with Start and Stop. Projects lists the projects this computer
takes part in, Cases the folder of cases, Activity what happened, and
Settings the choices a person can make. Closing the window leaves Starfish
running, and the starfish icon in the notification area opens the window again.
"""

import os
import sys
import time

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QSystemTrayIcon, QTabWidget, QToolBar, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from starfish_site import __version__
from starfish_site import config as C
from starfish_site.app import classic, engine, status as S, widgets

HELP_URL = 'https://denoslab.com/babelbrain-fl-docs/'
LOCAL_SECONDS = 1
ROUTER_SECONDS = 10
ACTIVITY_LIMIT = 500
DEVICES = (('', 'Automatic choice'), ('cpu', 'The processor (CPU)'), ('cuda', 'The NVIDIA graphics card (CUDA)'),
           ('mps', 'The Apple graphics chip (MPS)'))


def activity_text(logger_name, level, message):
    """A log line in words for the people at the site, or None to leave it out of the short list."""
    if logger_name == 'starfish_site':
        if message.startswith('starfish-site '):
            return 'Starfish started.'
        if message.startswith('Polling the router'):
            return 'Starfish waits for work from the main device.'
        if message.startswith('Run ') and ': ' in message:
            if message.endswith('again while it waits'):
                # A coordinator's repeat every few seconds while it waits
                return None
            state, _, rest = message.partition(': ')[2].partition(',')
            if state not in S.STATE_TEXT:
                return None
            rest = rest.strip()
            return '{}{}.'.format(S.STATE_TEXT[state], ', ' + rest if rest.startswith('round ') else '')
        if message.startswith('Stopping after'):
            return 'Starfish stops after the step in progress.'
        if message.startswith('Stopping now'):
            return 'Starfish stops at once.'
        if message == 'Stopped':
            return 'Starfish stopped.'
        if message.startswith('heartbeat failed') or message.startswith('fetch_run failed'):
            return 'The main device cannot be reached. Starfish keeps trying.'
        if message.endswith('works again'):
            return 'The main device can be reached again.'
        if level in ('ERROR', 'CRITICAL'):
            return 'Error. {}'.format(message)
        return None
    if level in ('ERROR', 'CRITICAL'):
        return 'Error. {}'.format(message)
    return None


class AboutDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('About Starfish Site')
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(classic.starfish_pixmap(64))
        icon.setAlignment(Qt.AlignTop)
        top.addWidget(icon)
        text = QLabel('<b style="font-size:14px">Starfish Site</b><br>Version {}<br><br>'
                      'Starfish Site lets this computer take part in federated training of a shared '
                      'BabelBrain model. It is part of Starfish-FL.<br><br>'
                      'Licensed under the Apache License 2.0.<br>'
                      '<a href="{}">{}</a>'.format(__version__, HELP_URL, HELP_URL))
        text.setWordWrap(True)
        text.setOpenExternalLinks(True)
        top.addWidget(text, 1)
        layout.addLayout(top)
        line = QLabel()
        line.setFrameStyle(QLabel.HLine | QLabel.Sunken)
        layout.addWidget(line)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.setFixedWidth(380)


class CheckDialog(QDialog):

    def __init__(self, site, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Check This Computer')
        self.resize(520, 420)
        layout = QVBoxLayout(self)
        intro = QLabel('Starfish asks the main device one test question and trains a small model on '
                       'made-up data. Your cases never leave this computer.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.panel = widgets.CheckPanel(site)
        layout.addWidget(self.panel, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.panel.ran and self.panel.process is None:
            QTimer.singleShot(0, self.panel.start)


def group(title, layout):
    box = QGroupBox(title)
    box.setLayout(layout)
    return box


def note(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet('color: #404040')
    return label


class MainWindow(QMainWindow):

    def __init__(self, site, on_setup_again=None):
        super().__init__()
        self.site = site
        self.values = site.settings()
        self.on_setup_again = on_setup_again
        self.router = None
        self.running = False
        self.started_at = None
        self.child = None
        self.log_offset = 0
        self.log_lines = []
        self.quitting = False
        self.told_about_tray = False
        self.setWindowTitle('Starfish Site')
        self.setWindowIcon(classic.app_icon())
        self.resize(640, 500)
        self._actions()
        self._menus()
        self._toolbar()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._status_tab(), '&Status')
        self.tabs.addTab(self._projects_tab(), '&Projects')
        self.tabs.addTab(self._cases_tab(), '&Cases')
        self.tabs.addTab(self._activity_tab(), 'Ac&tivity')
        self.tabs.addTab(self._settings_tab(), 'Se&ttings')
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self.tabs)
        self.setCentralWidget(holder)
        self._status_bar()
        self.tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray()
        self.local_timer = QTimer(self)
        self.local_timer.timeout.connect(self.refresh_local)
        self.local_timer.start(LOCAL_SECONDS * 1000)
        self.router_timer = QTimer(self)
        self.router_timer.timeout.connect(self.refresh_router)
        self.router_timer.start(ROUTER_SECONDS * 1000)
        self._asking = False
        self.refresh_local()
        self.refresh_router()
        self.cases.count(self.values.get('BABELBRAIN_FL_STORE'))

    # Building the window

    def _actions(self):
        self.act_start = QAction(classic.action_icon('start'), '&Start', self)
        self.act_start.setStatusTip('Start Starfish on this computer')
        self.act_start.triggered.connect(self.start_site)
        self.act_stop = QAction(classic.action_icon('stop'), 'S&top', self)
        self.act_stop.setStatusTip('Stop Starfish after the step in progress')
        self.act_stop.triggered.connect(lambda: self.stop_site(False))
        self.act_stop_now = QAction('Stop &Now', self)
        self.act_stop_now.setStatusTip('Stop Starfish at once, cutting the step in progress short')
        self.act_stop_now.triggered.connect(lambda: self.stop_site(True))
        self.act_check = QAction(widgets.standard_icon('SP_ComputerIcon'), '&Check This Computer...', self)
        self.act_check.setStatusTip('Run the same check as during setup')
        self.act_check.triggered.connect(self.check_computer)
        self.act_folder = QAction(widgets.standard_icon('SP_DirOpenIcon'), 'Change Cases &Folder...', self)
        self.act_folder.setStatusTip('Choose another folder with this computer\'s cases')
        self.act_folder.triggered.connect(self.change_folder)
        self.act_open_log = QAction('Open &Log Folder', self)
        self.act_open_log.triggered.connect(lambda: self.open_path(C.log_folder(self.site.home)))
        self.act_refresh = QAction(widgets.standard_icon('SP_BrowserReload'), '&Refresh', self)
        self.act_refresh.setShortcut(QKeySequence.Refresh)
        self.act_refresh.triggered.connect(self.refresh_all)
        self.act_close = QAction('&Close Window', self)
        self.act_close.setShortcut(QKeySequence.Close)
        self.act_close.triggered.connect(self.close)
        self.act_exit = QAction('E&xit', self)
        self.act_exit.triggered.connect(self.exit_app)
        self.act_help = QAction(widgets.standard_icon('SP_DialogHelpButton'), 'Starfish Site &Help', self)
        self.act_help.setShortcut(QKeySequence.HelpContents)
        self.act_help.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(HELP_URL)))
        self.act_about = QAction('&About Starfish Site', self)
        self.act_about.triggered.connect(lambda: AboutDialog(self).exec())

    def _menus(self):
        bar = self.menuBar()
        file_menu = bar.addMenu('&File')
        file_menu.addAction(self.act_open_log)
        file_menu.addSeparator()
        file_menu.addAction(self.act_close)
        file_menu.addAction(self.act_exit)
        site_menu = bar.addMenu('S&ite')
        for action in (self.act_start, self.act_stop, self.act_stop_now):
            site_menu.addAction(action)
        site_menu.addSeparator()
        site_menu.addAction(self.act_check)
        site_menu.addAction(self.act_folder)
        view_menu = bar.addMenu('&View')
        view_menu.addAction(self.act_refresh)
        help_menu = bar.addMenu('&Help')
        help_menu.addAction(self.act_help)
        help_menu.addSeparator()
        help_menu.addAction(self.act_about)

    def _toolbar(self):
        bar = QToolBar('Standard Buttons')
        bar.setMovable(False)
        bar.setIconSize(QSize(16, 16))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.addAction(self.act_start)
        bar.addAction(self.act_stop)
        bar.addSeparator()
        bar.addAction(self.act_check)
        bar.addAction(self.act_folder)
        bar.addSeparator()
        bar.addAction(self.act_refresh)
        bar.addAction(self.act_help)
        self.addToolBar(bar)

    def _status_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.state_icon = QLabel()
        self.state_icon.setAlignment(Qt.AlignTop)
        row.addWidget(self.state_icon)
        texts = QVBoxLayout()
        self.headline = QLabel()
        self.headline.setFont(classic.font(14, bold=True))
        self.headline.setWordWrap(True)
        texts.addWidget(self.headline)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        texts.addWidget(self.detail)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        texts.addWidget(self.progress)
        texts.addStretch(1)
        row.addLayout(texts, 1)
        buttons = QVBoxLayout()
        self.start_button = QPushButton(classic.action_icon('start'), ' Start')
        self.start_button.clicked.connect(self.start_site)
        self.stop_button = QPushButton(classic.action_icon('stop'), ' Stop')
        self.stop_button.clicked.connect(lambda: self.stop_site(False))
        for button in (self.start_button, self.stop_button):
            button.setMinimumWidth(96)
            buttons.addWidget(button)
        buttons.addStretch(1)
        row.addLayout(buttons)
        layout.addWidget(group('Starfish on this computer', row))

        form = QFormLayout()
        self.address = QLabel()
        self.address.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.connection = QLabel()
        self.connection.setWordWrap(True)
        self.known_as = QLabel()
        form.addRow('Address:', self.address)
        form.addRow('Connection:', self.connection)
        form.addRow('This computer\'s name:', self.known_as)
        layout.addWidget(group('Main device', form))

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        rows = (('Check This Computer...', self.check_computer,
                 'Runs the same check as during setup and says what to do about any problem.'),
                ('Show Activity', lambda: self.tabs.setCurrentIndex(3),
                 'Shows what Starfish did on this computer, with the time of each step.'),
                ('Contact the Admin...', self.contact_admin,
                 'Shows the note from the admin of the main device, such as how to reach them.'))
        for index, (text, slot, help_text) in enumerate(rows):
            button = QPushButton(text)
            button.setMinimumWidth(150)
            button.clicked.connect(slot)
            grid.addWidget(button, index, 0)
            grid.addWidget(note(help_text), index, 1)
        layout.addWidget(group('If something seems wrong', grid))
        layout.addStretch(1)
        return page

    def _projects_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(note('The projects this computer takes part in. The admin of the main device starts '
                              'and stops them, and nothing here needs your action.'))
        self.projects = QTreeWidget()
        self.projects.setRootIsDecorated(False)
        self.projects.setHeaderLabels(['Project', 'Role', 'Round', 'State'])
        self.projects.setColumnWidth(0, 220)
        self.projects.setColumnWidth(1, 90)
        self.projects.setColumnWidth(2, 60)
        layout.addWidget(self.projects, 1)
        return page

    def _cases_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel('Folder:'))
        self.folder = QLabel()
        self.folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.folder.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.folder.setStyleSheet('background: white; padding: 2px')
        row.addWidget(self.folder, 1)
        change = QPushButton('C&hange Folder...')
        change.clicked.connect(self.change_folder)
        row.addWidget(change)
        layout.addLayout(row)
        self.cases = widgets.CasesBox()
        layout.addWidget(self.cases)
        buttons = QHBoxLayout()
        again = QPushButton('&Look Again')
        again.clicked.connect(lambda: self.cases.count(self.values.get('BABELBRAIN_FL_STORE')))
        show = QPushButton('&Open Folder')
        show.clicked.connect(lambda: self.open_path(self.values.get('BABELBRAIN_FL_STORE')))
        buttons.addWidget(again)
        buttons.addWidget(show)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addWidget(note('Starfish only reads the cases in this folder. They never leave this computer, '
                              'and only what the model learns from them goes to the main device.'))
        layout.addStretch(1)
        return page

    def _activity_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.activity = QTreeWidget()
        self.activity.setRootIsDecorated(False)
        self.activity.setHeaderLabels(['Time', 'What happened'])
        self.activity.setColumnWidth(0, 120)
        self.activity.setWordWrap(True)
        layout.addWidget(self.activity, 1)
        row = QHBoxLayout()
        self.details = QCheckBox('Show all &details')
        self.details.toggled.connect(self.rebuild_activity)
        row.addWidget(self.details)
        row.addStretch(1)
        open_log = QPushButton('Open &Log File')
        open_log.clicked.connect(lambda: self.open_path(self.site.log_path()))
        row.addWidget(open_log)
        layout.addLayout(row)
        return page

    def _settings_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        start_box = QVBoxLayout()
        self.autostart = QCheckBox('Start Starfish when I &log on to this computer')
        self.autostart.setChecked(engine.autostart_enabled())
        self.autostart.toggled.connect(self.set_autostart)
        start_box.addWidget(self.autostart)
        start_box.addWidget(note('Starfish then works in the background from the moment you log on.'))
        layout.addWidget(group('Starting', start_box))

        train_box = QFormLayout()
        self.device = QComboBox()
        self.device.setMaximumWidth(300)
        for value, text in DEVICES:
            self.device.addItem(text, value)
        current = (self.values.get('BABELBRAIN_FL_DEVICE') or '').strip()
        self.device.setCurrentIndex(max(0, self.device.findData(current)))
        self.device.currentIndexChanged.connect(self.set_device)
        train_box.addRow('Train on:', self.device)
        train_box.addRow('', note('The automatic choice takes the graphics card when there is one that '
                                  'works, and the processor otherwise. A change applies from the next start '
                                  'of Starfish.'))
        layout.addWidget(group('Training', train_box))

        about = QFormLayout()
        self.site_folder = QLabel(str(self.site.home))
        self.site_folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.site_uid = QLabel(self.values.get('SITE_UID', ''))
        self.site_uid.setTextInteractionFlags(Qt.TextSelectableByMouse)
        about.addRow('Site folder:', self.site_folder)
        about.addRow('Site ID:', self.site_uid)
        again = QPushButton('Set &Up Again...')
        again.clicked.connect(self.setup_again)
        again_row = QHBoxLayout()
        again_row.addWidget(again)
        again_row.addStretch(1)
        about.addRow('', again_row)
        about.addRow('', note('Set Up Again opens the setup wizard, for a new invitation or another folder '
                              'of cases.'))
        layout.addWidget(group('This computer', about))
        layout.addStretch(1)
        return page

    def _status_bar(self):
        bar = self.statusBar()
        self.sb_connection = QLabel()
        self.sb_site = QLabel()
        self.sb_state = QLabel()
        for label in (self.sb_connection, self.sb_site, self.sb_state):
            label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
            label.setContentsMargins(4, 0, 4, 0)
        bar.addPermanentWidget(self.sb_connection, 1)
        bar.addPermanentWidget(self.sb_site)
        bar.addPermanentWidget(self.sb_state)

    def _tray(self):
        self.tray = QSystemTrayIcon(classic.state_icon('stop'), self)
        menu = QMenu()
        open_action = menu.addAction('&Open Starfish Site')
        font = open_action.font()
        font.setBold(True)
        open_action.setFont(font)
        open_action.triggered.connect(self.show_window)
        menu.addSeparator()
        menu.addAction(self.act_start)
        menu.addAction(self.act_stop)
        menu.addSeparator()
        menu.addAction(self.act_exit)
        self.tray_menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_clicked)
        self.tray.show()

    def _tray_clicked(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_window()

    # Showing what happens

    def refresh_all(self):
        self.values = self.site.settings()
        self.refresh_local()
        self.refresh_router()
        self.cases.count(self.values.get('BABELBRAIN_FL_STORE'))

    def refresh_local(self):
        pid = self.site.pid()
        self.running = pid is not None
        if self.child is not None and self.child.poll() is not None and not self.running:
            # The process the app started ended before it took the lock
            self.child = None
            if self.started_at and time.monotonic() - self.started_at < 60:
                self.started_at = None
                detail = self.site.last_error()
                QMessageBox.critical(self, 'Starfish Site', 'Starfish could not start on this computer.'
                                     + ('\n\nDetails:\n' + detail[-800:] if detail else ''))
        stopping = self.site.stopping() if self.running else None
        title, text = S.headline(self.running, stopping, self.router)
        training = None
        if self.running and self.router:
            training = next((r for r in self.router.runs if r.state in S.TRAINING), None)
        state = 'stop' if not self.running else 'run' if training else 'wait'
        self.state_icon.setPixmap(classic.state_pixmap(state, 48))
        self.headline.setText(title)
        self.detail.setText(text)
        if training and training.rounds:
            self.progress.setRange(0, training.rounds)
            self.progress.setValue(max(0, (training.round or 1) - 1))
            self.progress.setFormat('Round {} of {}'.format(training.round, training.rounds))
            self.progress.setVisible(True)
        else:
            self.progress.setVisible(False)
        enrolled = self.site.enrolled(self.values)
        self.act_start.setEnabled(enrolled and not self.running)
        self.start_button.setEnabled(enrolled and not self.running)
        self.act_stop.setEnabled(self.running and not stopping)
        self.stop_button.setEnabled(self.running and not stopping)
        self.act_stop_now.setEnabled(self.running)
        self.sb_state.setText('Running' if self.running else 'Stopped')
        if self.tray is not None:
            self.tray.setIcon(classic.state_icon(state))
            self.tray.setToolTip('Starfish Site, {}'.format(title.lower() if title.startswith('Starfish')
                                                            else title))
        self.read_activity()

    def refresh_router(self):
        if self._asking:
            return
        values = self.values
        self.address.setText(values.get('ROUTER_URL', 'Not set up'))
        if not self.site.enrolled(values):
            self.show_router(S.RouterStatus(error='This computer is not set up'))
            return
        self._asking = True

        def done(result, error):
            self._asking = False
            self.show_router(result if error is None else S.RouterStatus(error=str(error)))

        widgets.in_background(lambda: S.fetch(values), done)

    def show_router(self, router):
        self.router = router
        if router.reached:
            self.connection.setText('Connected')
            self.sb_connection.setText('Connected to the main device')
            self.sb_connection.setToolTip(self.values.get('ROUTER_URL', ''))
        else:
            self.connection.setText(router.error)
            self.sb_connection.setText(router.error)
        name = router.site_name or ''
        self.known_as.setText(name or 'Unknown until the main device answers')
        self.sb_site.setText(name or 'Starfish Site')
        self.setWindowTitle('{} - Starfish Site'.format(name) if name else 'Starfish Site')
        self.projects.clear()
        for project in router.projects:
            run = next((r for r in router.runs if r.project_id == project.project_id), None)
            item = QTreeWidgetItem(self.projects, [project.name, project.role, run.round_text if run else '',
                                                   run.state_text if run else 'No active run'])
            item.setToolTip(0, project.description or project.name)
            item.setIcon(0, widgets.standard_icon('SP_FileIcon'))
        self.refresh_local()

    def read_activity(self):
        lines, self.log_offset = engine.read_log(self.site.log_path(), self.log_offset)
        if not lines:
            return
        self.log_lines.extend(lines)
        del self.log_lines[:-5 * ACTIVITY_LIMIT]
        self.rebuild_activity()

    def rebuild_activity(self, *args):
        all_lines = self.details.isChecked()
        rows = []
        for line in self.log_lines:
            parsed = engine.parse_log_line(line)
            if parsed is None:
                if all_lines and rows:
                    rows.append(('', line))
                continue
            day, clock, level, logger_name, message = parsed
            text = message if all_lines else activity_text(logger_name, level, message)
            if text:
                rows.append(('{} {}'.format(day, clock), text))
        rows = rows[-ACTIVITY_LIMIT:]
        self.activity.clear()
        for when, text in reversed(rows):
            item = QTreeWidgetItem(self.activity, [when, text])
            item.setToolTip(1, text)

    # Acting

    def start_site(self):
        if not self.site.enrolled():
            QMessageBox.information(self, 'Starfish Site', 'This computer is not set up. Choose Set Up Again '
                                    'on the Settings tab.')
            return
        if self.site.pid() is not None:
            return
        try:
            self.child = self.site.start()
        except OSError as e:
            QMessageBox.critical(self, 'Starfish Site', 'Starfish could not start.\n\n{}'.format(e))
            return
        self.started_at = time.monotonic()
        self.statusBar().showMessage('Starting Starfish...', 5000)
        QTimer.singleShot(1500, self.refresh_local)

    def stop_site(self, now=False):
        if self.site.pid() is None:
            return
        if now:
            answer = QMessageBox.warning(self, 'Starfish Site', 'Stop Now cuts the step in progress short, '
                                         'and the round on this computer starts over when Starfish runs '
                                         'again.\n\nDo you want to stop now?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.site.stop(now=now)
        self.statusBar().showMessage('Stopping Starfish...', 5000)
        self.refresh_local()

    def check_computer(self):
        CheckDialog(self.site, self).exec()

    def change_folder(self):
        start = self.values.get('BABELBRAIN_FL_STORE') or os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(self, 'Choose the Folder with Your Cases', start)
        if not path:
            return
        self.values = self.site.update(BABELBRAIN_FL_STORE=os.path.abspath(path))
        self.folder.setText(self.values['BABELBRAIN_FL_STORE'])
        self.cases.count(self.values['BABELBRAIN_FL_STORE'])
        if self.running:
            QMessageBox.information(self, 'Starfish Site', 'Starfish uses the new folder from its next start. '
                                    'Stop and start it to use the folder now.')

    def set_autostart(self, enabled):
        try:
            engine.set_autostart(enabled)
        except OSError as e:
            QMessageBox.critical(self, 'Starfish Site', 'The setting could not be saved.\n\n{}'.format(e))

    def set_device(self, index):
        self.values = self.site.update(BABELBRAIN_FL_DEVICE=self.device.itemData(index) or None)

    def contact_admin(self):
        text = self.values.get('STARFISH_ADMIN_NOTE')
        QMessageBox.information(self, 'Contact the Admin', text or 'The invitation held no note from the admin. '
                                'Ask the person who sent the invitation.')

    def setup_again(self):
        if self.on_setup_again is not None:
            self.on_setup_again()

    def open_path(self, path):
        if not path:
            return
        if not os.path.exists(path):
            QMessageBox.information(self, 'Starfish Site', '{} does not exist. Starfish makes it when it first needs it.'.format(path))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def settings_changed(self):
        """After the setup wizard ran again."""
        self.values = self.site.settings()
        self.site_uid.setText(self.values.get('SITE_UID', ''))
        self.folder.setText(self.values.get('BABELBRAIN_FL_STORE', ''))
        self.autostart.setChecked(engine.autostart_enabled())
        self.refresh_all()

    # Showing, hiding and quitting

    def showEvent(self, event):
        self.folder.setText(self.values.get('BABELBRAIN_FL_STORE', ''))
        super().showEvent(event)

    def show_window(self):
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        if self.quitting or self.tray is None:
            event.accept()
            if self.tray is None and not self.quitting:
                self.exit_app()
            return
        event.ignore()
        self.hide()
        if not self.told_about_tray:
            self.told_about_tray = True
            self.tray.showMessage('Starfish Site', 'Starfish keeps working in the background. The starfish '
                                  'icon here opens the window again.', classic.app_icon(), 8000)

    def exit_app(self):
        if self.site.pid() is not None:
            answer = QMessageBox.question(
                self, 'Starfish Site', 'Starfish is running on this computer. Exit closes this window and '
                'the starfish icon, and Starfish can keep working without them.\n\nDo you want to stop '
                'Starfish as well?', QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Yes:
                self.site.stop()
        self.quitting = True
        if self.tray is not None:
            self.tray.hide()
        from PySide6.QtWidgets import QApplication
        QApplication.quit()
