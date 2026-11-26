"""
The setup wizard that Starfish Site opens on its first start.

Its five pages follow the mockup, in the form of a Windows 2000 wizard. A
welcome page with the tall picture, a page that opens the invitation and
enrols this computer, a page for the folder of cases, a page that checks the
computer, and a last page with the tall picture again that starts the site.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressDialog, QPushButton,
                               QRadioButton, QVBoxLayout, QWizard, QWizardPage)

from starfish_site import cli
from starfish_site import config as C
from starfish_site import invitation as I
from starfish_site.app import classic, engine, widgets

PAGE_WELCOME, PAGE_JOIN, PAGE_CASES, PAGE_CHECK, PAGE_FINISH = range(5)
WIZARD_SIZE = (580, 440)


def big_title(text):
    """The large bold title of a page with the tall picture, as Windows 2000 wizards have it."""
    return '<span style="font-size:16px; font-weight:bold">{}</span>'.format(text)


def paragraph(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    return label


def settings_text(text):
    """Text for a value of site.env, which holds no straight quote and no line break."""
    return ' '.join(text.replace("'", '’').split())


def friendly_enrol_error(error):
    """What to tell a person when enrolment failed, from :class:`starfish_site.cli.EnrolError`."""
    return {
        'url': 'The address of the main device is not complete. It needs to start with https:// '
               'or http://, for example https://10.0.0.5:8443.',
        'certificate': 'The main device\'s security certificate is not trusted by this computer. '
                       'Use the invitation file from the admin, which carries the right certificate.',
        'unreachable': 'This computer cannot reach the main device. Check that the computer is '
                       'connected to the network, and ask the admin whether the main device is running.',
        'refused': 'The main device refused the invitation. It may have been used already or have '
                   'expired. Ask the admin for a new one.',
    }.get(error.kind, 'The main device could not set up this computer. If another site already uses '
                      'this name, try another one.\n\nDetails: {}'.format(error))


class WelcomePage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle(big_title('Welcome to the Starfish Site Setup Wizard'))
        self.setPixmap(QWizard.WatermarkPixmap, classic.watermark())
        layout = QVBoxLayout(self)
        layout.addWidget(paragraph('This wizard connects this computer to the testbed in a few short steps.'))
        layout.addWidget(paragraph('Starfish Site lets this computer help train a shared BabelBrain model '
                                   'together with other hospitals. The files of your cases stay on this '
                                   'computer. Only what the model learns from them goes to the main device.'))
        layout.addWidget(paragraph('Before you start, have these two things ready.'))
        layout.addWidget(paragraph('    1.  The invitation file from the admin of the main device.\n'
                                   '    2.  The folder with this computer\'s cases.'))
        layout.addStretch(1)
        layout.addWidget(paragraph('To continue, click Next.'))


class JoinPage(QWizardPage):
    """Opens the invitation, names this computer and enrols it."""

    def __init__(self, site, invitation_path=None):
        super().__init__()
        self.site = site
        self.invitation = None
        self.setTitle('Open your invitation')
        self.setSubTitle('The invitation is a small file from the admin of the main device. It connects '
                         'this computer to the testbed and works only once.')
        layout = QVBoxLayout(self)
        self.already = QCheckBox()
        self.already.toggled.connect(self._mode_changed)
        layout.addWidget(self.already)

        self.choice = QButtonGroup(self)
        self.by_file = QRadioButton('I have an &invitation file')
        self.by_code = QRadioButton('I have the &address and the code from the admin')
        self.by_file.setChecked(True)
        self.choice.addButton(self.by_file)
        self.choice.addButton(self.by_code)
        self.choice.buttonToggled.connect(self._mode_changed)
        layout.addWidget(self.by_file)
        row = QHBoxLayout()
        row.setContentsMargins(20, 0, 0, 0)
        self.file = QLineEdit()
        self.file.setReadOnly(True)
        self.file.setPlaceholderText('No file chosen')
        row.addWidget(self.file, 1)
        self.browse = QPushButton('B&rowse...')
        self.browse.clicked.connect(self.pick_file)
        row.addWidget(self.browse)
        layout.addLayout(row)

        layout.addWidget(self.by_code)
        grid = QGridLayout()
        grid.setContentsMargins(20, 0, 0, 0)
        self.address = QLineEdit()
        self.address.setPlaceholderText('https://...')
        self.code = QLineEdit()
        grid.addWidget(QLabel('Main device:'), 0, 0)
        grid.addWidget(self.address, 0, 1)
        grid.addWidget(QLabel('Code:'), 1, 0)
        grid.addWidget(self.code, 1, 1)
        layout.addLayout(grid)

        self.details = QGroupBox('Invitation')
        form = QFormLayout(self.details)
        self.testbed, self.device, self.valid, self.note = QLabel(), QLabel(), QLabel(), QLabel()
        for label in (self.testbed, self.device, self.valid, self.note):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow('Testbed:', self.testbed)
        form.addRow('Main device:', self.device)
        form.addRow('Valid until:', self.valid)
        form.addRow('Note:', self.note)
        layout.addWidget(self.details)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel('&Name this computer:'))
        self.name = QLineEdit()
        self.name.setMaxLength(100)
        name_row.itemAt(0).widget().setBuddy(self.name)
        name_row.addWidget(self.name, 1)
        layout.addLayout(name_row)
        self.name_hint = paragraph('The admin and the other sites see this name, for example Hospital A, '
                                   'Room 3.')
        self.name_hint.setStyleSheet('color: #404040')
        layout.addWidget(self.name_hint)
        layout.addStretch(1)
        for edit in (self.address, self.code, self.name):
            edit.textChanged.connect(self.completeChanged)
        if invitation_path:
            self.load(invitation_path)
        self._mode_changed()

    def initializePage(self):
        values = self.site.settings()
        enrolled = self.site.enrolled(values)
        self.already.setVisible(enrolled)
        if enrolled:
            self.already.setText('Keep the connection this computer has, to {}'.format(values['ROUTER_URL']))
            if self.invitation is None:
                self.already.setChecked(True)
        self._mode_changed()

    def _mode_changed(self, *args):
        keep = self.already.isVisible() and self.already.isChecked()
        by_file = self.by_file.isChecked()
        for widget in (self.by_file, self.by_code, self.name, self.name_hint):
            widget.setEnabled(not keep)
        self.file.setEnabled(not keep and by_file)
        self.browse.setEnabled(not keep and by_file)
        self.address.setEnabled(not keep and not by_file)
        self.code.setEnabled(not keep and not by_file)
        self.details.setVisible(by_file and self.invitation is not None and not keep)
        self.completeChanged.emit()

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Invitation', os.path.expanduser('~'),
            'Starfish invitations (*{});;All files (*)'.format(I.SUFFIX))
        if path:
            self.load(path)

    def load(self, path):
        try:
            found = I.read(path)
        except I.InvitationError as e:
            QMessageBox.critical(self, 'Starfish Site Setup', str(e))
            return False
        self.invitation = found
        self.by_file.setChecked(True)
        self.file.setText(path)
        self.testbed.setText(found.testbed or 'Not named')
        self.device.setText(found.router)
        self.valid.setText(I.describe_time(found.expires_at) + (', expired' if found.expired() else ''))
        self.note.setText(found.note or 'None')
        if self.already.isVisible():
            self.already.setChecked(False)
        self._mode_changed()
        return True

    def isComplete(self):
        if self.already.isVisible() and self.already.isChecked():
            return True
        if not self.name.text().strip():
            return False
        if self.by_file.isChecked():
            return self.invitation is not None
        return bool(self.address.text().strip() and self.code.text().strip())

    def validatePage(self):
        if self.already.isVisible() and self.already.isChecked():
            return True
        if self.by_file.isChecked():
            found = self.invitation
            if found.expired():
                QMessageBox.warning(self, 'Starfish Site Setup', 'This invitation expired on {}. Ask the '
                                    'admin for a new one.'.format(I.describe_time(found.expires_at)))
                return False
            router, code, pem, note = found.router, found.code, found.ca_cert, found.note
        else:
            router, code, pem, note = self.address.text().strip(), self.code.text().strip(), None, ''
        name = self.name.text().strip()
        home = self.site.home
        force = self.site.enrolled()

        def work():
            ca_path = None
            if pem:
                home.mkdir(parents=True, exist_ok=True)
                ca_path = home / C.CA_NAME
                ca_path.write_text(pem, encoding='ascii')
            answer, _ = cli.enrol_site(home, router, code, name, ca_cert=str(ca_path) if ca_path else None,
                                       force=force)
            if note:
                try:
                    self.site.update(STARFISH_ADMIN_NOTE=settings_text(note))
                except (OSError, ValueError):
                    # The site is enrolled, and the note is only for Contact the Admin
                    pass
            return answer

        return run_with_progress(self, 'Connecting to the main device...', work, friendly_enrol_error,
                                 cli.EnrolError)


def run_with_progress(parent, text, work, explain, expected):
    """Run ``work`` in the background behind a busy dialog. True when it worked, else say why."""
    from PySide6.QtCore import QEventLoop

    dialog = QProgressDialog(text, None, 0, 0, parent)
    dialog.setWindowTitle('Starfish Site Setup')
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setCancelButton(None)
    loop = QEventLoop()
    outcome = {}

    def done(result, error):
        outcome['result'], outcome['error'] = result, error
        loop.quit()

    widgets.in_background(work, done)
    dialog.show()
    loop.exec()
    dialog.close()
    error = outcome.get('error')
    if error is None:
        return True
    if isinstance(error, expected):
        QMessageBox.critical(parent, 'Starfish Site Setup', explain(error))
    else:
        QMessageBox.critical(parent, 'Starfish Site Setup', 'Something went wrong.\n\n{}: {}'.format(
            error.__class__.__name__, error))
    return False


class CasesPage(QWizardPage):

    def __init__(self, site):
        super().__init__()
        self.site = site
        self.setTitle('Choose the folder with your cases')
        self.setSubTitle('Starfish reads the cases in this folder. It never changes, moves or sends them.')
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel('&Folder:'))
        self.folder = QLineEdit()
        self.folder.setReadOnly(True)
        self.folder.setPlaceholderText('No folder chosen')
        row.itemAt(0).widget().setBuddy(self.folder)
        row.addWidget(self.folder, 1)
        browse = QPushButton('B&rowse...')
        browse.clicked.connect(self.pick)
        row.addWidget(browse)
        layout.addLayout(row)
        self.found = widgets.CasesBox()
        self.found.counted.connect(lambda total: self.completeChanged.emit())
        layout.addWidget(self.found)
        layout.addWidget(paragraph('New cases saved in this folder join by themselves at the next round.'))
        layout.addStretch(1)

    def initializePage(self):
        current = self.site.settings().get('BABELBRAIN_FL_STORE')
        if current and not self.folder.text():
            self.folder.setText(current)
            self.found.count(current)

    def pick(self):
        start = self.folder.text() or os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(self, 'Choose the Folder with Your Cases', start)
        if path:
            self.folder.setText(os.path.normpath(path))
            self.found.count(self.folder.text())
            self.completeChanged.emit()

    def isComplete(self):
        return bool(self.folder.text()) and self.found.total is not None

    def validatePage(self):
        if not self.found.total:
            answer = QMessageBox.question(
                self, 'Starfish Site Setup', 'No usable case was found in this folder. Starfish can train '
                'only once cases are saved there.\n\nDo you want to keep this folder anyway?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return False
        self.site.update(BABELBRAIN_FL_STORE=os.path.abspath(self.folder.text()))
        return True


class CheckPage(QWizardPage):

    def __init__(self, site):
        super().__init__()
        self.setTitle('Check this computer')
        self.setSubTitle('Starfish asks the main device one test question and trains a small model on '
                         'made-up data. Your cases never leave this computer.')
        layout = QVBoxLayout(self)
        self.panel = widgets.CheckPanel(site)
        self.panel.finished.connect(lambda failed, warned: self.completeChanged.emit())
        layout.addWidget(self.panel)

    def initializePage(self):
        if not self.panel.ran and self.panel.process is None:
            self.panel.start()

    def isComplete(self):
        return self.panel.ran and self.panel.process is None

    def validatePage(self):
        if self.panel.failed:
            answer = QMessageBox.warning(
                self, 'Starfish Site Setup', 'The check found problems, and Starfish may not work on this '
                'computer until they are solved. You can run the check again later from the main window.'
                '\n\nDo you want to finish the setup anyway?', QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            return answer == QMessageBox.Yes
        return True


class FinishPage(QWizardPage):

    def __init__(self):
        super().__init__()
        self.setTitle(big_title('Completing the Starfish Site Setup Wizard'))
        self.setPixmap(QWizard.WatermarkPixmap, classic.watermark())
        layout = QVBoxLayout(self)
        layout.addWidget(paragraph('This computer is ready.'))
        layout.addWidget(paragraph('Starfish works in the background while you use the computer as usual. '
                                   'The starfish icon {} shows that it runs, and a click on it opens the main '
                                   'window.'.format(widgets.tray_place())))
        self.start_now = QCheckBox('Start Starfish &now')
        self.start_now.setChecked(True)
        self.at_logon = QCheckBox('Start Starfish when I &log on to this computer')
        self.at_logon.setChecked(True)
        layout.addSpacing(8)
        layout.addWidget(self.start_now)
        layout.addWidget(self.at_logon)
        layout.addStretch(1)
        layout.addWidget(paragraph('To close this wizard, click Finish.'))


class SetupWizard(QWizard):
    """The first-start wizard. After Finish, ``start_now`` and ``at_logon`` hold the choices."""

    def __init__(self, site, invitation_path=None, parent=None):
        super().__init__(parent)
        self.site = site
        self.setWindowTitle('Starfish Site Setup')
        self.setWindowIcon(classic.app_icon())
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.NoBackButtonOnLastPage, True)
        self.setPixmap(QWizard.LogoPixmap, classic.logo())
        self.setPage(PAGE_WELCOME, WelcomePage())
        self.join = JoinPage(site, invitation_path)
        self.setPage(PAGE_JOIN, self.join)
        self.setPage(PAGE_CASES, CasesPage(site))
        self.check = CheckPage(site)
        self.setPage(PAGE_CHECK, self.check)
        self.finish_page = FinishPage()
        self.setPage(PAGE_FINISH, self.finish_page)
        self.setMinimumSize(*WIZARD_SIZE)
        self.resize(*WIZARD_SIZE)
        if invitation_path and self.join.invitation is not None:
            self.setStartId(PAGE_JOIN)

    @property
    def start_now(self):
        return self.finish_page.start_now.isChecked()

    @property
    def at_logon(self):
        return self.finish_page.at_logon.isChecked()

    def open_invitation(self, path):
        """An invitation opened while the wizard is on screen, by a double-click on its file."""
        if self.join.load(path) and self.currentId() == PAGE_WELCOME:
            self.next()

    def reject(self):
        if self.currentId() in (PAGE_CASES, PAGE_CHECK):
            answer = QMessageBox.question(
                self, 'Starfish Site Setup', 'This computer is connected to the main device, and the setup '
                'is not finished. Starfish Site opens this wizard again at its next start.\n\nDo you want '
                'to quit the setup?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        super().reject()
