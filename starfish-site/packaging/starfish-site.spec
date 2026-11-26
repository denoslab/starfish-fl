# PyInstaller recipe for Starfish Site, run by build.py.
#
# One folder holds two programs that share one Python and its libraries.
# "Starfish Site" is the windowed app, and "starfish-site" the command line,
# which the app runs for the site's own process and for the check. On macOS
# the folder becomes the app bundle "Starfish Site.app".
#
# build.py sets STARFISH_SITE_ART to the folder of the rendered icons and
# STARFISH_SITE_VERSION to the version of starfish_site.

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = os.path.dirname(os.path.abspath(SPEC))
ART = os.environ['STARFISH_SITE_ART']
VERSION = os.environ.get('STARFISH_SITE_VERSION', '0.0.0')
BUNDLE_ID = 'com.denoslab.starfish-site'
if sys.platform == 'win32':
    ICON = os.path.join(ART, 'starfish-site.ico')
elif sys.platform == 'darwin':
    ICON = os.path.join(ART, 'starfish-site.icns')
else:
    ICON = None

# The controller loads a project's task by its name, and Celery its parts, which
# PyInstaller cannot follow. The app only runs the BabelBrain FNO task.
hidden = (collect_submodules('starfish_site')
          + collect_submodules('starfish.controller.tasks.babel_brain_fno')
          + collect_submodules('celery') + collect_submodules('kombu.transport')
          + ['starfish.settings', 'starfish.controller.site_status_task'])
# setuptools and pkg_resources come in only through torch's tools for building C++ extensions,
# which a site never uses, and PyInstaller's start-up hook for pkg_resources fails without
# the parts that setuptools vendors
excludes = ['tkinter', 'matplotlib', 'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx',
            'setuptools', 'pkg_resources', 'torch.utils.tensorboard',
            'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.Qt3DCore', 'PySide6.QtWebEngineCore']

a = Analysis(
    [os.path.join(HERE, 'launch_app.py'), os.path.join(HERE, 'launch_command.py')],
    hiddenimports=hidden,
    # The sample store's manifest schema
    datas=collect_data_files('starfish.controller.tasks.babel_brain_fno'),
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
# The runtime hooks come first in a.scripts, then the two launch scripts in order
hooks, app_script, command_script = a.scripts[:-2], a.scripts[-2], a.scripts[-1]
app_exe = EXE(pyz, hooks + [app_script], [], exclude_binaries=True, name='Starfish Site',
              console=False, icon=ICON, upx=False)
command_exe = EXE(pyz, hooks + [command_script], [], exclude_binaries=True, name='starfish-site',
                  console=True, icon=ICON, upx=False)
coll = COLLECT(app_exe, command_exe, a.binaries, a.datas, name='Starfish Site', upx=False)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Starfish Site.app',
        icon=ICON,
        bundle_identifier=BUNDLE_ID,
        version=VERSION,
        info_plist={
            'CFBundleName': 'Starfish Site',
            'CFBundleDisplayName': 'Starfish Site',
            'CFBundleShortVersionString': VERSION,
            'CFBundleVersion': VERSION,
            'LSMinimumSystemVersion': '11.0',
            'NSHighResolutionCapable': True,
            'NSHumanReadableCopyright': 'Starfish-FL, Apache License 2.0',
            'CFBundleDocumentTypes': [{
                'CFBundleTypeName': 'Starfish Invitation',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Owner',
                'LSItemContentTypes': [BUNDLE_ID + '.invitation'],
            }],
            'UTExportedTypeDeclarations': [{
                'UTTypeIdentifier': BUNDLE_ID + '.invitation',
                'UTTypeDescription': 'Starfish Invitation',
                'UTTypeConformsTo': ['public.data'],
                'UTTypeTagSpecification': {'public.filename-extension': ['starfish-invitation']},
            }],
        },
    )
