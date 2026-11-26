"""
Build Starfish Site for this system.

    python starfish-site/packaging/build.py [--out DIR] [--no-installer]

Renders the icon and the setup program's pictures from the app's own drawing
code, freezes the app and the ``starfish-site`` command with PyInstaller, and
wraps them in the setup program of Inno Setup on Windows or in a disk image
on macOS. On Linux it stops after PyInstaller, which serves tests. Runs in an
environment that has the controller, starfish-site with its ``app`` extra,
torch and PyInstaller installed, and Inno Setup 6 on Windows.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
REPO = SITE.parent
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SCALES = ((1.0, '100'), (1.5, '150'), (2.0, '200'))
# The small picture in the header of Inno Setup's classic wizard, in points
HEADER_SIZE = (55, 58)


def version():
    text = (SITE / 'starfish_site' / '__init__.py').read_text(encoding='utf-8')
    return re.search(r"__version__ = '([^']+)'", text).group(1)


def render_art(folder):
    """The icon in every size, as .ico and .icns, and the wizard's pictures as BMP files."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
    from PySide6.QtWidgets import QApplication

    from starfish_site.app import classic

    app = QApplication.instance() or QApplication([sys.argv[0]])
    classic.apply(app)
    folder.mkdir(parents=True, exist_ok=True)
    for size in ICON_SIZES + (512, 1024):
        classic.starfish_pixmap(size).save(str(folder / 'icon-{}.png'.format(size)))

    from PIL import Image
    largest = Image.open(folder / 'icon-256.png')
    largest.save(folder / 'starfish-site.ico', sizes=[(s, s) for s in ICON_SIZES])
    if sys.platform == 'darwin':
        iconset = folder / 'starfish-site.iconset'
        iconset.mkdir(exist_ok=True)
        for size in (16, 32, 128, 256, 512):
            shutil.copy(folder / 'icon-{}.png'.format(size), iconset / 'icon_{0}x{0}.png'.format(size))
            shutil.copy(folder / 'icon-{}.png'.format(size * 2), iconset / 'icon_{0}x{0}@2x.png'.format(size))
        subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(folder / 'starfish-site.icns')],
                       check=True)

    def save_bmp(pixmap, path):
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        if not image.save(str(path), 'BMP'):
            raise RuntimeError('could not write {}'.format(path))

    for scale, name in SCALES:
        save_bmp(classic.watermark(scale), folder / 'wizard-{}.bmp'.format(name))
        width, height = HEADER_SIZE
        header = QPixmap(round(width * scale), round(height * scale))
        header.fill(QColor('#FFFFFF'))
        header.setDevicePixelRatio(scale)
        painter = QPainter(header)
        classic.draw_starfish(painter, width / 2, height / 2 + 1, min(width, height) / 2 - 4, rotation=8)
        painter.end()
        save_bmp(header, folder / 'header-{}.bmp'.format(name))
    return folder


def freeze(out, work, art):
    env = dict(os.environ, STARFISH_SITE_ART=str(art), STARFISH_SITE_VERSION=version())
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--log-level', 'WARN',
                    '--distpath', str(out), '--workpath', str(work), str(HERE / 'starfish-site.spec')],
                   check=True, env=env)
    if sys.platform == 'darwin':
        return out / 'Starfish Site.app'
    return out / 'Starfish Site'


def find_iscc():
    found = shutil.which('iscc') or shutil.which('ISCC')
    if found:
        return found
    for base in (os.environ.get('ProgramFiles(x86)'), os.environ.get('ProgramFiles')):
        if base and (Path(base) / 'Inno Setup 6' / 'ISCC.exe').is_file():
            return str(Path(base) / 'Inno Setup 6' / 'ISCC.exe')
    raise SystemExit('Inno Setup 6 is not installed, see https://jrsoftware.org/isinfo.php')


def windows_setup(frozen, art, out):
    subprocess.run([find_iscc(), '/Q', '/DAppVersion=' + version(), '/DSourceDir=' + str(frozen),
                    '/DArtDir=' + str(art), '/DOutputDir=' + str(out), '/DLicenseFile=' + str(REPO / 'LICENSE'),
                    str(HERE / 'starfish-site.iss')], check=True)
    return out / 'StarfishSite-Setup-{}.exe'.format(version())


def mac_disk_image(bundle, out):
    stage = out / 'dmg'
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    subprocess.run(['ditto', str(bundle), str(stage / bundle.name)], check=True)
    os.symlink('/Applications', stage / 'Applications')
    image = out / 'StarfishSite-{}.dmg'.format(version())
    if image.exists():
        image.unlink()
    subprocess.run(['hdiutil', 'create', '-volname', 'Starfish Site', '-srcfolder', str(stage), '-ov',
                    '-format', 'UDZO', str(image)], check=True)
    shutil.rmtree(stage, ignore_errors=True)
    return image


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default=str(REPO / 'dist'), help='output folder, default dist/ in the repository')
    p.add_argument('--no-installer', action='store_true', help='stop after PyInstaller')
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    art = render_art(out / 'art')
    frozen = freeze(out, out / 'work', art)
    print('Froze', frozen)
    if args.no_installer:
        return 0
    if sys.platform == 'win32':
        print('Wrote', windows_setup(frozen, art, out))
    elif sys.platform == 'darwin':
        print('Wrote', mac_disk_image(frozen, out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
