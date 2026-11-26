"""
The look of Starfish Site, after the usual Windows software of 2000 to 2010.

Qt's Windows style draws the classic raised buttons, sunken fields, dotted
tree lines and block progress bars on every system, and the palette holds the
grey and navy of Windows 2000. Tahoma at 8 points was the interface font of
Windows 2000 and XP, and macOS ships it as well. The starfish icon and the
wizard's pictures are drawn here in code, and the installers' images come
from the same functions through ``render_art``.
"""

import math
import sys

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QGuiApplication, QIcon, QLinearGradient, QPainter,
                           QPainterPath, QPainterPathStroker, QPalette, QPen, QPixmap, QPolygonF,
                           QRadialGradient)

FACE = '#D4D0C8'
NAVY = '#0A246A'
SKY = '#A6CAF0'
DESKTOP = '#3A6EA5'
SHADOW = '#808080'
DARK_SHADOW = '#404040'
INFO = '#FFFFE1'
GREY_TEXT = '#808080'
# The starfish
CORAL_LIGHT = '#FFC27A'
CORAL = '#F08A3C'
CORAL_DARK = '#B4501A'
OUTLINE = '#6E2A08'
FONT_FAMILIES = ['Tahoma', 'Verdana', 'DejaVu Sans Condensed', 'Arial']
# Tahoma 8 pt at 96 dpi, in pixels
FONT_PIXELS = 11


def palette():
    """The Windows Classic colours of Windows 2000."""
    p = QPalette()
    face, white, black = QColor(FACE), QColor('#FFFFFF'), QColor('#000000')
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role, colour in ((QPalette.Window, face), (QPalette.Button, face), (QPalette.Base, white),
                             (QPalette.AlternateBase, QColor('#F4F3EE')), (QPalette.Light, white),
                             (QPalette.Midlight, face), (QPalette.Mid, QColor(SHADOW)),
                             (QPalette.Dark, QColor(SHADOW)), (QPalette.Shadow, QColor(DARK_SHADOW)),
                             (QPalette.WindowText, black), (QPalette.ButtonText, black),
                             (QPalette.Text, black), (QPalette.BrightText, white),
                             (QPalette.Highlight, QColor(NAVY)), (QPalette.HighlightedText, white),
                             (QPalette.ToolTipBase, QColor(INFO)), (QPalette.ToolTipText, black),
                             (QPalette.Link, QColor('#0000FF')), (QPalette.LinkVisited, QColor('#800080')),
                             (QPalette.PlaceholderText, QColor(GREY_TEXT))):
            p.setColor(group, role, colour)
    for role in (QPalette.WindowText, QPalette.ButtonText, QPalette.Text):
        p.setColor(QPalette.Disabled, role, QColor(GREY_TEXT))
    p.setColor(QPalette.Inactive, QPalette.Highlight, QColor(NAVY))
    return p


def font(pixels=FONT_PIXELS, bold=False):
    """Tahoma, or the closest font this computer has, at ``pixels`` on a 96 dpi screen."""
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    screen = QGuiApplication.primaryScreen()
    dpi = screen.logicalDotsPerInchY() if screen else 96.0
    # Point sizes, which QWizard scales its titles from, and the same pixels on a Mac's 72 dpi
    f.setPointSizeF(pixels * 72.0 / dpi)
    f.setBold(bold)
    f.setStyleStrategy(QFont.PreferDefault)
    return f


def apply(app):
    """Give the whole app the classic look."""
    app.setStyle('Windows')
    # The palette also gives tool tips their pale yellow. A style sheet for the whole app would wrap
    # the Windows style in another and change how every widget is drawn.
    app.setPalette(palette())
    app.setFont(font())
    if sys.platform == 'darwin':
        # Icons in menus, as on Windows
        app.setAttribute(Qt.AA_DontShowIconsInMenus, False)


# The starfish

def starfish_path(cx, cy, radius, rotation=0.0):
    """A plump five-armed starfish, centred at (cx, cy), reaching ``radius`` at its tips."""
    width = radius * 0.30
    outer = radius - width / 2
    inner = outer * 0.42
    points = []
    for k in range(10):
        angle = math.radians(-90 + rotation + 36 * k)
        r = outer if k % 2 == 0 else inner
        points.append(QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle)))
    star = QPainterPath()
    star.addPolygon(QPolygonF(points))
    star.closeSubpath()
    stroker = QPainterPathStroker()
    stroker.setWidth(width)
    stroker.setJoinStyle(Qt.RoundJoin)
    stroker.setCapStyle(Qt.RoundCap)
    return star.united(stroker.createStroke(star)).simplified()


def draw_starfish(painter, cx, cy, radius, rotation=0.0, colours=None, outline=True, dots=None):
    """Paint the starfish with a soft radial shading, a dark outline and, when large, its dots."""
    light, mid, dark, line = colours or (CORAL_LIGHT, CORAL, CORAL_DARK, OUTLINE)
    path = starfish_path(cx, cy, radius, rotation)
    gradient = QRadialGradient(QPointF(cx - radius * 0.15, cy - radius * 0.2), radius * 1.05)
    gradient.setColorAt(0.0, QColor(light))
    gradient.setColorAt(0.55, QColor(mid))
    gradient.setColorAt(1.0, QColor(dark))
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setBrush(QBrush(gradient))
    if outline:
        painter.setPen(QPen(QColor(line), max(1.0, radius / 18.0), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    else:
        painter.setPen(Qt.NoPen)
    painter.drawPath(path)
    if dots is None:
        dots = radius >= 20
    if dots:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 236, 200, 200))
        dot = max(1.0, radius * 0.045)
        for k in range(5):
            angle = math.radians(-90 + rotation + 72 * k)
            for step in (0.25, 0.45, 0.63):
                x = cx + radius * step * math.cos(angle)
                y = cy + radius * step * math.sin(angle)
                painter.drawEllipse(QPointF(x, y), dot, dot)
        painter.setBrush(QColor(255, 236, 200, 230))
        painter.drawEllipse(QPointF(cx, cy), dot * 1.6, dot * 1.6)
    painter.restore()


def _canvas(width, height):
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    return pixmap


def starfish_pixmap(size, greyed=False):
    """The app's icon at ``size`` pixels square."""
    pixmap = _canvas(size, size)
    painter = QPainter(pixmap)
    colours = ('#E8E8E8', '#B8B8B8', '#8C8C8C', '#505050') if greyed else None
    margin = max(0.5, size * 0.03)
    draw_starfish(painter, size / 2, size / 2 + size * 0.02, size / 2 - margin, colours=colours)
    painter.end()
    return pixmap


def app_icon():
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        icon.addPixmap(starfish_pixmap(size))
    return icon


def _badge(painter, kind, x, y, size):
    """A small round badge in the corner of an icon, green with a triangle or red with a square."""
    fill = {'run': '#2E9A2E', 'stop': '#C62828', 'wait': '#E0A100'}[kind]
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(QPen(QColor('#FFFFFF'), max(1.0, size / 10.0)))
    painter.setBrush(QColor(fill))
    painter.drawEllipse(QRectF(x, y, size, size))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor('#FFFFFF'))
    inset = size * 0.3
    if kind == 'run':
        painter.drawPolygon(QPolygonF([QPointF(x + inset * 1.15, y + inset), QPointF(x + size - inset * 0.9, y + size / 2),
                                       QPointF(x + inset * 1.15, y + size - inset)]))
    elif kind == 'stop':
        painter.drawRect(QRectF(x + inset, y + inset, size - 2 * inset, size - 2 * inset))
    else:
        bar = size * 0.12
        painter.drawRect(QRectF(x + size / 2 - bar * 1.8, y + inset, bar, size - 2 * inset))
        painter.drawRect(QRectF(x + size / 2 + bar * 0.8, y + inset, bar, size - 2 * inset))
    painter.restore()


def state_pixmap(state, size=32):
    """The starfish with a badge for a site that runs, waits or stopped, or greyed before setup."""
    pixmap = starfish_pixmap(size, greyed=state == 'setup')
    if state in ('run', 'stop', 'wait'):
        painter = QPainter(pixmap)
        badge = size * 0.5
        _badge(painter, state, size - badge, size - badge, badge)
        painter.end()
    return pixmap


def state_icon(state):
    icon = QIcon()
    for size in (16, 24, 32, 48):
        icon.addPixmap(state_pixmap(state, size))
    return icon


def action_pixmap(kind, size=16):
    """Toolbar pictures for Start, a green triangle, and Stop, a red square, with dark outlines."""
    pixmap = _canvas(size, size)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    s = size / 16.0
    if kind == 'start':
        gradient = QLinearGradient(0, 0, 0, size)
        gradient.setColorAt(0, QColor('#7CD66B'))
        gradient.setColorAt(1, QColor('#1E7A1E'))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor('#0F4D0F'), max(1.0, s)))
        painter.drawPolygon(QPolygonF([QPointF(4 * s, 2 * s), QPointF(13.5 * s, 8 * s), QPointF(4 * s, 14 * s)]))
    elif kind == 'stop':
        gradient = QLinearGradient(0, 0, 0, size)
        gradient.setColorAt(0, QColor('#F07A6A'))
        gradient.setColorAt(1, QColor('#A31B1B'))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor('#5E0B0B'), max(1.0, s)))
        painter.drawRect(QRectF(3 * s, 3 * s, 10 * s, 10 * s))
    elif kind in ('online', 'offline', 'unknown'):
        fill = {'online': '#2E9A2E', 'offline': '#C62828', 'unknown': '#A0A0A0'}[kind]
        gradient = QRadialGradient(QPointF(6 * s, 6 * s), 7 * s)
        gradient.setColorAt(0, QColor('#FFFFFF'))
        gradient.setColorAt(0.35, QColor(fill).lighter(130))
        gradient.setColorAt(1, QColor(fill))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(fill).darker(170), max(1.0, s * 0.8)))
        painter.drawEllipse(QRectF(3.5 * s, 3.5 * s, 9 * s, 9 * s))
    painter.end()
    return pixmap


def action_icon(kind):
    icon = QIcon()
    for size in (16, 24, 32):
        icon.addPixmap(action_pixmap(kind, size))
    return icon


# The wizard's pictures

WATERMARK_SIZE = (164, 314)
LOGO_SIZE = 49


def watermark(scale=1.0):
    """
    The tall picture on the left of the first and last wizard pages, as in Windows 2000 setups.

    ``scale`` gives the pixels per point, 1.5 and 2 for the setup program on screens set to
    150 and 200 percent.
    """
    width, height = WATERMARK_SIZE
    pixmap = QPixmap(round(width * scale), round(height * scale))
    pixmap.setDevicePixelRatio(scale)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    gradient = QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, QColor(NAVY))
    gradient.setColorAt(0.55, QColor(DESKTOP))
    gradient.setColorAt(1.0, QColor(SKY))
    painter.fillRect(QRectF(0, 0, width, height), gradient)
    # Faint rings of water behind the big starfish
    painter.setPen(QPen(QColor(255, 255, 255, 40), 1.2))
    painter.setBrush(Qt.NoBrush)
    for r in (40, 62, 86, 112):
        painter.drawEllipse(QPointF(width * 0.62, height * 0.70), r, r * 0.55)
    draw_starfish(painter, width * 0.60, height * 0.69, width * 0.43, rotation=12)
    draw_starfish(painter, width * 0.18, height * 0.93, width * 0.15, rotation=-20)
    painter.setPen(QColor('#FFFFFF'))
    title = font(22, bold=True)
    painter.setFont(title)
    painter.drawText(QRectF(14, 22, width - 20, 32), Qt.AlignLeft | Qt.AlignVCenter, 'Starfish')
    painter.setFont(font(17))
    painter.drawText(QRectF(15, 52, width - 20, 26), Qt.AlignLeft | Qt.AlignVCenter, 'Site')
    painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
    painter.drawLine(QPointF(15, 84), QPointF(width - 18, 84))
    painter.end()
    return pixmap


def logo(size=LOGO_SIZE, scale=1.0, background=None):
    """The small picture at the right of a wizard page's white header."""
    pixmap = QPixmap(round(size * scale), round(size * scale))
    pixmap.fill(QColor(background) if background else Qt.transparent)
    pixmap.setDevicePixelRatio(scale)
    painter = QPainter(pixmap)
    draw_starfish(painter, size / 2, size / 2 + 1, size / 2 - 3, rotation=8)
    painter.end()
    return pixmap
