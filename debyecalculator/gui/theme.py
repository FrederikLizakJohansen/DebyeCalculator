"""
Light and dark application themes (Qt Fusion style with explicit palettes).
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

THEME_MODES = ('System', 'Light', 'Dark')


def system_is_dark() -> bool:
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, 'colorScheme', None)
    if scheme is not None:
        return scheme() == Qt.ColorScheme.Dark
    # Qt < 6.5: infer from the default window color
    return QApplication.palette().color(QPalette.Window).lightness() < 128


def _palette(dark: bool) -> QPalette:
    palette = QPalette()
    if dark:
        colors = {
            QPalette.Window: '#2b2d30', QPalette.WindowText: '#dfe1e5', QPalette.Base: '#1e1f22',
            QPalette.AlternateBase: '#26282b', QPalette.ToolTipBase: '#3c3f41', QPalette.ToolTipText: '#dfe1e5',
            QPalette.Text: '#dfe1e5', QPalette.Button: '#393b40', QPalette.ButtonText: '#dfe1e5',
            QPalette.BrightText: '#ff6b68', QPalette.Link: '#589df6', QPalette.Highlight: '#2f65ca',
            QPalette.HighlightedText: '#ffffff', QPalette.PlaceholderText: '#8c8f94', QPalette.Mid: '#4e5157',
            QPalette.Dark: '#1b1c1f', QPalette.Light: '#4e5157', QPalette.Shadow: '#111214',
        }
        disabled = '#6f737a'
    else:
        colors = {
            QPalette.Window: '#f3f3f3', QPalette.WindowText: '#1f1f1f', QPalette.Base: '#ffffff',
            QPalette.AlternateBase: '#f5f7fa', QPalette.ToolTipBase: '#ffffdc', QPalette.ToolTipText: '#1f1f1f',
            QPalette.Text: '#1f1f1f', QPalette.Button: '#e9e9e9', QPalette.ButtonText: '#1f1f1f',
            QPalette.BrightText: '#d32f2f', QPalette.Link: '#1565c0', QPalette.Highlight: '#3d7fd9',
            QPalette.HighlightedText: '#ffffff', QPalette.PlaceholderText: '#8a8a8a', QPalette.Mid: '#c4c4c4',
            QPalette.Dark: '#a0a0a0', QPalette.Light: '#ffffff', QPalette.Shadow: '#7a7a7a',
        }
        disabled = '#9a9a9a'
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(disabled))
    return palette


def apply_theme(mode: str) -> bool:
    """
    Apply 'System', 'Light' or 'Dark' to the application and return whether the result is dark.
    """
    app = QApplication.instance()
    dark = system_is_dark() if mode == 'System' else mode == 'Dark'
    app.setStyle('Fusion')
    app.setPalette(_palette(dark))
    app.setStyleSheet(_style_sheet(dark))
    return dark


def _style_sheet(dark: bool) -> str:
    # Fusion draws check box outlines from the window color, which is nearly invisible on a dark window
    check = (Path(__file__).parent / 'assets' / 'check.svg').as_posix()
    border, base, accent, disabled = (('#8c8f94', '#1e1f22', '#3d7fd9', '#4e5157') if dark
                                      else ('#8a8a8a', '#ffffff', '#3d7fd9', '#c4c4c4'))
    return f"""
        QCheckBox::indicator, QTableView::indicator {{
            width: 14px; height: 14px; border: 1px solid {border}; border-radius: 3px; background: {base};
        }}
        QCheckBox::indicator:checked, QTableView::indicator:checked {{
            background: {accent}; border-color: {accent}; image: url({check});
        }}
        QCheckBox::indicator:disabled {{ border-color: {disabled}; }}
        QCheckBox::indicator:checked:disabled {{ background: {disabled}; border-color: {disabled}; }}
    """
