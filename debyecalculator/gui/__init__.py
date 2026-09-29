"""
Desktop GUI for DebyeCalculator (Qt, via PySide6 and pyqtgraph).

Start it with `debyecalculator-gui [structure files...]` or `python -m debyecalculator.gui`.
"""

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        from PySide6.QtWidgets import QApplication
        import pyqtgraph as pg
    except ImportError:
        sys.stderr.write(
            'The DebyeCalculator GUI needs PySide6 and pyqtgraph. Install them with\n'
            '    pip install "debyecalculator[gui]"\n'
        )
        return 1

    from debyecalculator.gui.window import MainWindow

    pg.setConfigOptions(antialias=True, foreground='k')
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('DebyeCalculator')
    window = MainWindow(files=argv)
    window.show()
    return app.exec()
