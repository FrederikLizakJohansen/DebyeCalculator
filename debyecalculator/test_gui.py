import json

import numpy as np
import pytest
import torch

from debyecalculator import DebyeCalculator
from debyecalculator.gui.engine import Engine, Parameters, StructureSpec

CIF = 'debyecalculator/data/AntiFluorite_Co2O.cif'
XYZ = 'debyecalculator/data/AntiFluorite_Co2O_r10.xyz'


def test_engine_matches_calculator():
    result = Engine().compute(Parameters(), [StructureSpec(CIF, radius=8)])[0]
    expected = DebyeCalculator(device='cpu')._get_all(CIF, radii=8)
    for name in 'isfg':
        assert np.allclose(getattr(result, name), getattr(expected, name), rtol=1e-5, atol=1e-6), name
    assert result.element_pairs == ['Co-Co', 'Co-O', 'O-O']


def test_engine_reuses_q_space_results_for_real_space_parameters():
    engine, params = Engine(), Parameters()
    specs = [StructureSpec(CIF, radius=6), StructureSpec(XYZ)]
    first = engine.compute(params, specs)
    params.qdamp, params.lorch_mod = 0.08, True
    second = engine.compute(params, specs)
    assert len(engine._q_results) == 2
    for a, b in zip(first, second):
        assert np.array_equal(a.i, b.i)
        assert not np.allclose(a.g, b.g)


def test_engine_partial_and_invalid_parameters():
    engine = Engine()
    partial = engine.compute(Parameters(), [StructureSpec(CIF, radius=6, partial='Co-O')])[0]
    expected = DebyeCalculator(device='cpu').iq(CIF, radii=6, partial='Co-O')[1]
    assert np.allclose(partial.i, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        engine.compute(Parameters(qmin=5.0, qmax=2.0), [StructureSpec(CIF, radius=6)])


def test_window_smoke(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    pytest.importorskip('pyqtgraph')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from debyecalculator.gui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(files=[CIF, XYZ])

    def settle():
        for _ in range(500):
            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec()
            if not (window._busy or window._pending or window._debounce.isActive()):
                break

    settle()
    assert all(item.result is not None for item in window.items)
    for mode in ('Stacked', 'Separate', 'Overlay'):
        window.mode_combo.setCurrentText(mode)
    window.qdamp_slider.spin.setValue(0.1)
    settle()

    assert len(window.export_data(str(tmp_path))) == 4
    assert len(window.export_particles(str(tmp_path))) == 2
    window.plot_panel.export_image(str(tmp_path / 'figure.png'))
    assert (tmp_path / 'figure.png').stat().st_size > 0

    session = json.loads(json.dumps(window.session()))
    window.load_session(session)
    settle()
    assert len(window.items) == 2 and window.params.qdamp == pytest.approx(0.1)
    window.close()
