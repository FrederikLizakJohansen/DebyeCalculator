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

    from PySide6.QtCore import QSettings

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.IniFormat)
    window = MainWindow(files=[CIF, XYZ], settings=settings)

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


def test_engine_partial_breakdown_and_cancel():
    from debyecalculator.debye_calculator import CalculationCancelled

    engine = Engine()
    result = engine.compute(Parameters(), [StructureSpec(CIF, radius=8, show_partials=True)])[0]
    assert sorted(result.partials) == ['Co-Co', 'Co-O', 'O-O']
    for name in 'isfg':
        total = sum(values[name] for values in result.partials.values())
        assert np.allclose(total, getattr(result, name), rtol=1e-4, atol=1e-5 * np.abs(getattr(result, name)).max())

    with pytest.raises(CalculationCancelled):
        Engine().compute(Parameters(), [StructureSpec(CIF, radius=20)], progress=lambda fraction: fraction < 0.2)


def test_data_loading_and_comparison(tmp_path):
    from debyecalculator.gui.data import compare, load_xy, q_to_two_theta, two_theta_to_q

    x = np.linspace(1, 20, 200)
    y = np.sin(x)
    path = tmp_path / 'data.gr'
    path.write_text('# header\nsome metadata line\n' + '\n'.join(f'{a:.6f}, {b:.6f}' for a, b in zip(x, 3 * y)))
    loaded_x, loaded_y = load_xy(str(path))
    assert np.allclose(loaded_x, x, atol=1e-6) and np.allclose(loaded_y, 3 * y, atol=1e-6)

    comparison = compare(loaded_x, loaded_y, x, y, fit_scale=True)
    assert comparison.scale == pytest.approx(3.0, rel=1e-5)
    assert comparison.rw < 1e-5
    assert compare(loaded_x, loaded_y, x, y, fit_scale=False).scale == 1.0

    q = np.linspace(0.5, 7, 50)
    assert np.allclose(two_theta_to_q(q_to_two_theta(q, 1.5406), 1.5406), q)


def test_window_data_overlay_and_session(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    pytest.importorskip('pyqtgraph')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import QEventLoop, QSettings, QTimer
    from PySide6.QtWidgets import QApplication
    from debyecalculator.gui.window import MainWindow

    r, g = DebyeCalculator(device='cpu').gr(CIF, radii=6)
    data_path = tmp_path / 'measured.gr'
    np.savetxt(data_path, np.column_stack([r, 2.0 * g]))

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.IniFormat)
    window = MainWindow(settings=settings, restore_session=False)
    item = window.add_file(CIF, radius=6)

    def settle():
        for _ in range(500):
            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec()
            if not (window._busy or window._pending or window._debounce.isActive()):
                break

    settle()
    data = window.add_data(str(data_path))
    window._refresh_plots()
    assert data.compare_id == item.id
    assert data.comparison.scale == pytest.approx(2.0, rel=1e-4)
    assert data.comparison.rw < 1e-4

    window.table.selectRow(0)
    window.show_partials_check.setChecked(True)
    settle()
    assert len(item.result.partials) == 3
    assert len(window.export_data(str(tmp_path))) == 8  # total + 3 partials, Q and r files each

    window.close()  # saves the session to the settings
    restored = MainWindow(settings=settings)
    assert [d.label for d in restored.data_items] == ['measured.gr']
    assert restored.items[0].spec.show_partials
    restored.close()


def test_window_layout_stays_bounded_with_many_files(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    pytest.importorskip('pyqtgraph')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from debyecalculator.gui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.IniFormat)
    window = MainWindow(settings=settings, restore_session=False)
    window.schedule = lambda: None
    window.resize(1100, 900)
    window.show()
    app.processEvents()

    assert window.table.height() > window.table.sizeHint().height()
    window.tabs.setCurrentIndex(1)
    app.processEvents()
    assert window.data_table.height() > window.data_table.sizeHint().height()

    for index in range(20):
        window.add_file(CIF, label=f'A very long structure label {index}')
    minimum_width = window.minimumSizeHint().width()
    long_text = 'I(Q)  Q = 1.234   ' + '   '.join(f'{item.label}: 12345.6789' for item in window.items)
    window.cursor_label.setText(long_text)
    window._set_status('/a/very/long/path/' * 100)
    window._refresh_compare_combo()
    app.processEvents()

    assert window.minimumSizeHint().width() == minimum_width
    assert window.cursor_label.fullText() == long_text
    assert window.cursor_label.toolTip() == long_text
    assert window.cursor_label.text().endswith('…')
    window.close()
