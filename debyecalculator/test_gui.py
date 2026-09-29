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


def test_engine_unit_cell_views():
    engine = Engine()
    views = {'input': engine.unit_cell(CIF, 'input')}
    try:
        import pymatgen  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match='requires pymatgen'):
            engine.unit_cell(CIF, 'primitive')
    else:
        views.update({mode: engine.unit_cell(CIF, mode) for mode in
                      ('primitive', 'conventional', 'reduced_niggli', 'reduced_lll')})
    for view in views.values():
        assert view.lattice.shape == (3, 3)
        assert abs(np.linalg.det(view.lattice)) > 0
        assert len(view.elements) >= view.atom_count
        assert view.xyz.shape == (len(view.elements), 3)
    if 'primitive' in views:
        assert views['primitive'].atom_count <= views['conventional'].atom_count


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
    window.partials_only_check.setChecked(True)
    from debyecalculator.gui.plots import build_curves
    entries, data_entries = window._plot_entries()
    curves = build_curves(entries, data_entries, window.plot_options, 'g', [0], '#202020')
    calculated_labels = [curve.label for curve in curves if curve.label.startswith(item.label)]
    assert len(calculated_labels) == 3
    assert all(' · ' in label for label in calculated_labels)
    item.spec.partial = 'Co-O'
    pair_entry = window._plot_entries()[0][0]
    assert pair_entry.show_total and not pair_entry.show_partials
    item.spec.partial = None
    assert len(window.export_data(str(tmp_path))) == 8  # total + 3 partials, Q and r files each

    window.close()  # saves the session to the settings
    restored = MainWindow(settings=settings)
    assert [d.label for d in restored.data_items] == ['measured.gr']
    assert restored.items[0].spec.show_partials
    assert restored.items[0].spec.partials_only
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
    schedule_calls = []
    window.schedule = lambda: schedule_calls.append(True)
    window.resize(1100, 900)
    window.show()
    app.processEvents()

    assert window.table.height() > window.table.sizeHint().height()
    window.tabs.setCurrentIndex(1)
    app.processEvents()
    assert window.data_table.height() > window.data_table.sizeHint().height()

    window.add_files([CIF] * 20)
    assert len(schedule_calls) == 1
    for index, item in enumerate(window.items):
        item.label = f'A very long structure label {index}'
    window._refresh_table_values()
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


def test_hidden_particle_and_unit_cell_view(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    pytest.importorskip('pyqtgraph')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import QEventLoop, QSettings, QTimer
    from PySide6.QtWidgets import QApplication, QCheckBox
    from debyecalculator.gui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.IniFormat)
    window = MainWindow(settings=settings, restore_session=False)
    window.schedule = lambda: None
    item = window.add_file(CIF, radius=6, visible=False)
    window.show()
    app.processEvents()

    def wait_for_view():
        for _ in range(500):
            loop = QEventLoop()
            QTimer.singleShot(10, loop.quit)
            loop.exec()
            if window._pending_view_key is None and window._displayed_view is not None:
                return
        pytest.fail('3D structure view did not finish loading')

    window.view_button.click()
    wait_for_view()
    assert window.view_button.text() == 'Hide particle in 3D'
    assert item.result is None and not item.visible
    assert len(window.particle_view._elements) > 0

    window.particle_mode_combo.setCurrentIndex(window.particle_mode_combo.findData('input'))
    wait_for_view()
    assert len(window.particle_view._cell_edges) == 12
    visibility_check = window.table.cellWidget(0, 0).findChild(QCheckBox)
    visibility_check.setChecked(True)
    visibility_check.setChecked(False)
    app.processEvents()
    assert window._selected_item() is item and not item.visible
    assert len(window.particle_view._cell_edges) == 12
    assert len(window.particle_view.cell_back.getData()[0]) > 0
    assert len(window.particle_view.cell_front.getData()[0]) > 0
    assert window.particle_view.cell_back.zValue() < window.particle_view.scatter.zValue()
    assert window.particle_view.cell_front.zValue() > window.particle_view.scatter.zValue()
    window.particle_view.center_view()

    from debyecalculator.gui import particle_view as particle_view_module
    monkeypatch.setattr(particle_view_module, 'LARGE_PARTICLE_THRESHOLD', 10)
    monkeypatch.setattr(particle_view_module, 'RENDER_ATOM_LIMIT', 6)
    xyz = np.column_stack((np.arange(12), np.zeros(12), np.zeros(12)))
    window.particle_view.set_particle(['C'] * 12, xyz, 'Large particle')
    assert len(window.particle_view._elements) == 6
    assert window.particle_view._atom_count == 12
    assert np.allclose(window.particle_view._xyz[:, 0], [-5.5, -4.5, -3.5, 3.5, 4.5, 5.5])
    assert 'particles above 10 atoms show an outer shell (6 atoms displayed)' in window.particle_view.info.text()

    window.view_button.click()
    assert window.particle_dock.isHidden()
    assert window.view_button.text() == 'Show particle in 3D'
    window.close()
