"""
Main window of the desktop GUI.
"""

import copy
import json
import time
from dataclasses import asdict
from itertools import count
from pathlib import Path
from typing import List, Optional

import torch
from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from debyecalculator.gui.engine import (
    STRUCTURE_SUFFIXES, Engine, Parameters, Result, StructureSpec, available_devices,
)
from debyecalculator.gui.plots import FUNCTIONS, LEGEND_POSITIONS, MODES, PlotEntry, PlotOptions, PlotPanel
from debyecalculator.gui.theme import THEME_MODES, apply_theme
from debyecalculator.gui.widgets import ColorButton, FloatSlider


def package_version() -> str:
    try:
        from importlib.metadata import version
        return version('debyecalculator')
    except Exception:
        return ''


PALETTES = {
    'Tableau 10': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                   '#bcbd22', '#17becf'],
    'Dark 2': ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'],
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Cividis': 'cividis',
}

PRESETS = {
    'Small-angle scattering': dict(params=dict(qmin=0.0, qmax=3.0, qstep=0.01), functions=('i',), log_iq=True, log_q=True),
    'Powder diffraction': dict(params=dict(qmin=1.0, qmax=8.0, qstep=0.1), functions=('i',), log_iq=False, log_q=False),
    'Total scattering': dict(params=dict(qmin=1.0, qmax=30.0, qstep=0.05), functions=('i', 'f', 'g'), log_iq=False, log_q=False),
}


def palette_colors(name: str, n: int) -> List[str]:
    palette = PALETTES.get(name, PALETTES['Tableau 10'])
    if isinstance(palette, list):
        return [palette[k % len(palette)] for k in range(n)]
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    cmap = colormaps[palette]
    return [to_hex(cmap(0.1 + 0.8 * k / max(1, n - 1))) for k in range(n)]


def file_stem(label: str, fallback: str) -> str:
    """
    File-name-safe ASCII version of a label, e.g. 'AntiFluorite_Co2O (r = 11 Å)' -> 'AntiFluorite_Co2O_r11A'.
    """
    text = label.replace(' = ', '').replace(' Å', 'A').replace('Å', 'A')
    stem, previous = [], ''
    for char in text:
        char = char if (char.isascii() and (char.isalnum() or char in '-.')) else '_'
        if not (char == '_' and previous == '_'):
            stem.append(char)
        previous = char
    return ''.join(stem).strip('_') or fallback


class StructureItem:
    _ids = count()

    def __init__(self, spec: StructureSpec, label: str, color: QColor, visible: bool = True):
        self.id = next(self._ids)
        self.spec = spec
        self.label = label
        self.color = QColor(color)
        self.visible = visible
        self.result: Optional[Result] = None
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(spec=asdict(self.spec), label=self.label, color=self.color.name(), visible=self.visible)


class ComputeWorker(QObject):
    finished = Signal(int, object, str, float)

    def __init__(self):
        super().__init__()
        self.engine = Engine()

    @Slot(int, object, object)
    def run(self, request_id: int, params: Parameters, jobs: list) -> None:
        start = time.perf_counter()
        results, errors = {}, []
        for item_id, spec in jobs:
            try:
                results[item_id] = self.engine.compute(params, [spec])[0]
            except Exception as error:  # reported in the status bar; other structures still update
                results[item_id] = error
                errors.append(f'{Path(spec.path).name}: {error}')
        self.finished.emit(request_id, results, '; '.join(errors), time.perf_counter() - start)


class MainWindow(QMainWindow):
    request = Signal(int, object, object)

    def __init__(self, files: Optional[List[str]] = None):
        super().__init__()
        self.setWindowTitle(f'DebyeCalculator {package_version()}'.strip())
        self.resize(1500, 900)
        self.setAcceptDrops(True)

        self.settings = QSettings('DebyeCalculator', 'DebyeCalculator')
        self.items: List[StructureItem] = []
        self.params = Parameters()
        self.plot_options = PlotOptions()
        self.palette_name = 'Tableau 10'
        self.theme_mode = str(self.settings.value('theme', 'System'))
        self.dark = False
        self._request_ids = count(1)
        self._busy = False
        self._pending = False
        self._updating_controls = False

        self._build_ui()
        self._build_menu()
        self._start_worker()

        self._debounce = QTimer(self, singleShot=True, interval=15)
        self._debounce.timeout.connect(self._dispatch)

        self._apply_params_to_controls()
        self._apply_plot_options_to_controls()
        self.set_theme(self.theme_mode)
        hints = QGuiApplication.styleHints()
        if hasattr(hints, 'colorSchemeChanged'):
            hints.colorSchemeChanged.connect(lambda _scheme: self.theme_mode == 'System' and self.set_theme('System'))
        for path in files or []:
            self.add_file(path)

    # -- worker ------------------------------------------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = ComputeWorker()
        self._worker.moveToThread(self._thread)
        self.request.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def closeEvent(self, event) -> None:
        self._thread.quit()
        self._thread.wait(5000)
        super().closeEvent(event)

    def schedule(self) -> None:
        """
        Request a recomputation. Requests arriving while a calculation runs collapse into one follow-up request.
        """
        self._debounce.start()

    def _dispatch(self) -> None:
        if self._busy:
            self._pending = True
            return
        jobs = [(item.id, copy.deepcopy(item.spec)) for item in self.items if item.visible]
        if not jobs:
            self._refresh_plots()
            return
        self._busy = True
        self._pending = False
        self._set_status('Calculating…')
        self.request.emit(next(self._request_ids), copy.deepcopy(self.params), jobs)

    @Slot(int, object, str, float)
    def _on_finished(self, _request_id: int, results: dict, error: str, elapsed: float) -> None:
        self._busy = False
        by_id = {item.id: item for item in self.items}
        for item_id, result in results.items():
            item = by_id.get(item_id)
            if item is None:
                continue
            if isinstance(result, Exception):
                item.error = str(result)
            else:
                item.result, item.error = result, None
        self._refresh_table_values()
        self._refresh_partials()
        self._refresh_plots()

        atoms = sum(item.result.num_atoms for item in self.items if item.visible and item.result is not None)
        if error:
            self._set_status(error, error=True)
        else:
            self._set_status(f'{len(results)} structure(s), {atoms:,} atoms, calculated in {elapsed * 1e3:.0f} ms')
        if self._pending:
            self._dispatch()

    # -- UI construction ---------------------------------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.plot_panel = PlotPanel()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll(self._structures_tab()), 'Structures')
        self.tabs.addTab(self._scroll(self._scattering_tab()), 'Scattering')
        self.tabs.addTab(self._scroll(self._plot_tab()), 'Plot')
        self.tabs.addTab(self._scroll(self._performance_tab()), 'Performance')
        self.tabs.setMinimumWidth(460)

        splitter = QSplitter()
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.plot_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setSizes([520, 980])
        self.setCentralWidget(splitter)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    @staticmethod
    def _button_row(buttons) -> QHBoxLayout:
        row = QHBoxLayout()
        for text, slot, tooltip in buttons:
            button = QPushButton(text)
            button.clicked.connect(slot)
            if tooltip:
                button.setToolTip(tooltip)
            row.addWidget(button)
        return row

    def _structures_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addLayout(self._button_row([
            ('Add files…', self.add_files_dialog, 'Add .cif, .xyz or other structure files (Ctrl+O)'),
            ('Duplicate', self.duplicate_selected, 'Copy the selected structure, e.g. to compare radii'),
            ('Remove', self.remove_selected, 'Remove the selected structure'),
        ]))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['', '', 'Structure', 'Atoms'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        for column in (0, 1, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideMiddle)
        self.table.setMinimumHeight(190)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)

        layout.addLayout(self._button_row([
            ('Show all', lambda: self._set_all_visible(True), None),
            ('Hide all', lambda: self._set_all_visible(False), None),
            ('Remove all', self.remove_all, None),
        ]))

        hint = QLabel('Tip: drop structure files onto the window to add them.')
        hint.setEnabled(False)
        layout.addWidget(hint)

        self.detail_box = QGroupBox('Selected structure')
        form = QFormLayout(self.detail_box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.label_edit = QLineEdit()
        self.label_edit.textEdited.connect(self._on_label_edited)
        form.addRow('Label', self.label_edit)
        self.radius_slider = FloatSlider('Radius', 2.0, 30.0, 10.0, decimals=2, step=0.5, unit='Å', label_width=0,
                                         hard_maximum=500.0)
        self.radius_slider.label.hide()
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        form.addRow('Radius', self.radius_slider)
        radius_all = QPushButton('Apply radius to all CIFs')
        radius_all.clicked.connect(self.apply_radius_to_all)
        form.addRow('', radius_all)
        self.lightweight_check = QCheckBox('Lightweight cut')
        self.lightweight_check.setToolTip('Keep all atoms within the radius, without the bond analysis that '
                                          'completes coordination around the surface metal atoms')
        self.lightweight_check.toggled.connect(self._on_lightweight_toggled)
        form.addRow('', self.lightweight_check)
        self.partial_combo = QComboBox()
        self.partial_combo.currentIndexChanged.connect(self._on_partial_changed)
        form.addRow('Partial', self.partial_combo)
        self.detail_box.setEnabled(False)
        layout.addWidget(self.detail_box)

        colors = QGroupBox('Colours')
        color_layout = QHBoxLayout(colors)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES))
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        color_layout.addWidget(QLabel('Palette'))
        color_layout.addWidget(self.palette_combo, 1)
        recolor = QPushButton('Recolour all')
        recolor.clicked.connect(self.recolor_all)
        color_layout.addWidget(recolor)
        layout.addWidget(colors)
        layout.addStretch(1)
        return page

    def _scattering_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        presets = QGroupBox('Presets')
        grid = QGridLayout(presets)
        for index, name in enumerate(PRESETS):
            button = QPushButton(name)
            button.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            grid.addWidget(button, index // 2, index % 2)
        reset = QPushButton('Reset to defaults')
        reset.clicked.connect(self.reset_parameters)
        grid.addWidget(reset, 1, 1)
        layout.addWidget(presets)

        general = QGroupBox('Radiation')
        general_layout = QFormLayout(general)
        self.radiation_combo = QComboBox()
        self.radiation_combo.addItems(['X-ray', 'Neutron'])
        self.radiation_combo.currentIndexChanged.connect(self._on_params_changed)
        general_layout.addRow('Type', self.radiation_combo)
        self.self_scattering_check = QCheckBox('Include self-scattering in I(Q)')
        self.self_scattering_check.toggled.connect(self._on_params_changed)
        general_layout.addRow(self.self_scattering_check)
        layout.addWidget(general)

        q_box = QGroupBox('Q-space')
        q_layout = QVBoxLayout(q_box)
        self.qmin_slider = FloatSlider('Qmin', 0.0, 10.0, 1.0, decimals=2, step=0.1, unit='Å⁻¹', hard_maximum=200.0)
        self.qmax_slider = FloatSlider('Qmax', 0.1, 40.0, 30.0, decimals=2, step=0.5, unit='Å⁻¹', hard_maximum=200.0)
        self.qstep_auto = QCheckBox('Qstep from r-range: π / (rmax + rstep)')
        self.qstep_slider = FloatSlider('Qstep', 0.001, 0.5, 0.05, decimals=4, step=0.005, log=True, unit='Å⁻¹')
        self.biso_slider = FloatSlider('Biso', 0.0, 3.0, 0.3, decimals=3, step=0.05, unit='Å²')
        self.rthres_slider = FloatSlider('rthres', 0.0, 5.0, 0.0, decimals=2, step=0.1, unit='Å')
        self.rthres_slider.setToolTip('Exclude atom pairs closer than this distance')
        for widget in (self.qmin_slider, self.qmax_slider, self.qstep_auto, self.qstep_slider, self.biso_slider,
                       self.rthres_slider):
            q_layout.addWidget(widget)
        layout.addWidget(q_box)

        r_box = QGroupBox('Real space, G(r)')
        r_layout = QVBoxLayout(r_box)
        self.rmin_slider = FloatSlider('rmin', 0.0, 50.0, 0.0, decimals=2, step=0.5, unit='Å', hard_maximum=1000.0)
        self.rmax_slider = FloatSlider('rmax', 1.0, 50.0, 20.0, decimals=2, step=0.5, unit='Å', hard_maximum=1000.0)
        self.rstep_slider = FloatSlider('rstep', 0.001, 0.5, 0.01, decimals=4, step=0.005, log=True, unit='Å')
        self.qdamp_slider = FloatSlider('Qdamp', 0.0, 0.2, 0.04, decimals=4, step=0.005, unit='Å⁻¹')
        self.lorch_check = QCheckBox('Lorch modification')
        for widget in (self.rmin_slider, self.rmax_slider, self.rstep_slider, self.qdamp_slider, self.lorch_check):
            r_layout.addWidget(widget)
        layout.addWidget(r_box)

        self.parameter_sliders = [self.qmin_slider, self.qmax_slider, self.qstep_slider, self.biso_slider,
                                  self.rthres_slider, self.rmin_slider, self.rmax_slider, self.rstep_slider,
                                  self.qdamp_slider]
        for slider in self.parameter_sliders:
            slider.valueChanged.connect(self._on_params_changed)
        self.qstep_auto.toggled.connect(self._on_params_changed)
        self.lorch_check.toggled.connect(self._on_params_changed)
        layout.addStretch(1)
        return page

    def _plot_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        functions = QGroupBox('Functions')
        function_layout = QVBoxLayout(functions)
        checks = QHBoxLayout()
        self.function_checks = {}
        for key, (title, *_) in FUNCTIONS.items():
            check = QCheckBox(title)
            check.toggled.connect(self._on_plot_options_changed)
            checks.addWidget(check)
            self.function_checks[key] = check
        function_layout.addLayout(checks)
        function_layout.addLayout(self._button_row([
            ('All', lambda: self._set_functions(tuple(FUNCTIONS)), None),
            ('None', lambda: self._set_functions(()), None),
            ('Q-space only', lambda: self._set_functions(('i', 's', 'f')), None),
            ('I(Q) + G(r)', lambda: self._set_functions(('i', 'g')), None),
        ]))
        layout.addWidget(functions)

        arrangement = QGroupBox('Co-plotting')
        form = QFormLayout(arrangement)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        self.mode_combo.setToolTip('Overlay: all structures on the same axes\n'
                                   'Stacked: curves shifted by an offset\n'
                                   'Separate: one row of plots per structure')
        self.mode_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        form.addRow('Mode', self.mode_combo)
        self.offset_slider = FloatSlider('', 0.0, 3.0, 1.0, decimals=2, step=0.05, label_width=0)
        self.offset_slider.label.hide()
        self.offset_slider.setToolTip('Spacing between stacked curves, in units of the largest curve')
        self.offset_slider.valueChanged.connect(self._on_plot_options_changed)
        form.addRow('Stack offset', self.offset_slider)
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 4)
        self.columns_spin.valueChanged.connect(self._on_plot_options_changed)
        form.addRow('Plots per row', self.columns_spin)
        layout.addWidget(arrangement)

        appearance = QGroupBox('Appearance')
        appearance_layout = QGridLayout(appearance)
        self.normalize_check = QCheckBox('Normalise curves')
        self.normalize_check.setToolTip('Scale each curve to a maximum absolute value of 1')
        self.log_iq_check = QCheckBox('Log I(Q) axis')
        self.log_q_check = QCheckBox('Log Q axis (I(Q))')
        self.grid_check = QCheckBox('Grid')
        self.markers_check = QCheckBox('Markers')
        self.auto_range_check = QCheckBox('Auto-range')
        self.auto_range_check.setToolTip('Rescale the axes to the data on every update.\n'
                                         'Off: keep the current zoom while parameters change.')
        self.legend_check = QCheckBox('Legend')
        checks = [self.normalize_check, self.log_iq_check, self.log_q_check, self.grid_check, self.markers_check,
                  self.auto_range_check, self.legend_check]
        for index, check in enumerate(checks):
            check.toggled.connect(self._on_plot_options_changed)
            appearance_layout.addWidget(check, index // 2, index % 2)
        self.legend_combo = QComboBox()
        self.legend_combo.addItems(list(LEGEND_POSITIONS))
        self.legend_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        appearance_layout.addWidget(self.legend_combo, len(checks) // 2, 1)
        self.line_width_slider = FloatSlider('Line width', 0.5, 5.0, 1.5, decimals=1, step=0.5, label_width=70)
        self.line_width_slider.valueChanged.connect(self._on_plot_options_changed)
        appearance_layout.addWidget(self.line_width_slider, len(checks) // 2 + 1, 0, 1, 2)
        reset_view = QPushButton('Reset view (Ctrl+R)')
        reset_view.clicked.connect(self.plot_panel.reset_view)
        appearance_layout.addWidget(reset_view, len(checks) // 2 + 2, 0, 1, 2)
        layout.addWidget(appearance)

        export = QGroupBox('Figure export')
        export_form = QFormLayout(export)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.valueChanged.connect(self._on_plot_options_changed)
        export_form.addRow('Resolution [dpi]', self.dpi_spin)
        size_row = QHBoxLayout()
        self.panel_width_spin = QDoubleSpinBox()
        self.panel_height_spin = QDoubleSpinBox()
        for spin in (self.panel_width_spin, self.panel_height_spin):
            spin.setRange(1.0, 20.0)
            spin.setSingleStep(0.1)
            spin.setSuffix(' in')
            spin.valueChanged.connect(self._on_plot_options_changed)
            size_row.addWidget(spin)
        export_form.addRow('Size per plot (w × h)', size_row)
        export_button = QPushButton('Export figure…')
        export_button.clicked.connect(self.export_figure_dialog)
        export_form.addRow(export_button)
        layout.addWidget(export)
        layout.addStretch(1)
        return page

    def _performance_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        hardware = QGroupBox('Hardware')
        form = QFormLayout(hardware)
        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices())
        self.device_combo.currentIndexChanged.connect(self._on_params_changed)
        form.addRow('Device', self.device_combo)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(['float32', 'float64'])
        self.dtype_combo.currentIndexChanged.connect(self._on_params_changed)
        form.addRow('Precision', self.dtype_combo)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 256)
        self.threads_spin.setSpecialValueText(f'Default ({torch.get_num_threads()})')
        self.threads_spin.valueChanged.connect(self._on_params_changed)
        form.addRow('CPU threads', self.threads_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(10_000, 2_000_000_000)
        self.batch_spin.setSingleStep(1_000_000)
        self.batch_spin.setGroupSeparatorShown(True)
        self.batch_spin.setKeyboardTracking(False)
        self.batch_spin.valueChanged.connect(self._on_params_changed)
        form.addRow('Batch size (atom pairs)', self.batch_spin)
        layout.addWidget(hardware)

        updates = QGroupBox('Updates')
        updates_layout = QVBoxLayout(updates)
        self.live_check = QCheckBox('Update while dragging sliders')
        self.live_check.setToolTip('Off: calculate when a slider is released, useful for large particles')
        self.live_check.setChecked(True)
        self.live_check.toggled.connect(self._on_live_toggled)
        updates_layout.addWidget(self.live_check)
        layout.addWidget(updates)
        layout.addStretch(1)
        return page

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu('&File')
        entries = [
            ('Add structure files…', QKeySequence.Open, self.add_files_dialog),
            (None, None, None),
            ('Open session…', None, self.load_session_dialog),
            ('Save session…', QKeySequence.Save, self.save_session_dialog),
            (None, None, None),
            ('Export data (CSV)…', QKeySequence('Ctrl+E'), self.export_data_dialog),
            ('Export figure (PNG/SVG/PDF)…', QKeySequence('Ctrl+Shift+E'), self.export_figure_dialog),
            ('Export particles (XYZ)…', None, self.export_particles_dialog),
            (None, None, None),
            ('Quit', QKeySequence.Quit, self.close),
        ]
        for text, shortcut, slot in entries:
            if text is None:
                file_menu.addSeparator()
                continue
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)

        view_menu = self.menuBar().addMenu('&View')
        reset = QAction('Reset view', self)
        reset.setShortcut(QKeySequence('Ctrl+R'))
        reset.triggered.connect(self.plot_panel.reset_view)
        view_menu.addAction(reset)
        theme_menu = view_menu.addMenu('Theme')
        group = QActionGroup(self)
        self.theme_actions = {}
        for mode in THEME_MODES:
            action = QAction(mode, self, checkable=True)
            action.triggered.connect(lambda _=False, m=mode: self.set_theme(m))
            group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[mode] = action

    # -- theme -------------------------------------------------------------------------------------------------------

    def set_theme(self, mode: str) -> None:
        self.theme_mode = mode if mode in THEME_MODES else 'System'
        self.dark = apply_theme(self.theme_mode)
        self.settings.setValue('theme', self.theme_mode)
        self.theme_actions[self.theme_mode].setChecked(True)
        self.plot_panel.set_dark(self.dark)
        self._refresh_table_values()
        self._set_status(self.status_label.text(), error=self.status_label.property('error') is True)

    # -- structures --------------------------------------------------------------------------------------------------

    def add_files_dialog(self) -> None:
        patterns = ' '.join(f'*{suffix}' for suffix in STRUCTURE_SUFFIXES)
        paths, _ = QFileDialog.getOpenFileNames(self, 'Add structure files', '', f'Structures ({patterns});;All files (*)')
        for path in paths:
            self.add_file(path)

    def add_file(self, path: str, radius: float = 10.0, label: Optional[str] = None, color: Optional[str] = None,
                 visible: bool = True, lightweight: bool = False, partial: Optional[str] = None) -> StructureItem:
        spec = StructureSpec(path=str(Path(path).resolve()), radius=radius, lightweight=lightweight, partial=partial)
        if label is None:
            label = Path(path).stem + (f' (r = {radius:g} Å)' if spec.is_cif else '')
        if color is None:
            color = palette_colors(self.palette_name, len(self.items) + 1)[-1]
        item = StructureItem(spec, label, QColor(color), visible)
        self.items.append(item)
        self._rebuild_table()
        self.table.selectRow(len(self.items) - 1)
        self.schedule()
        return item

    def duplicate_selected(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        spec = item.spec
        self.add_file(spec.path, radius=spec.radius, label=f'{item.label} (copy)', lightweight=spec.lightweight,
                      partial=spec.partial)

    def remove_selected(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        self.items.remove(item)
        self._rebuild_table()
        self._refresh_plots()
        self.schedule()

    def remove_all(self) -> None:
        self.items = []
        self._rebuild_table()
        self._refresh_plots()
        self._set_status('')

    def _set_all_visible(self, visible: bool) -> None:
        for item in self.items:
            item.visible = visible
        self._rebuild_table()
        self._refresh_plots()
        self.schedule()

    def apply_radius_to_all(self) -> None:
        radius = self.radius_slider.value()
        for item in self.items:
            if item.spec.is_cif:
                self._set_item_radius(item, radius)
        self._refresh_table_values()
        self.schedule()

    def recolor_all(self) -> None:
        colors = palette_colors(self.palette_name, len(self.items))
        for item, color in zip(self.items, colors):
            item.color = QColor(color)
        self._rebuild_table()
        self._refresh_plots()

    def _on_palette_changed(self, name: str) -> None:
        self.palette_name = name
        self.recolor_all()

    def _selected_item(self) -> Optional[StructureItem]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.items[rows[0].row()] if rows and rows[0].row() < len(self.items) else None

    def _rebuild_table(self) -> None:
        selected = self._selected_item()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.items))
        for row, item in enumerate(self.items):
            visible = QCheckBox()
            visible.setChecked(item.visible)
            visible.setToolTip('Show and calculate')
            visible.toggled.connect(lambda checked, it=item: self._on_visible_toggled(it, checked))
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(6, 0, 0, 0)
            cell_layout.addWidget(visible)
            self.table.setCellWidget(row, 0, cell)

            color = ColorButton(item.color)
            color.colorChanged.connect(lambda c, it=item: self._on_color_changed(it, c))
            self.table.setCellWidget(row, 1, color)
            self.table.setItem(row, 2, QTableWidgetItem())
            atoms = QTableWidgetItem()
            atoms.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 3, atoms)
        if selected in self.items:
            self.table.selectRow(self.items.index(selected))
        self.table.blockSignals(False)
        self._refresh_table_values()
        self._on_selection_changed()

    def _refresh_table_values(self) -> None:
        if self.table.rowCount() != len(self.items):
            return
        error_color = QColor('#ff6b68' if self.dark else '#c62828')
        for row, item in enumerate(self.items):
            name = self.table.item(row, 2)
            atoms = self.table.item(row, 3)
            if name is None or atoms is None:
                continue
            name.setText(item.label)
            name.setToolTip(item.spec.path + (f'\n{item.error}' if item.error else ''))
            name.setData(Qt.ForegroundRole, error_color if item.error else None)
            atoms.setText(f'{item.result.num_atoms:,}' if item.result is not None else '–')

    def _on_selection_changed(self) -> None:
        item = self._selected_item()
        self.detail_box.setEnabled(item is not None)
        if item is None:
            return
        self._updating_controls = True
        self.label_edit.setText(item.label)
        self.radius_slider.setEnabled(item.spec.is_cif)
        self.lightweight_check.setEnabled(item.spec.is_cif)
        self.radius_slider.setValue(item.spec.radius)
        self.lightweight_check.setChecked(item.spec.lightweight)
        self._fill_partials(item)
        self._updating_controls = False

    def _fill_partials(self, item: StructureItem) -> None:
        self.partial_combo.blockSignals(True)
        self.partial_combo.clear()
        self.partial_combo.addItem('All pairs', None)
        for pair in (item.result.element_pairs if item.result is not None else []):
            self.partial_combo.addItem(pair, pair)
        index = self.partial_combo.findData(item.spec.partial)
        self.partial_combo.setCurrentIndex(max(index, 0))
        self.partial_combo.blockSignals(False)

    def _refresh_partials(self) -> None:
        item = self._selected_item()
        if item is not None:
            self._fill_partials(item)

    def _on_visible_toggled(self, item: StructureItem, checked: bool) -> None:
        item.visible = checked
        self._refresh_plots()
        self.schedule()

    def _on_color_changed(self, item: StructureItem, color: QColor) -> None:
        item.color = color
        self._refresh_plots()

    def _on_label_edited(self, text: str) -> None:
        item = self._selected_item()
        if item is not None:
            item.label = text
            self._refresh_table_values()
            self._refresh_plots()

    def _set_item_radius(self, item: StructureItem, radius: float) -> None:
        old_suffix = f' (r = {item.spec.radius:g} Å)'
        item.spec.radius = radius
        if item.label.endswith(old_suffix):
            item.label = item.label[: -len(old_suffix)] + f' (r = {radius:g} Å)'

    def _on_radius_changed(self, radius: float) -> None:
        item = self._selected_item()
        if item is None or self._updating_controls or not item.spec.is_cif:
            return
        self._set_item_radius(item, radius)
        self.label_edit.setText(item.label)
        self._refresh_table_values()
        self.schedule()

    def _on_lightweight_toggled(self, checked: bool) -> None:
        item = self._selected_item()
        if item is not None and not self._updating_controls:
            item.spec.lightweight = checked
            self.schedule()

    def _on_partial_changed(self, _index: int) -> None:
        item = self._selected_item()
        if item is not None and not self._updating_controls:
            item.spec.partial = self.partial_combo.currentData()
            self.schedule()

    # -- parameters --------------------------------------------------------------------------------------------------

    def _apply_params_to_controls(self) -> None:
        p = self.params
        self._updating_controls = True
        self.radiation_combo.setCurrentIndex(0 if p.radiation_type == 'xray' else 1)
        self.self_scattering_check.setChecked(p.include_self_scattering)
        self.qmin_slider.setValue(p.qmin)
        self.qmax_slider.setValue(p.qmax)
        self.qstep_auto.setChecked(p.qstep is None)
        self.qstep_slider.setValue(p.effective_qstep())
        self.qstep_slider.setEnabled(p.qstep is not None)
        self.biso_slider.setValue(p.biso)
        self.rthres_slider.setValue(p.rthres)
        self.rmin_slider.setValue(p.rmin)
        self.rmax_slider.setValue(p.rmax)
        self.rstep_slider.setValue(p.rstep)
        self.qdamp_slider.setValue(p.qdamp)
        self.lorch_check.setChecked(p.lorch_mod)
        self.device_combo.setCurrentIndex(max(self.device_combo.findText(p.device), 0))
        self.dtype_combo.setCurrentIndex(max(self.dtype_combo.findText(p.dtype), 0))
        self.threads_spin.setValue(p.num_threads)
        self.batch_spin.setValue(p.batch_size)
        self._updating_controls = False

    def _on_params_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        p = self.params
        p.radiation_type = 'xray' if self.radiation_combo.currentIndex() == 0 else 'neutron'
        p.include_self_scattering = self.self_scattering_check.isChecked()
        p.qmin, p.qmax = self.qmin_slider.value(), self.qmax_slider.value()
        p.rmin, p.rmax, p.rstep = self.rmin_slider.value(), self.rmax_slider.value(), self.rstep_slider.value()
        p.qstep = None if self.qstep_auto.isChecked() else self.qstep_slider.value()
        self.qstep_slider.setEnabled(not self.qstep_auto.isChecked())
        if p.qstep is None:
            self._updating_controls = True
            self.qstep_slider.setValue(p.effective_qstep())
            self._updating_controls = False
        p.biso, p.rthres, p.qdamp = self.biso_slider.value(), self.rthres_slider.value(), self.qdamp_slider.value()
        p.lorch_mod = self.lorch_check.isChecked()
        p.device = self.device_combo.currentText()
        p.dtype = self.dtype_combo.currentText()
        p.num_threads = self.threads_spin.value()
        p.batch_size = self.batch_spin.value()
        self.schedule()

    def _on_live_toggled(self, live: bool) -> None:
        for slider in self.parameter_sliders + [self.radius_slider]:
            slider.setTracking(live)

    def apply_preset(self, name: str) -> None:
        preset = PRESETS[name]
        for key, value in preset['params'].items():
            setattr(self.params, key, value)
        self.plot_options.functions = preset['functions']
        self.plot_options.log_iq = preset['log_iq']
        self.plot_options.log_q = preset['log_q']
        self._apply_params_to_controls()
        self._apply_plot_options_to_controls()
        self.plot_panel.set_options(self.plot_options)
        self.plot_panel.reset_view()
        self.schedule()

    def reset_parameters(self) -> None:
        p = self.params
        self.params = Parameters(device=p.device, dtype=p.dtype, batch_size=p.batch_size, num_threads=p.num_threads)
        self.plot_options = PlotOptions()
        self._apply_params_to_controls()
        self._apply_plot_options_to_controls()
        self.plot_panel.set_options(self.plot_options)
        self.plot_panel.reset_view()
        self.schedule()

    # -- plotting ----------------------------------------------------------------------------------------------------

    def _set_functions(self, functions: tuple) -> None:
        self._updating_controls = True
        for key, check in self.function_checks.items():
            check.setChecked(key in functions)
        self._updating_controls = False
        self._on_plot_options_changed()

    def _apply_plot_options_to_controls(self) -> None:
        o = self.plot_options
        self._updating_controls = True
        for key, check in self.function_checks.items():
            check.setChecked(key in o.functions)
        self.mode_combo.setCurrentIndex(MODES.index(o.mode))
        self.offset_slider.setValue(o.offset)
        self.offset_slider.setEnabled(o.mode == 'Stacked')
        self.columns_spin.setValue(o.columns)
        self.columns_spin.setEnabled(o.mode != 'Separate')
        self.normalize_check.setChecked(o.normalize)
        self.log_iq_check.setChecked(o.log_iq)
        self.log_q_check.setChecked(o.log_q)
        self.grid_check.setChecked(o.grid)
        self.markers_check.setChecked(o.markers)
        self.auto_range_check.setChecked(o.auto_range)
        self.legend_check.setChecked(o.legend)
        self.legend_combo.setCurrentText(o.legend_position)
        self.legend_combo.setEnabled(o.legend)
        self.line_width_slider.setValue(o.line_width)
        self.dpi_spin.setValue(o.export_dpi)
        self.panel_width_spin.setValue(o.export_panel_width)
        self.panel_height_spin.setValue(o.export_panel_height)
        self._updating_controls = False

    def _on_plot_options_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        o = self.plot_options
        o.functions = tuple(key for key, check in self.function_checks.items() if check.isChecked())
        o.mode = self.mode_combo.currentText()
        o.offset = self.offset_slider.value()
        o.columns = self.columns_spin.value()
        o.normalize = self.normalize_check.isChecked()
        o.log_iq = self.log_iq_check.isChecked()
        o.log_q = self.log_q_check.isChecked()
        o.grid = self.grid_check.isChecked()
        o.markers = self.markers_check.isChecked()
        o.auto_range = self.auto_range_check.isChecked()
        o.legend = self.legend_check.isChecked()
        o.legend_position = self.legend_combo.currentText()
        o.line_width = self.line_width_slider.value()
        o.export_dpi = self.dpi_spin.value()
        o.export_panel_width = self.panel_width_spin.value()
        o.export_panel_height = self.panel_height_spin.value()
        self.offset_slider.setEnabled(o.mode == 'Stacked')
        self.columns_spin.setEnabled(o.mode != 'Separate')
        self.legend_combo.setEnabled(o.legend)
        self.plot_panel.set_options(o)

    def _refresh_plots(self) -> None:
        entries = [PlotEntry(item.label, item.color, item.result) for item in self.items
                   if item.visible and item.result is not None]
        self.plot_panel.set_entries(entries)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty('error', error)
        color = ('#ff6b68' if self.dark else '#c62828') if error else ''
        self.status_label.setStyleSheet(f'color: {color};' if color else '')

    # -- drag and drop -----------------------------------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).is_file():
                self.add_file(path)

    # -- export ------------------------------------------------------------------------------------------------------

    def _metadata_lines(self, item: StructureItem) -> List[str]:
        lines = [f'# DebyeCalculator {package_version()}'.rstrip(), f'# structure: {item.spec.path}']
        if item.spec.is_cif:
            lines.append(f'# radius [Å]: {item.spec.radius:g}')
        if item.result is not None:
            lines.append(f'# atoms: {item.result.num_atoms}')
        if item.spec.partial:
            lines.append(f'# partial: {item.spec.partial}')
        for key, value in self.params.to_dict().items():
            lines.append(f'# {key}: {value}')
        lines.append(f'# qstep (effective): {self.params.effective_qstep():.6g}')
        return lines

    def export_data_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Export data to folder')
        if directory:
            written = self.export_data(directory)
            self._set_status(f'Wrote {len(written)} files to {directory}')

    def export_data(self, directory: str) -> List[Path]:
        """
        Write <label>_Q.csv (Q, I, S, F) and <label>_r.csv (r, G) for every shown structure.
        """
        import numpy as np
        written = []
        for item in self.items:
            if not item.visible or item.result is None:
                continue
            stem = file_stem(item.label, f'structure_{item.id}')
            header = '\n'.join(self._metadata_lines(item))
            result = item.result
            q_path = Path(directory) / f'{stem}_Q.csv'
            np.savetxt(q_path, np.column_stack([result.q, result.i, result.s, result.f]), delimiter=',',
                       header=header + '\nQ [1/Å],I(Q),S(Q),F(Q)', comments='')
            r_path = Path(directory) / f'{stem}_r.csv'
            np.savetxt(r_path, np.column_stack([result.r, result.g]), delimiter=',',
                       header=header + '\nr [Å],G(r)', comments='')
            written += [q_path, r_path]
        return written

    def export_figure_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Export figure', 'debyecalculator.png',
                                              'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)')
        if path:
            try:
                self.plot_panel.export_image(path)
                self._set_status(f'Wrote {path}')
            except (OSError, ValueError) as error:
                self._set_status(f'Figure export failed: {error}', error=True)

    def export_particles_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Export particles to folder')
        if directory:
            written = self.export_particles(directory)
            self._set_status(f'Wrote {len(written)} files to {directory}')

    def export_particles(self, directory: str) -> List[Path]:
        written = []
        for item in self.items:
            if not item.visible:
                continue
            particle = self._worker.engine._particles.get_item(item.spec.particle_key())
            if particle is None:
                continue
            path = Path(directory) / f'{file_stem(item.label, f"structure_{item.id}")}.xyz'
            with open(path, 'w') as f:
                f.write(f'{particle.size}\n{item.label}\n')
                for element, (x, y, z) in zip(particle.elements, particle.xyz):
                    f.write(f'{element} {x:.6f} {y:.6f} {z:.6f}\n')
            written.append(path)
        return written

    # -- sessions ----------------------------------------------------------------------------------------------------

    def session(self) -> dict:
        return dict(parameters=self.params.to_dict(), plot=asdict(self.plot_options), palette=self.palette_name,
                    structures=[item.to_dict() for item in self.items])

    def save_session_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save session', 'session.json', 'Session (*.json)')
        if path:
            Path(path).write_text(json.dumps(self.session(), indent=1))
            self._set_status(f'Saved session to {path}')

    def load_session_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open session', '', 'Session (*.json)')
        if path:
            try:
                self.load_session(json.loads(Path(path).read_text()))
            except (OSError, ValueError, KeyError, TypeError) as error:
                QMessageBox.warning(self, 'Open session', f'Could not open {path}:\n{error}')

    def load_session(self, session: dict) -> None:
        params = Parameters()
        for key, value in session.get('parameters', {}).items():
            if hasattr(params, key):
                setattr(params, key, value)
        if params.device not in available_devices():
            params.device = 'cpu'
        self.params = params
        options = PlotOptions()
        for key, value in session.get('plot', {}).items():
            if hasattr(options, key):
                setattr(options, key, tuple(value) if key == 'functions' else value)
        self.plot_options = options
        self.palette_name = session.get('palette', self.palette_name)
        self.palette_combo.blockSignals(True)
        self.palette_combo.setCurrentText(self.palette_name)
        self.palette_combo.blockSignals(False)
        self.items = []
        self._apply_params_to_controls()
        self._apply_plot_options_to_controls()
        self.plot_panel.set_options(self.plot_options)
        for entry in session.get('structures', []):
            spec = entry['spec']
            self.add_file(spec['path'], radius=spec.get('radius', 10.0), label=entry.get('label'),
                          color=entry.get('color'), visible=entry.get('visible', True),
                          lightweight=spec.get('lightweight', False), partial=spec.get('partial'))
        self._refresh_plots()
        self.schedule()
