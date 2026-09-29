"""
Main window of the desktop GUI.
"""

import copy
import json
import threading
import time
from dataclasses import asdict
from itertools import count
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from debyecalculator.debye_calculator import CalculationCancelled
from debyecalculator.gui.data import DATA_SUFFIXES, compare, guess_function, load_xy, q_to_two_theta, two_theta_to_q
from debyecalculator.gui.engine import (
    STRUCTURE_SUFFIXES, Engine, Parameters, Result, StructureSpec, available_devices,
)
from debyecalculator.gui.particle_view import ParticleView
from debyecalculator.gui.plots import (
    FUNCTIONS, LEGEND_POSITIONS, MODES, DataEntry, PlotEntry, PlotOptions, PlotPanel,
)
from debyecalculator.gui.theme import THEME_MODES, apply_theme
from debyecalculator.gui.widgets import ColorButton, ElidedLabel, FloatSlider


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

# X-ray tube and common wavelengths [Å]
WAVELENGTHS = {'Cu Kα (1.5406 Å)': 1.5406, 'Co Kα (1.7890 Å)': 1.7890, 'Mo Kα (0.7107 Å)': 0.7107,
               'Ag Kα (0.5594 Å)': 0.5594, 'Custom': None}

MAX_RECENT = 10
SLOW_CALCULATION_MS = 250  # show progress and the cancel button after this time
STRUCTURE_VIEW_MODES = (
    ('Particle', 'particle'),
    ('Input cell', 'input'),
    ('Primitive cell', 'primitive'),
    ('Conventional cell', 'conventional'),
    ('Reduced cell (Niggli)', 'reduced_niggli'),
    ('Reduced cell (LLL)', 'reduced_lll'),
)


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


class DataItem:
    """
    Experimental data. x is stored as Q (or r for G(r)); data measured against 2θ is converted on loading.
    """
    _ids = count()

    def __init__(self, path: str, function: str, x_unit: str = 'Q', wavelength: float = 0.7107,
                 label: Optional[str] = None, visible: bool = True, color: Optional[str] = None,
                 compare_id: Optional[int] = None, fit_scale: bool = True, show_difference: bool = True):
        self.id = next(self._ids)
        self.path = str(Path(path).resolve())
        self.function = function
        self.x_unit = x_unit
        self.wavelength = wavelength
        self.label = label or Path(path).name
        self.visible = visible
        self.color = QColor(color) if color else None
        self.compare_id = compare_id
        self.fit_scale = fit_scale
        self.show_difference = show_difference
        self.raw_x, self.y = load_xy(self.path)
        self.comparison = None

    @property
    def x(self) -> np.ndarray:
        if self.function == 'i' and self.x_unit == '2θ':
            return two_theta_to_q(self.raw_x, self.wavelength)
        return self.raw_x

    def to_dict(self) -> dict:
        return dict(path=self.path, function=self.function, x_unit=self.x_unit, wavelength=self.wavelength,
                    label=self.label, visible=self.visible, color=self.color.name() if self.color else None,
                    compare_id=self.compare_id, fit_scale=self.fit_scale, show_difference=self.show_difference)


class ComputeWorker(QObject):
    finished = Signal(int, object, str, float, bool)
    progress = Signal(int, float)
    viewReady = Signal(object, object, str)

    def __init__(self):
        super().__init__()
        self.engine = Engine()

    @Slot(int, object, object, object)
    def run(self, request_id: int, params: Parameters, jobs: list, cancel: threading.Event) -> None:
        start = time.perf_counter()
        results, errors = {}, []
        last_reported = [-1.0]

        for number, (item_id, spec) in enumerate(jobs):
            def report(fraction, number=number):
                overall = (number + fraction) / len(jobs)
                if overall - last_reported[0] >= 0.01:
                    last_reported[0] = overall
                    self.progress.emit(request_id, overall)
                return not cancel.is_set()

            try:
                results[item_id] = self.engine.compute(params, [spec], progress=report)[0]
            except CalculationCancelled:
                self.finished.emit(request_id, {}, '', time.perf_counter() - start, True)
                return
            except Exception as error:  # reported in the status bar; other structures still update
                results[item_id] = error
                errors.append(f'{Path(spec.path).name}: {error}')
        self.finished.emit(request_id, results, '; '.join(errors), time.perf_counter() - start, False)

    @Slot(object, object, str)
    def load_view(self, key: tuple, spec: StructureSpec, mode: str) -> None:
        try:
            structure = self.engine.particle(spec) if mode == 'particle' else self.engine.unit_cell(spec.path, mode)
            self.viewReady.emit(key, structure, '')
        except Exception as error:
            self.viewReady.emit(key, None, str(error))


class MainWindow(QMainWindow):
    request = Signal(int, object, object, object)
    view_request = Signal(object, object, str)

    def __init__(self, files: Optional[List[str]] = None, restore_session: Optional[bool] = None,
                 settings: Optional[QSettings] = None):
        super().__init__()
        self.setWindowTitle(f'DebyeCalculator {package_version()}'.strip())
        self.resize(1500, 900)
        self.setAcceptDrops(True)

        self.settings = settings if settings is not None else QSettings('DebyeCalculator', 'DebyeCalculator')
        self.items: List[StructureItem] = []
        self.data_items: List[DataItem] = []
        self.params = Parameters()
        self.plot_options = PlotOptions()
        self.palette_name = 'Tableau 10'
        self.theme_mode = str(self.settings.value('theme', 'System'))
        self.dark = False
        self._request_ids = count(1)
        self._busy = False
        self._pending = False
        self._cancel_event: Optional[threading.Event] = None
        self._calculation_start = 0.0
        self._updating_controls = False
        self._closing = False
        self._pending_view_key = None
        self._displayed_view_key = None
        self._displayed_view = None
        self._queued_view = None
        self._view_debounce = QTimer(self, singleShot=True, interval=40)
        self._view_debounce.timeout.connect(self._dispatch_particle_view)

        self._build_ui()
        self._build_menu()
        self._start_worker()

        self._debounce = QTimer(self, singleShot=True, interval=15)
        self._debounce.timeout.connect(self._dispatch)
        self._slow_timer = QTimer(self, singleShot=True, interval=SLOW_CALCULATION_MS)
        self._slow_timer.timeout.connect(self._show_progress)

        self._apply_params_to_controls()
        self._apply_plot_options_to_controls()
        self.set_theme(self.theme_mode)
        hints = QGuiApplication.styleHints()
        if hasattr(hints, 'colorSchemeChanged'):
            hints.colorSchemeChanged.connect(lambda _scheme: self.theme_mode == 'System' and self.set_theme('System'))

        self._restore_window_state()
        if restore_session is None:
            restore_session = not files and self.restore_action.isChecked()
        if restore_session:
            self._restore_last_session()
        self.open_paths(files or [])

    # -- worker ------------------------------------------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = ComputeWorker()
        self._worker.moveToThread(self._thread)
        self.request.connect(self._worker.run)
        self.view_request.connect(self._worker.load_view)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self._on_progress)
        self._worker.viewReady.connect(self._on_view_ready)
        self._thread.start()

    def closeEvent(self, event) -> None:
        if not self._closing:
            self._closing = True
            self._save_window_state()
            self._view_debounce.stop()
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._thread.quit()
        if not self._thread.wait(100):
            self._set_status('Closing safely after the current structure operation…')
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        super().closeEvent(event)

    def schedule(self) -> None:
        """
        Request a recomputation. Requests arriving while a calculation runs collapse into one follow-up request.
        """
        self._debounce.start()

    def cancel_calculation(self) -> None:
        if self._busy and self._cancel_event is not None:
            self._pending = False
            self._cancel_event.set()
            self._set_status('Cancelling…')

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
        self._cancel_event = threading.Event()
        self._calculation_start = time.perf_counter()
        self._slow_timer.start()
        self._set_status('Calculating…')
        self.request.emit(next(self._request_ids), copy.deepcopy(self.params), jobs, self._cancel_event)

    def _show_progress(self) -> None:
        if self._busy:
            self.progress_bar.show()
            self.cancel_button.show()

    def _hide_progress(self) -> None:
        self._slow_timer.stop()
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        self.cancel_button.hide()

    @Slot(int, float)
    def _on_progress(self, _request_id: int, fraction: float) -> None:
        self.progress_bar.setValue(int(round(fraction * 100)))
        if self.progress_bar.isVisible():
            elapsed = time.perf_counter() - self._calculation_start
            remaining = elapsed / fraction - elapsed if fraction > 0.02 else None
            text = f'Calculating… {fraction * 100:.0f} %'
            if remaining is not None:
                text += f', about {remaining:.0f} s left'
            self._set_status(text)

    @Slot(int, object, str, float, bool)
    def _on_finished(self, _request_id: int, results: dict, error: str, elapsed: float, cancelled: bool) -> None:
        self._busy = False
        self._hide_progress()
        if cancelled:
            self._set_status('Calculation cancelled; the plots show the previous results')
            if self._pending:
                self._dispatch()
            return

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
        self._refresh_particle_view()

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
        self.cursor_label = ElidedLabel(' ')
        self.cursor_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plot_panel.cursorMoved.connect(lambda text: self.cursor_label.setText(text or ' '))
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)
        plot_layout.addWidget(self.plot_panel, 1)
        plot_layout.addWidget(self.cursor_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll(self._structures_tab()), 'Structures')
        self.tabs.addTab(self._scroll(self._data_tab()), 'Data')
        self.tabs.addTab(self._scroll(self._scattering_tab()), 'Scattering')
        self.tabs.addTab(self._scroll(self._plot_tab()), 'Plot')
        self.tabs.addTab(self._scroll(self._performance_tab()), 'Performance')
        self.tabs.setMinimumWidth(460)

        self.splitter = QSplitter()
        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(plot_area)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(0, False)
        self.splitter.setSizes([520, 980])
        self.setCentralWidget(self.splitter)

        self.particle_view = ParticleView()
        particle_panel = QWidget()
        particle_layout = QVBoxLayout(particle_panel)
        particle_layout.setContentsMargins(0, 0, 0, 0)
        particle_controls = QHBoxLayout()
        particle_controls.addWidget(QLabel('View'))
        self.particle_mode_combo = QComboBox()
        for label, mode in STRUCTURE_VIEW_MODES:
            self.particle_mode_combo.addItem(label, mode)
        saved_mode = str(self.settings.value('particle_mode', 'particle'))
        self.particle_mode_combo.setCurrentIndex(max(0, self.particle_mode_combo.findData(saved_mode)))
        self.particle_mode_combo.currentIndexChanged.connect(self._on_particle_mode_changed)
        particle_controls.addWidget(self.particle_mode_combo, 1)
        center_button = QPushButton('Center')
        center_button.setToolTip('Center the structure and reset zoom')
        center_button.clicked.connect(self.particle_view.center_view)
        particle_controls.addWidget(center_button)
        particle_layout.addLayout(particle_controls)
        particle_layout.addWidget(self.particle_view, 1)
        self.particle_dock = QDockWidget('Particle', self)
        self.particle_dock.setObjectName('particle_dock')
        self.particle_dock.setWidget(particle_panel)
        self.particle_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.particle_view.setMinimumWidth(260)
        self.addDockWidget(Qt.RightDockWidgetArea, self.particle_dock)
        self.particle_dock.hide()
        self.particle_dock.visibilityChanged.connect(self._on_particle_view_visibility_changed)

        self.status_label = QLabel()
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMaximumWidth(180)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setToolTip('Stop the running calculation (Esc)')
        self.cancel_button.clicked.connect(self.cancel_calculation)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self._hide_progress_widgets()

    def _hide_progress_widgets(self) -> None:
        self.progress_bar.hide()
        self.cancel_button.hide()

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

    @staticmethod
    def _make_table(headers: List[str], stretch_column: int) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        header = table.horizontalHeader()
        for column in range(len(headers)):
            header.setSectionResizeMode(column, QHeaderView.Stretch if column == stretch_column
                                        else QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.ElideMiddle)
        return table

    def _structures_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addLayout(self._button_row([
            ('Add files…', self.add_files_dialog, 'Add .cif, .xyz or other structure files (Ctrl+O)'),
            ('Duplicate', self.duplicate_selected, 'Copy the selected structure, e.g. to compare radii'),
            ('Remove', self.remove_selected, 'Remove the selected structure'),
        ]))

        self.table = self._make_table(['', '', 'Structure', 'Atoms'], stretch_column=2)
        self.table.setMinimumHeight(190)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 1)

        layout.addLayout(self._button_row([
            ('Show all', lambda: self._set_all_visible(True), None),
            ('Hide all', lambda: self._set_all_visible(False), None),
            ('Remove all', self.remove_all, None),
        ]))

        hint = QLabel('Tip: drop structure or data files onto the window to add them.')
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
        self.show_partials_check = QCheckBox('Show all element-pair partials')
        self.show_partials_check.setToolTip('Draw every element-pair contribution as a dashed curve; '
                                            'the partials add up to the total')
        self.show_partials_check.toggled.connect(self._on_show_partials_toggled)
        form.addRow('', self.show_partials_check)
        self.view_button = QPushButton('Show particle in 3D')
        self.view_button.clicked.connect(self.toggle_particle_view)
        form.addRow('', self.view_button)
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
        return page

    def _data_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(self._button_row([
            ('Load data…', self.add_data_dialog, 'Load measured I(Q), S(Q), F(Q) or G(r) (two-column text files)'),
            ('Remove', self.remove_selected_data, None),
        ]))
        self.data_table = self._make_table(['', 'Data', 'Function', 'Rw'], stretch_column=1)
        self.data_table.setMinimumHeight(150)
        self.data_table.itemSelectionChanged.connect(self._on_data_selection_changed)
        layout.addWidget(self.data_table, 1)

        self.data_box = QGroupBox('Selected data')
        form = QFormLayout(self.data_box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.data_label_edit = QLineEdit()
        self.data_label_edit.textEdited.connect(self._on_data_changed)
        form.addRow('Label', self.data_label_edit)
        self.data_function_combo = QComboBox()
        for key, (title, *_) in FUNCTIONS.items():
            self.data_function_combo.addItem(title, key)
        self.data_function_combo.currentIndexChanged.connect(self._on_data_changed)
        form.addRow('Function', self.data_function_combo)
        self.data_unit_combo = QComboBox()
        self.data_unit_combo.addItems(['Q', '2θ'])
        self.data_unit_combo.setToolTip('x-axis of the file for I(Q) data; 2θ is converted to Q with the wavelength '
                                        'set in the Plot tab')
        self.data_unit_combo.currentIndexChanged.connect(self._on_data_changed)
        form.addRow('x-axis of file', self.data_unit_combo)
        self.data_compare_combo = QComboBox()
        self.data_compare_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.data_compare_combo.setMinimumContentsLength(12)
        self.data_compare_combo.currentIndexChanged.connect(self._on_data_changed)
        form.addRow('Compare with', self.data_compare_combo)
        self.data_fit_check = QCheckBox('Fit scale of the calculation to the data')
        self.data_fit_check.toggled.connect(self._on_data_changed)
        form.addRow('', self.data_fit_check)
        self.data_difference_check = QCheckBox('Show difference curve')
        self.data_difference_check.toggled.connect(self._on_data_changed)
        form.addRow('', self.data_difference_check)
        self.data_result_label = QLabel('')
        self.data_result_label.setWordWrap(True)
        form.addRow('', self.data_result_label)
        self.data_box.setEnabled(False)
        layout.addWidget(self.data_box)

        note = QLabel('Rw = √(Σ(d − s·c)² / Σd²) over the data points inside the calculated range, '
                      'with s the fitted scale (1 without fit).')
        note.setWordWrap(True)
        note.setEnabled(False)
        layout.addWidget(note)
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

        axis_box = QGroupBox('I(Q) axis')
        axis_form = QFormLayout(axis_box)
        self.iq_axis_combo = QComboBox()
        self.iq_axis_combo.addItems(['Q [Å⁻¹]', '2θ [°]'])
        self.iq_axis_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        axis_form.addRow('x-axis', self.iq_axis_combo)
        wavelength_row = QHBoxLayout()
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.addItems(list(WAVELENGTHS))
        self.wavelength_combo.currentTextChanged.connect(self._on_wavelength_preset)
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setDecimals(4)
        self.wavelength_spin.setRange(0.01, 10.0)
        self.wavelength_spin.setSingleStep(0.01)
        self.wavelength_spin.setSuffix(' Å')
        self.wavelength_spin.setKeyboardTracking(False)
        self.wavelength_spin.valueChanged.connect(self._on_wavelength_changed)
        wavelength_row.addWidget(self.wavelength_combo, 1)
        wavelength_row.addWidget(self.wavelength_spin)
        axis_form.addRow('Wavelength', wavelength_row)
        layout.addWidget(axis_box)

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
        self.cursor_check = QCheckBox('Cursor readout')
        self.cursor_check.setToolTip('Crosshair with the values of all curves below the plots')
        self.legend_check = QCheckBox('Legend')
        checks = [self.normalize_check, self.log_iq_check, self.log_q_check, self.grid_check, self.markers_check,
                  self.auto_range_check, self.cursor_check, self.legend_check]
        for index, check in enumerate(checks):
            check.toggled.connect(self._on_plot_options_changed)
            appearance_layout.addWidget(check, index // 2, index % 2)
        row = len(checks) // 2
        appearance_layout.addWidget(QLabel('Legend position'), row, 0)
        self.legend_combo = QComboBox()
        self.legend_combo.addItems(list(LEGEND_POSITIONS))
        self.legend_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        appearance_layout.addWidget(self.legend_combo, row, 1)
        self.line_width_slider = FloatSlider('Line width', 0.5, 5.0, 1.5, decimals=1, step=0.5, label_width=70)
        self.line_width_slider.valueChanged.connect(self._on_plot_options_changed)
        appearance_layout.addWidget(self.line_width_slider, row + 1, 0, 1, 2)
        reset_view = QPushButton('Reset view (Ctrl+R)')
        reset_view.clicked.connect(self.plot_panel.reset_view)
        appearance_layout.addWidget(reset_view, row + 2, 0, 1, 2)
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
        cancel_hint = QLabel(f'Calculations taking longer than {SLOW_CALCULATION_MS} ms show a progress bar '
                             'and can be cancelled with Esc.')
        cancel_hint.setWordWrap(True)
        cancel_hint.setEnabled(False)
        updates_layout.addWidget(cancel_hint)
        layout.addWidget(updates)
        layout.addStretch(1)
        return page

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu('&File')

        def add(menu, text, slot, shortcut=None):
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)
            return action

        add(file_menu, 'Add structure files…', self.add_files_dialog, QKeySequence.Open)
        add(file_menu, 'Load experimental data…', self.add_data_dialog, QKeySequence('Ctrl+D'))
        self.recent_menu = file_menu.addMenu('Open recent')
        self._update_recent_menu()
        file_menu.addSeparator()
        add(file_menu, 'Open session…', self.load_session_dialog)
        add(file_menu, 'Save session…', self.save_session_dialog, QKeySequence.Save)
        file_menu.addSeparator()
        add(file_menu, 'Export data (CSV)…', self.export_data_dialog, QKeySequence('Ctrl+E'))
        add(file_menu, 'Export figure (PNG/SVG/PDF)…', self.export_figure_dialog, QKeySequence('Ctrl+Shift+E'))
        add(file_menu, 'Export particles (XYZ)…', self.export_particles_dialog)
        file_menu.addSeparator()
        add(file_menu, 'Quit', self.close, QKeySequence.Quit)

        view_menu = self.menuBar().addMenu('&View')
        add(view_menu, 'Reset view', self.plot_panel.reset_view, QKeySequence('Ctrl+R'))
        particle_action = self.particle_dock.toggleViewAction()
        particle_action.setText('Particle (3D)')
        particle_action.setShortcut(QKeySequence('Ctrl+3'))
        view_menu.addAction(particle_action)
        theme_menu = view_menu.addMenu('Theme')
        group = QActionGroup(self)
        self.theme_actions = {}
        for mode in THEME_MODES:
            action = QAction(mode, self, checkable=True)
            action.triggered.connect(lambda _=False, m=mode: self.set_theme(m))
            group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[mode] = action
        view_menu.addSeparator()
        self.restore_action = QAction('Restore last session at startup', self, checkable=True)
        self.restore_action.setChecked(self.settings.value('restore_session', True, type=bool))
        self.restore_action.toggled.connect(lambda checked: self.settings.setValue('restore_session', checked))
        view_menu.addAction(self.restore_action)

        cancel = QAction('Cancel calculation', self)
        cancel.setShortcut(QKeySequence(Qt.Key_Escape))
        cancel.triggered.connect(self.cancel_calculation)
        self.addAction(cancel)

    # -- theme and window state --------------------------------------------------------------------------------------

    def set_theme(self, mode: str) -> None:
        self.theme_mode = mode if mode in THEME_MODES else 'System'
        self.dark = apply_theme(self.theme_mode)
        self.settings.setValue('theme', self.theme_mode)
        self.theme_actions[self.theme_mode].setChecked(True)
        self.plot_panel.set_dark(self.dark)
        self.particle_view.set_dark(self.dark)
        self._refresh_table_values()
        self._set_status(self.status_label.text(), error=self.status_label.property('error') is True)

    def _save_window_state(self) -> None:
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('window_state', self.saveState())
        self.settings.setValue('splitter', self.splitter.saveState())
        self.settings.setValue('particle_mode', self.particle_mode_combo.currentData())
        self.settings.setValue('last_session', json.dumps(self.session()))

    def _restore_window_state(self) -> None:
        for key, restore in (('geometry', self.restoreGeometry), ('window_state', self.restoreState),
                             ('splitter', self.splitter.restoreState)):
            value = self.settings.value(key)
            if value is not None:
                restore(value)

    def _restore_last_session(self) -> None:
        text = self.settings.value('last_session')
        if not text:
            return
        try:
            session = json.loads(text)
            session['structures'] = [s for s in session.get('structures', []) if Path(s['spec']['path']).is_file()]
            session['data'] = [d for d in session.get('data', []) if Path(d['path']).is_file()]
            self.load_session(session)
        except (ValueError, KeyError, TypeError, OSError):
            pass

    # -- recent files ------------------------------------------------------------------------------------------------

    def _last_directory(self) -> str:
        return str(self.settings.value('last_directory', ''))

    def _remember(self, path: str) -> None:
        path = str(Path(path).resolve())
        self.settings.setValue('last_directory', str(Path(path).parent))
        recent = [p for p in self.settings.value('recent_files', [], type=list) if p != path]
        self.settings.setValue('recent_files', [path] + recent[:MAX_RECENT - 1])
        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self.settings.value('recent_files', [], type=list) if Path(p).is_file()]
        for path in recent:
            action = QAction(Path(path).name, self)
            action.setToolTip(path)
            action.triggered.connect(lambda _=False, p=path: self.open_path(p))
            self.recent_menu.addAction(action)
        self.recent_menu.setEnabled(bool(recent))
        if recent:
            self.recent_menu.addSeparator()
            clear = QAction('Clear list', self)
            clear.triggered.connect(lambda: (self.settings.setValue('recent_files', []), self._update_recent_menu()))
            self.recent_menu.addAction(clear)

    def open_path(self, path: str) -> None:
        """
        Add a structure or data file, depending on its suffix.
        """
        if Path(path).suffix.lower() in DATA_SUFFIXES:
            self.add_data(path)
        else:
            self.add_file(path)

    def open_paths(self, paths: List[str]) -> None:
        """Open several dropped or command-line files with batched GUI updates."""
        data_paths = [path for path in paths if Path(path).suffix.lower() in DATA_SUFFIXES]
        structure_paths = [path for path in paths if Path(path).suffix.lower() not in DATA_SUFFIXES]
        self.add_files(structure_paths)
        self.add_data_files(data_paths)

    # -- structures --------------------------------------------------------------------------------------------------

    def add_files_dialog(self) -> None:
        patterns = ' '.join(f'*{suffix}' for suffix in STRUCTURE_SUFFIXES)
        paths, _ = QFileDialog.getOpenFileNames(self, 'Add structure files', self._last_directory(),
                                                f'Structures ({patterns});;All files (*)')
        self.add_files(paths)

    def add_files(self, paths: List[str]) -> List[StructureItem]:
        """Add several structures with one table rebuild and one calculation request."""
        added = [self.add_file(path, _refresh=False) for path in paths]
        if added:
            self._finish_adding_files()
        return added

    def add_file(self, path: str, radius: float = 10.0, label: Optional[str] = None, color: Optional[str] = None,
                 visible: bool = True, lightweight: bool = False, partial: Optional[str] = None,
                 show_partials: bool = False, _refresh: bool = True) -> StructureItem:
        spec = StructureSpec(path=str(Path(path).resolve()), radius=radius, lightweight=lightweight, partial=partial,
                             show_partials=show_partials)
        if label is None:
            label = Path(path).stem + (f' (r = {radius:g} Å)' if spec.is_cif else '')
        if color is None:
            color = palette_colors(self.palette_name, len(self.items) + 1)[-1]
        item = StructureItem(spec, label, QColor(color), visible)
        self.items.append(item)
        self._remember(path)
        if _refresh:
            self._finish_adding_files()
        return item

    def _finish_adding_files(self, schedule: bool = True) -> None:
        self._rebuild_table()
        if self.items:
            self.table.selectRow(len(self.items) - 1)
        self._refresh_compare_combo()
        if schedule:
            self.schedule()

    def duplicate_selected(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        spec = item.spec
        self.add_file(spec.path, radius=spec.radius, label=f'{item.label} (copy)', lightweight=spec.lightweight,
                      partial=spec.partial, show_partials=spec.show_partials)

    def remove_selected(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        self.items.remove(item)
        self._rebuild_table()
        self._refresh_compare_combo()
        self._refresh_plots()
        self.schedule()

    def remove_all(self) -> None:
        self.items = []
        self._rebuild_table()
        self._refresh_compare_combo()
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

    def _visibility_cell(self, checked: bool, slot) -> QWidget:
        check = QCheckBox()
        check.setChecked(checked)
        check.setToolTip('Show')
        check.toggled.connect(slot)
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.addWidget(check)
        return cell

    def _rebuild_table(self) -> None:
        selected = self._selected_item()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.items))
        for row, item in enumerate(self.items):
            self.table.setCellWidget(row, 0, self._visibility_cell(
                item.visible, lambda checked, it=item: self._on_visible_toggled(it, checked)))
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
            self._refresh_particle_view()
            return
        self._updating_controls = True
        self.label_edit.setText(item.label)
        self.radius_slider.setEnabled(item.spec.is_cif)
        self.lightweight_check.setEnabled(item.spec.is_cif)
        self.radius_slider.setValue(item.spec.radius)
        self.lightweight_check.setChecked(item.spec.lightweight)
        self.show_partials_check.setChecked(item.spec.show_partials)
        self._fill_partials(item)
        self._updating_controls = False
        self._refresh_particle_view()

    def _fill_partials(self, item: StructureItem) -> None:
        self.partial_combo.blockSignals(True)
        self.partial_combo.clear()
        self.partial_combo.addItem('All pairs', None)
        for pair in (item.result.element_pairs if item.result is not None else []):
            self.partial_combo.addItem(pair, pair)
        index = self.partial_combo.findData(item.spec.partial)
        self.partial_combo.setCurrentIndex(max(index, 0))
        self.partial_combo.blockSignals(False)
        self.show_partials_check.setEnabled(item.spec.partial is None)

    def _refresh_partials(self) -> None:
        item = self._selected_item()
        if item is not None:
            self._fill_partials(item)

    def _on_visible_toggled(self, item: StructureItem, checked: bool) -> None:
        item.visible = checked
        self._refresh_plots()
        self._refresh_particle_view()
        self.schedule()

    def _on_color_changed(self, item: StructureItem, color: QColor) -> None:
        item.color = color
        self._refresh_plots()

    def _on_label_edited(self, text: str) -> None:
        item = self._selected_item()
        if item is not None:
            item.label = text
            self._refresh_table_values()
            self._refresh_compare_combo()
            self._refresh_plots()
            self._refresh_particle_view()

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
        self._refresh_compare_combo()
        self._refresh_particle_view()
        self.schedule()

    def _on_lightweight_toggled(self, checked: bool) -> None:
        item = self._selected_item()
        if item is not None and not self._updating_controls:
            item.spec.lightweight = checked
            self._refresh_particle_view()
            self.schedule()

    def _on_partial_changed(self, _index: int) -> None:
        item = self._selected_item()
        if item is not None and not self._updating_controls:
            item.spec.partial = self.partial_combo.currentData()
            self.show_partials_check.setEnabled(item.spec.partial is None)
            self.schedule()

    def _on_show_partials_toggled(self, checked: bool) -> None:
        item = self._selected_item()
        if item is not None and not self._updating_controls:
            item.spec.show_partials = checked
            self.schedule()

    # -- particle view -----------------------------------------------------------------------------------------------

    def toggle_particle_view(self) -> None:
        if self.particle_dock.isHidden():
            self.particle_dock.show()
            self.particle_dock.raise_()
            if not self.particle_dock.isFloating():
                # After the dock has been laid out
                QTimer.singleShot(0, lambda: self.particle_dock.width() < 380 and
                                  self.resizeDocks([self.particle_dock], [440], Qt.Horizontal))
            self._refresh_particle_view()
            QTimer.singleShot(50, self.particle_view.center_view)
        else:
            self.particle_dock.hide()

    def show_particle_view(self) -> None:
        """Compatibility helper for callers that explicitly open the viewer."""
        if self.particle_dock.isHidden():
            self.toggle_particle_view()

    def _on_particle_view_visibility_changed(self, visible: bool) -> None:
        self.view_button.setText('Hide particle in 3D' if visible else 'Show particle in 3D')
        if visible:
            self._refresh_particle_view()
        elif self._view_debounce.isActive():
            self._view_debounce.stop()
            self._queued_view = None
            self._pending_view_key = None

    def _on_particle_mode_changed(self, _index: int) -> None:
        self.settings.setValue('particle_mode', self.particle_mode_combo.currentData())
        self._refresh_particle_view()

    def _current_view_key(self):
        item = self._selected_item()
        if item is None:
            return None
        mode = self.particle_mode_combo.currentData()
        geometry_key = item.spec.particle_key() if mode == 'particle' else (item.spec.path, mode)
        return item.id, mode, geometry_key

    def _show_view_geometry(self, structure) -> None:
        item = self._selected_item()
        if item is None:
            return
        self.particle_view.set_structure(
            structure.elements, structure.xyz, item.label,
            lattice=getattr(structure, 'lattice', None), atom_count=getattr(structure, 'atom_count', None),
        )

    def _refresh_particle_view(self) -> None:
        if self.particle_dock.isHidden():
            return
        item = self._selected_item()
        key = self._current_view_key()
        if item is None or key is None:
            self._view_debounce.stop()
            self._queued_view = None
            self._pending_view_key = None
            self._displayed_view_key = None
            self._displayed_view = None
            self.particle_view.set_structure(None, None)
            return
        if key == self._displayed_view_key and self._displayed_view is not None:
            self._show_view_geometry(self._displayed_view)
            return
        if key == self._pending_view_key:
            return
        self._pending_view_key = key
        self.particle_view.set_message(item.label, 'Loading…')
        self._queued_view = (key, copy.deepcopy(item.spec), self.particle_mode_combo.currentData())
        self._view_debounce.start()

    def _dispatch_particle_view(self) -> None:
        request, self._queued_view = self._queued_view, None
        if request is None:
            return
        key, spec, mode = request
        if self.particle_dock.isHidden() or key != self._current_view_key():
            if key == self._pending_view_key:
                self._pending_view_key = None
            return
        self.view_request.emit(key, spec, mode)

    @Slot(object, object, str)
    def _on_view_ready(self, key: tuple, structure, error: str) -> None:
        if key == self._pending_view_key:
            self._pending_view_key = None
        if key != self._current_view_key() or self.particle_dock.isHidden():
            return
        item = self._selected_item()
        if error:
            self._displayed_view_key = None
            self._displayed_view = None
            self.particle_view.set_message(item.label if item is not None else '', error)
            return
        self._displayed_view_key = key
        self._displayed_view = structure
        self._show_view_geometry(structure)

    # -- experimental data -------------------------------------------------------------------------------------------

    def add_data_dialog(self) -> None:
        patterns = ' '.join(f'*{suffix}' for suffix in DATA_SUFFIXES)
        paths, _ = QFileDialog.getOpenFileNames(self, 'Load experimental data', self._last_directory(),
                                                f'Data ({patterns});;All files (*)')
        self.add_data_files(paths)

    def add_data_files(self, paths: List[str]) -> List[DataItem]:
        """Load several data files with one table and plot refresh."""
        added = [item for path in paths if (item := self.add_data(path, _refresh=False)) is not None]
        if added:
            self._finish_adding_data()
        return added

    def add_data(self, path: str, _refresh: bool = True, **kwargs) -> Optional[DataItem]:
        kwargs.setdefault('function', guess_function(path))
        kwargs.setdefault('wavelength', self.plot_options.wavelength)
        if 'compare_id' not in kwargs and self.items:
            selected = self._selected_item()
            kwargs['compare_id'] = (selected or self.items[0]).id
        try:
            item = DataItem(path, **kwargs)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, 'Load experimental data', f'Could not read {path}:\n{error}')
            return None
        self.data_items.append(item)
        self._remember(path)
        if item.function not in self.plot_options.functions:
            self.plot_options.functions = tuple(f for f in FUNCTIONS if f in self.plot_options.functions + (item.function,))
        if _refresh:
            self._finish_adding_data()
        return item

    def _finish_adding_data(self) -> None:
        self._apply_plot_options_to_controls()
        self.plot_panel.set_options(self.plot_options)
        self._rebuild_data_table()
        if self.data_items:
            self.data_table.selectRow(len(self.data_items) - 1)
            self.tabs.setCurrentIndex(1)
        self._refresh_plots()

    def remove_selected_data(self) -> None:
        item = self._selected_data()
        if item is not None:
            self.data_items.remove(item)
            self._rebuild_data_table()
            self._refresh_plots()

    def _selected_data(self) -> Optional[DataItem]:
        rows = self.data_table.selectionModel().selectedRows() if self.data_table.selectionModel() else []
        return self.data_items[rows[0].row()] if rows and rows[0].row() < len(self.data_items) else None

    def _rebuild_data_table(self) -> None:
        selected = self._selected_data()
        self.data_table.blockSignals(True)
        self.data_table.setRowCount(len(self.data_items))
        for row, item in enumerate(self.data_items):
            self.data_table.setCellWidget(row, 0, self._visibility_cell(
                item.visible, lambda checked, it=item: self._on_data_visible_toggled(it, checked)))
            for column in (1, 2, 3):
                cell = QTableWidgetItem()
                if column == 3:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.data_table.setItem(row, column, cell)
        if selected in self.data_items:
            self.data_table.selectRow(self.data_items.index(selected))
        self.data_table.blockSignals(False)
        self._refresh_data_table_values()
        self._on_data_selection_changed()

    def _refresh_data_table_values(self) -> None:
        if self.data_table.rowCount() != len(self.data_items):
            return
        for row, item in enumerate(self.data_items):
            cells = [self.data_table.item(row, column) for column in (1, 2, 3)]
            if any(cell is None for cell in cells):
                continue
            cells[0].setText(item.label)
            cells[0].setToolTip(item.path)
            cells[1].setText(FUNCTIONS[item.function][0])
            cells[2].setText(f'{item.comparison.rw:.4f}' if item.comparison is not None else '–')

    def _refresh_compare_combo(self) -> None:
        item = self._selected_data()
        self.data_compare_combo.blockSignals(True)
        self.data_compare_combo.clear()
        self.data_compare_combo.addItem('Nothing', None)
        for structure in self.items:
            self.data_compare_combo.addItem(structure.label, structure.id)
        if item is not None:
            self.data_compare_combo.setCurrentIndex(max(self.data_compare_combo.findData(item.compare_id), 0))
        self.data_compare_combo.blockSignals(False)

    def _on_data_selection_changed(self) -> None:
        item = self._selected_data()
        self.data_box.setEnabled(item is not None)
        if item is None:
            self.data_result_label.setText('')
            return
        self._updating_controls = True
        self.data_label_edit.setText(item.label)
        self.data_function_combo.setCurrentIndex(max(self.data_function_combo.findData(item.function), 0))
        self.data_unit_combo.setCurrentText(item.x_unit)
        self.data_unit_combo.setEnabled(item.function == 'i')
        self._refresh_compare_combo()
        self.data_fit_check.setChecked(item.fit_scale)
        self.data_difference_check.setChecked(item.show_difference)
        self._updating_controls = False
        self._refresh_data_result()

    def _on_data_changed(self, *_args) -> None:
        item = self._selected_data()
        if item is None or self._updating_controls:
            return
        item.label = self.data_label_edit.text()
        item.function = self.data_function_combo.currentData()
        item.x_unit = self.data_unit_combo.currentText()
        item.wavelength = self.plot_options.wavelength
        item.compare_id = self.data_compare_combo.currentData()
        item.fit_scale = self.data_fit_check.isChecked()
        item.show_difference = self.data_difference_check.isChecked()
        self.data_unit_combo.setEnabled(item.function == 'i')
        if item.function not in self.plot_options.functions:
            self._set_functions(self.plot_options.functions + (item.function,))
        self._refresh_plots()

    def _on_data_visible_toggled(self, item: DataItem, checked: bool) -> None:
        item.visible = checked
        self._refresh_plots()

    def _refresh_data_result(self) -> None:
        item = self._selected_data()
        if item is None or item.comparison is None:
            self.data_result_label.setText('No overlap with the compared calculation' if item and item.compare_id
                                           is not None else '')
            return
        c = item.comparison
        text = f'Scale {c.scale:.5g}   Rw {c.rw:.4f}   ({c.x.size} points)'
        compared = next((s for s in self.items if s.id == item.compare_id and s.result is not None), None)
        if compared is not None and c.x.size > 1:
            calc_x = compared.result.r if item.function == 'g' else compared.result.q
            calc_step, data_step = float(np.median(np.diff(calc_x))), float(np.median(np.diff(c.x)))
            if calc_step > 2 * data_step:
                name = 'rstep' if item.function == 'g' else 'Qstep'
                text += (f'\nThe calculated {name} ({calc_step:.3g}) is coarser than the data spacing '
                         f'({data_step:.3g}); a smaller {name} gives a more accurate comparison.')
        self.data_result_label.setText(text)

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
        self.iq_axis_combo.setCurrentIndex(1 if o.iq_two_theta else 0)
        preset = next((name for name, value in WAVELENGTHS.items() if value == o.wavelength), 'Custom')
        self.wavelength_combo.setCurrentText(preset)
        self.wavelength_spin.setValue(o.wavelength)
        self.normalize_check.setChecked(o.normalize)
        self.log_iq_check.setChecked(o.log_iq)
        self.log_q_check.setChecked(o.log_q)
        self.log_q_check.setText('Log 2θ axis (I(Q))' if o.iq_two_theta else 'Log Q axis (I(Q))')
        self.grid_check.setChecked(o.grid)
        self.markers_check.setChecked(o.markers)
        self.auto_range_check.setChecked(o.auto_range)
        self.cursor_check.setChecked(o.cursor)
        self.legend_check.setChecked(o.legend)
        self.legend_combo.setCurrentText(o.legend_position)
        self.legend_combo.setEnabled(o.legend)
        self.line_width_slider.setValue(o.line_width)
        self.dpi_spin.setValue(o.export_dpi)
        self.panel_width_spin.setValue(o.export_panel_width)
        self.panel_height_spin.setValue(o.export_panel_height)
        self._updating_controls = False

    def _on_wavelength_preset(self, name: str) -> None:
        value = WAVELENGTHS.get(name)
        if value is not None and not self._updating_controls:
            self.wavelength_spin.setValue(value)

    def _on_wavelength_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        preset = next((name for name, v in WAVELENGTHS.items() if v == value), 'Custom')
        self.wavelength_combo.blockSignals(True)
        self.wavelength_combo.setCurrentText(preset)
        self.wavelength_combo.blockSignals(False)
        for item in self.data_items:
            item.wavelength = value
        self._on_plot_options_changed()

    def _on_plot_options_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        o = self.plot_options
        o.functions = tuple(key for key, check in self.function_checks.items() if check.isChecked())
        o.mode = self.mode_combo.currentText()
        o.offset = self.offset_slider.value()
        o.columns = self.columns_spin.value()
        o.iq_two_theta = self.iq_axis_combo.currentIndex() == 1
        o.wavelength = self.wavelength_spin.value()
        o.normalize = self.normalize_check.isChecked()
        o.log_iq = self.log_iq_check.isChecked()
        o.log_q = self.log_q_check.isChecked()
        o.grid = self.grid_check.isChecked()
        o.markers = self.markers_check.isChecked()
        o.auto_range = self.auto_range_check.isChecked()
        o.cursor = self.cursor_check.isChecked()
        o.legend = self.legend_check.isChecked()
        o.legend_position = self.legend_combo.currentText()
        o.line_width = self.line_width_slider.value()
        o.export_dpi = self.dpi_spin.value()
        o.export_panel_width = self.panel_width_spin.value()
        o.export_panel_height = self.panel_height_spin.value()
        self.offset_slider.setEnabled(o.mode == 'Stacked')
        self.columns_spin.setEnabled(o.mode != 'Separate')
        self.legend_combo.setEnabled(o.legend)
        self.log_q_check.setText('Log 2θ axis (I(Q))' if o.iq_two_theta else 'Log Q axis (I(Q))')
        self.plot_panel.set_options(o)
        self._refresh_plots()

    def _plot_entries(self):
        shown = [item for item in self.items if item.visible and item.result is not None]
        index_of = {item.id: index for index, item in enumerate(shown)}
        entries = [PlotEntry(item.label, item.color, item.result) for item in shown]

        data_entries = []
        for data in self.data_items:
            data.comparison = None
            compared = next((item for item in shown if item.id == data.compare_id), None)
            if compared is not None:
                result = compared.result
                calc_x = result.r if data.function == 'g' else result.q
                data.comparison = compare(data.x, data.y, calc_x, getattr(result, data.function), data.fit_scale)
                if data.comparison is not None and data.fit_scale and data.visible:
                    entries[index_of[compared.id]].scales[data.function] = data.comparison.scale
            if not data.visible:
                continue
            show_difference = data.show_difference and data.comparison is not None
            data_entries.append(DataEntry(
                label=data.label, color=data.color, function=data.function, x=data.x, y=data.y,
                compare_index=index_of.get(data.compare_id),
                difference_x=data.comparison.x if show_difference else None,
                difference_y=data.comparison.difference if show_difference else None,
            ))
        return entries, data_entries

    def _refresh_plots(self) -> None:
        entries, data_entries = self._plot_entries()
        self.plot_panel.set_entries(entries, data_entries)
        self._refresh_data_table_values()
        self._refresh_data_result()

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setToolTip(text)
        self.status_label.setProperty('error', error)
        color = ('#ff6b68' if self.dark else '#c62828') if error else ''
        self.status_label.setStyleSheet(f'color: {color};' if color else '')

    # -- drag and drop -----------------------------------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.open_paths([path for path in paths if path and Path(path).is_file()])

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
        if self.plot_options.iq_two_theta:
            lines.append(f'# wavelength [Å]: {self.plot_options.wavelength:g}')
        return lines

    def export_data_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Export data to folder', self._last_directory())
        if directory:
            written = self.export_data(directory)
            self._set_status(f'Wrote {len(written)} files to {directory}')

    def export_data(self, directory: str) -> List[Path]:
        """
        Write <label>_Q.csv (Q, [2θ,] I, S, F) and <label>_r.csv (r, G) for every shown structure, and
        <label>_<pair>_Q.csv / _r.csv for shown partials.
        """
        written = []
        two_theta = self.plot_options.iq_two_theta
        for item in self.items:
            if not item.visible or item.result is None:
                continue
            stem = file_stem(item.label, f'structure_{item.id}')
            header = '\n'.join(self._metadata_lines(item))
            result = item.result
            sets = [(stem, {name: getattr(result, name) for name in 'isfg'})]
            sets += [(f'{stem}_{pair}', values) for pair, values in result.partials.items()]
            for name, values in sets:
                q_columns = [result.q] + ([q_to_two_theta(result.q, self.plot_options.wavelength)] if two_theta else [])
                q_header = 'Q [1/Å],' + ('2θ [deg],' if two_theta else '') + 'I(Q),S(Q),F(Q)'
                q_path = Path(directory) / f'{name}_Q.csv'
                np.savetxt(q_path, np.column_stack(q_columns + [values['i'], values['s'], values['f']]),
                           delimiter=',', header=header + '\n' + q_header, comments='')
                r_path = Path(directory) / f'{name}_r.csv'
                np.savetxt(r_path, np.column_stack([result.r, values['g']]), delimiter=',',
                           header=header + '\nr [Å],G(r)', comments='')
                written += [q_path, r_path]
        return written

    def export_figure_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Export figure', str(Path(self._last_directory()) / 'debyecalculator.png'),
                                              'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)')
        if path:
            try:
                self.plot_panel.export_image(path)
                self._set_status(f'Wrote {path}')
            except (OSError, ValueError) as error:
                self._set_status(f'Figure export failed: {error}', error=True)

    def export_particles_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Export particles to folder', self._last_directory())
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
                    structures=[item.to_dict() for item in self.items],
                    data=[dict(item.to_dict(), compare_index=next(
                        (index for index, s in enumerate(self.items) if s.id == item.compare_id), None))
                          for item in self.data_items])

    def save_session_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save session', str(Path(self._last_directory()) / 'session.json'),
                                              'Session (*.json)')
        if path:
            Path(path).write_text(json.dumps(self.session(), indent=1))
            self._set_status(f'Saved session to {path}')

    def load_session_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open session', self._last_directory(), 'Session (*.json)')
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
        self.data_items = []
        self._apply_params_to_controls()
        self._apply_plot_options_to_controls()
        self.plot_panel.set_options(self.plot_options)
        for entry in session.get('structures', []):
            spec = entry['spec']
            self.add_file(spec['path'], radius=spec.get('radius', 10.0), label=entry.get('label'),
                          color=entry.get('color'), visible=entry.get('visible', True),
                          lightweight=spec.get('lightweight', False), partial=spec.get('partial'),
                          show_partials=spec.get('show_partials', False), _refresh=False)
        self._finish_adding_files(schedule=False)
        for entry in session.get('data', []):
            compare_index = entry.get('compare_index')
            compare_id = self.items[compare_index].id if compare_index is not None and compare_index < len(self.items) else None
            self.add_data(entry['path'], _refresh=False, function=entry.get('function', 'i'),
                          x_unit=entry.get('x_unit', 'Q'),
                          wavelength=entry.get('wavelength', options.wavelength), label=entry.get('label'),
                          visible=entry.get('visible', True), color=entry.get('color'), compare_id=compare_id,
                          fit_scale=entry.get('fit_scale', True), show_difference=entry.get('show_difference', True))
        self._finish_adding_data()
        self.tabs.setCurrentIndex(0)
        self._refresh_plots()
        self.schedule()
