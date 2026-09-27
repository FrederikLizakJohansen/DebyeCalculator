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

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from debyecalculator.gui.engine import (
    STRUCTURE_SUFFIXES, Engine, Parameters, Result, StructureSpec, available_devices,
)
from debyecalculator.gui.plots import FUNCTIONS, MODES, PlotEntry, PlotOptions, PlotPanel
from debyecalculator.gui.widgets import ColorButton, FloatSlider

def package_version() -> str:
    try:
        from importlib.metadata import version
        return version('debyecalculator')
    except Exception:
        return ''


PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

PRESETS = {
    'Small-angle scattering': dict(params=dict(qmin=0.0, qmax=3.0, qstep=0.01), functions=('i',), log_iq=True, log_q=True),
    'Powder diffraction': dict(params=dict(qmin=1.0, qmax=8.0, qstep=0.1), functions=('i',), log_iq=False, log_q=False),
    'Total scattering': dict(params=dict(qmin=1.0, qmax=30.0, qstep=0.05), functions=('i', 'f', 'g'), log_iq=False, log_q=False),
}


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
        self.resize(1400, 850)
        self.setAcceptDrops(True)

        self.items: List[StructureItem] = []
        self.params = Parameters()
        self.plot_options = PlotOptions()
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
    def _on_finished(self, request_id: int, results: dict, error: str, elapsed: float) -> None:
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
        self._refresh_table()
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
        tabs = QTabWidget()
        tabs.addTab(self._scroll(self._structures_tab()), 'Structures')
        tabs.addTab(self._scroll(self._scattering_tab()), 'Scattering')
        tabs.addTab(self._scroll(self._plot_tab()), 'Plot')
        tabs.addTab(self._scroll(self._hardware_tab()), 'Hardware')
        tabs.setMinimumWidth(380)

        splitter = QSplitter()
        splitter.addWidget(tabs)
        splitter.addWidget(self.plot_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 980])
        self.setCentralWidget(splitter)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def _structures_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        buttons = QHBoxLayout()
        for text, slot in [('Add files…', self.add_files_dialog), ('Duplicate', self.duplicate_selected),
                           ('Remove', self.remove_selected)]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['', '', 'Structure', 'Atoms'])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        for column in (0, 1, 3):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(180)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)

        hint = QLabel('Drop .cif, .xyz and other structure files onto the window to add them.')
        hint.setStyleSheet('color: #777;')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.detail_box = QGroupBox('Selected structure')
        form = QFormLayout(self.detail_box)
        self.label_edit = QLineEdit()
        self.label_edit.textEdited.connect(self._on_label_edited)
        form.addRow('Label', self.label_edit)
        self.radius_slider = FloatSlider('Radius', 2.0, 60.0, 10.0, decimals=2, step=0.5, unit='Å', label_width=45)
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        form.addRow(self.radius_slider)
        self.lightweight_check = QCheckBox('Lightweight cut (all atoms within radius, no bond analysis)')
        self.lightweight_check.toggled.connect(self._on_lightweight_toggled)
        form.addRow(self.lightweight_check)
        self.partial_combo = QComboBox()
        self.partial_combo.currentIndexChanged.connect(self._on_partial_changed)
        form.addRow('Partial', self.partial_combo)
        self.detail_box.setEnabled(False)
        layout.addWidget(self.detail_box)
        layout.addStretch(1)
        return page

    def _scattering_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        presets = QGroupBox('Presets')
        preset_layout = QVBoxLayout(presets)
        row = QHBoxLayout()
        for name in PRESETS:
            button = QPushButton(name)
            button.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            row.addWidget(button)
        preset_layout.addLayout(row)
        reset = QPushButton('Reset to defaults')
        reset.clicked.connect(self.reset_parameters)
        preset_layout.addWidget(reset)
        layout.addWidget(presets)

        general = QGroupBox('Radiation')
        general_layout = QFormLayout(general)
        self.radiation_combo = QComboBox()
        self.radiation_combo.addItems(['X-ray', 'Neutron'])
        self.radiation_combo.currentIndexChanged.connect(self._on_params_changed)
        general_layout.addRow('Type', self.radiation_combo)
        layout.addWidget(general)

        q_box = QGroupBox('Q-space')
        q_layout = QVBoxLayout(q_box)
        self.qmin_slider = FloatSlider('Qmin', 0.0, 50.0, 1.0, decimals=2, step=0.1, unit='Å⁻¹')
        self.qmax_slider = FloatSlider('Qmax', 0.1, 60.0, 30.0, decimals=2, step=0.5, unit='Å⁻¹')
        self.qstep_auto = QCheckBox('Qstep from r-range (π / (rmax + rstep))')
        self.qstep_slider = FloatSlider('Qstep', 0.001, 0.5, 0.05, decimals=4, step=0.005, log=True, unit='Å⁻¹')
        self.biso_slider = FloatSlider('Biso', 0.0, 3.0, 0.3, decimals=3, step=0.05, unit='Å²')
        self.rthres_slider = FloatSlider('rthres', 0.0, 5.0, 0.0, decimals=2, step=0.1, unit='Å')
        for widget in (self.qmin_slider, self.qmax_slider, self.qstep_auto, self.qstep_slider, self.biso_slider,
                       self.rthres_slider):
            q_layout.addWidget(widget)
        layout.addWidget(q_box)

        r_box = QGroupBox('Real space (G(r))')
        r_layout = QVBoxLayout(r_box)
        self.rmin_slider = FloatSlider('rmin', 0.0, 100.0, 0.0, decimals=2, step=0.5, unit='Å')
        self.rmax_slider = FloatSlider('rmax', 1.0, 200.0, 20.0, decimals=2, step=0.5, unit='Å')
        self.rstep_slider = FloatSlider('rstep', 0.001, 0.5, 0.01, decimals=4, step=0.005, log=True, unit='Å')
        self.qdamp_slider = FloatSlider('Qdamp', 0.0, 0.2, 0.04, decimals=4, step=0.005, unit='Å⁻¹')
        self.lorch_check = QCheckBox('Lorch modification')
        for widget in (self.rmin_slider, self.rmax_slider, self.rstep_slider, self.qdamp_slider, self.lorch_check):
            r_layout.addWidget(widget)
        layout.addWidget(r_box)

        for slider in (self.qmin_slider, self.qmax_slider, self.qstep_slider, self.biso_slider, self.rthres_slider,
                       self.rmin_slider, self.rmax_slider, self.rstep_slider, self.qdamp_slider):
            slider.valueChanged.connect(self._on_params_changed)
        self.qstep_auto.toggled.connect(self._on_params_changed)
        self.lorch_check.toggled.connect(self._on_params_changed)
        layout.addStretch(1)
        return page

    def _plot_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        functions = QGroupBox('Functions')
        function_layout = QHBoxLayout(functions)
        self.function_checks = {}
        for key, (title, *_) in FUNCTIONS.items():
            check = QCheckBox(title)
            check.toggled.connect(self._on_plot_options_changed)
            function_layout.addWidget(check)
            self.function_checks[key] = check
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
        self.offset_slider = FloatSlider('Offset', 0.0, 3.0, 1.0, decimals=2, step=0.05, label_width=45)
        self.offset_slider.valueChanged.connect(self._on_plot_options_changed)
        form.addRow(self.offset_slider)
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 4)
        self.columns_spin.valueChanged.connect(self._on_plot_options_changed)
        form.addRow('Plots per row', self.columns_spin)
        layout.addWidget(arrangement)

        appearance = QGroupBox('Appearance')
        appearance_layout = QVBoxLayout(appearance)
        self.normalize_check = QCheckBox('Normalise each curve to its maximum')
        self.log_iq_check = QCheckBox('Logarithmic I(Q) axis')
        self.log_q_check = QCheckBox('Logarithmic Q axis for I(Q)')
        self.legend_check = QCheckBox('Legend')
        for check in (self.normalize_check, self.log_iq_check, self.log_q_check, self.legend_check):
            check.toggled.connect(self._on_plot_options_changed)
            appearance_layout.addWidget(check)
        self.line_width_slider = FloatSlider('Line width', 0.5, 5.0, 1.5, decimals=1, step=0.5)
        self.line_width_slider.valueChanged.connect(self._on_plot_options_changed)
        appearance_layout.addWidget(self.line_width_slider)
        reset_view = QPushButton('Reset view (auto-range)')
        reset_view.clicked.connect(self.plot_panel.reset_view)
        appearance_layout.addWidget(reset_view)
        layout.addWidget(appearance)
        layout.addStretch(1)
        return page

    def _hardware_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices())
        self.device_combo.currentIndexChanged.connect(self._on_params_changed)
        form.addRow('Device', self.device_combo)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(['float32', 'float64'])
        self.dtype_combo.currentIndexChanged.connect(self._on_params_changed)
        form.addRow('Precision', self.dtype_combo)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(10_000, 2_000_000_000)
        self.batch_spin.setSingleStep(1_000_000)
        self.batch_spin.setGroupSeparatorShown(True)
        self.batch_spin.setKeyboardTracking(False)
        self.batch_spin.valueChanged.connect(self._on_params_changed)
        form.addRow('Batch size (atom pairs)', self.batch_spin)
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
        item = StructureItem(spec, label, QColor(color or PALETTE[len(self.items) % len(PALETTE)]), visible)
        self.items.append(item)
        self._refresh_table()
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
        self._refresh_table()
        self._refresh_plots()
        self.schedule()

    def _selected_item(self) -> Optional[StructureItem]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.items[rows[0].row()] if rows and rows[0].row() < len(self.items) else None

    def _refresh_table(self) -> None:
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

            name = QTableWidgetItem(item.label)
            name.setToolTip(item.spec.path + (f'\n{item.error}' if item.error else ''))
            if item.error:
                name.setForeground(QColor('#c62828'))
            self.table.setItem(row, 2, name)
            atoms = QTableWidgetItem(f'{item.result.num_atoms:,}' if item.result is not None else '–')
            atoms.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 3, atoms)
        if selected in self.items:
            self.table.selectRow(self.items.index(selected))
        self.table.blockSignals(False)
        self._on_selection_changed()

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
            name = self.table.item(self.items.index(item), 2)
            if name is not None:
                name.setText(text)
            self._refresh_plots()

    def _on_radius_changed(self, radius: float) -> None:
        item = self._selected_item()
        if item is None or self._updating_controls or not item.spec.is_cif:
            return
        old_suffix = f' (r = {item.spec.radius:g} Å)'
        item.spec.radius = radius
        if item.label.endswith(old_suffix):
            item.label = item.label[: -len(old_suffix)] + f' (r = {radius:g} Å)'
            self.label_edit.setText(item.label)
            name = self.table.item(self.items.index(item), 2)
            if name is not None:
                name.setText(item.label)
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
        self.batch_spin.setValue(p.batch_size)
        self._updating_controls = False

    def _on_params_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        p = self.params
        p.radiation_type = 'xray' if self.radiation_combo.currentIndex() == 0 else 'neutron'
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
        p.batch_size = self.batch_spin.value()
        self.schedule()

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
        device, dtype, batch_size = self.params.device, self.params.dtype, self.params.batch_size
        self.params = Parameters(device=device, dtype=dtype, batch_size=batch_size)
        self.plot_options = PlotOptions()
        self._apply_params_to_controls()
        self._apply_plot_options_to_controls()
        self.plot_panel.set_options(self.plot_options)
        self.plot_panel.reset_view()
        self.schedule()

    # -- plotting ----------------------------------------------------------------------------------------------------

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
        self.legend_check.setChecked(o.legend)
        self.line_width_slider.setValue(o.line_width)
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
        o.legend = self.legend_check.isChecked()
        o.line_width = self.line_width_slider.value()
        self.offset_slider.setEnabled(o.mode == 'Stacked')
        self.columns_spin.setEnabled(o.mode != 'Separate')
        self.plot_panel.set_options(o)

    def _refresh_plots(self) -> None:
        entries = [PlotEntry(item.label, item.color, item.result) for item in self.items
                   if item.visible and item.result is not None]
        self.plot_panel.set_entries(entries)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet('color: #c62828;' if error else '')

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
        lines = [f'# DebyeCalculator {package_version()}'.rstrip(),
                 f'# structure: {item.spec.path}']
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
        path, _ = QFileDialog.getSaveFileName(self, 'Export figure', 'debyecalculator.png', 'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)')
        if path:
            self.plot_panel.export_image(path)
            self._set_status(f'Wrote {path}')

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
            stem = file_stem(item.label, f'structure_{item.id}')
            path = Path(directory) / f'{stem}.xyz'
            with open(path, 'w') as f:
                f.write(f'{particle.size}\n{item.label}\n')
                for element, (x, y, z) in zip(particle.elements, particle.xyz):
                    f.write(f'{element} {x:.6f} {y:.6f} {z:.6f}\n')
            written.append(path)
        return written

    # -- sessions ----------------------------------------------------------------------------------------------------

    def session(self) -> dict:
        return dict(parameters=self.params.to_dict(), plot=asdict(self.plot_options),
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
