"""
Plot area of the desktop GUI, based on pyqtgraph, and figure export with matplotlib.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from debyecalculator.gui.data import q_to_two_theta
from debyecalculator.gui.engine import Result

# key: (title, x attribute, x label, y label)
FUNCTIONS = {
    'i': ('I(Q)', 'q', 'Q [Å⁻¹]', 'I(Q) [counts]'),
    's': ('S(Q)', 'q', 'Q [Å⁻¹]', 'S(Q)'),
    'f': ('F(Q)', 'q', 'Q [Å⁻¹]', 'F(Q) [Å⁻¹]'),
    'g': ('G(r)', 'r', 'r [Å]', 'G(r) [Å⁻²]'),
}
TWO_THETA_LABEL = '2θ [°]'

MODES = ('Overlay', 'Stacked', 'Separate')
LEGEND_POSITIONS = {'Top right': (-10, 10), 'Top left': (10, 10), 'Bottom right': (-10, -10), 'Bottom left': (10, -10)}

THEMES = {
    False: dict(background='#ffffff', foreground='#202020', grid_alpha=0.25, data='#202020', cursor='#707070'),
    True: dict(background='#1e1f22', foreground='#d4d4d4', grid_alpha=0.18, data='#e8e8e8', cursor='#9a9a9a'),
}

# Fixed width of the left axis, so that tick labels of changing magnitude do not resize the plots
LEFT_AXIS_WIDTH = 72

PARTIAL_STYLES = [Qt.DashLine, Qt.DotLine, Qt.DashDotLine, Qt.DashDotDotLine]


@dataclass
class PlotEntry:
    label: str
    color: QColor
    result: Result
    scales: Dict[str, float] = field(default_factory=dict)  # per function, e.g. fitted to experimental data
    show_total: bool = True
    show_partials: bool = True


@dataclass
class DataEntry:
    label: str
    color: Optional[QColor]          # None: theme foreground
    function: str
    x: np.ndarray                     # Q or r
    y: np.ndarray
    compare_index: Optional[int] = None   # index of the compared PlotEntry
    difference_x: Optional[np.ndarray] = None
    difference_y: Optional[np.ndarray] = None


@dataclass
class PlotOptions:
    functions: Tuple[str, ...] = ('i', 's', 'f', 'g')
    mode: str = 'Overlay'
    offset: float = 1.0          # stacked mode: spacing in units of the largest curve amplitude
    normalize: bool = False      # scale each curve to a maximum absolute value of 1
    log_iq: bool = False         # logarithmic y-axis for I(Q)
    log_q: bool = False          # logarithmic Q-axis for I(Q)
    iq_two_theta: bool = False   # I(Q) against 2θ instead of Q
    wavelength: float = 0.7107   # Å, for the 2θ axis
    legend: bool = True
    legend_position: str = 'Top right'
    grid: bool = True
    markers: bool = False
    auto_range: bool = True      # rescale the axes to the data on every update
    cursor: bool = True          # crosshair with value readout
    line_width: float = 1.5
    columns: int = 2             # overlay and stacked mode: plots per row
    export_dpi: int = 300
    export_panel_width: float = 4.2   # inches per plot in exported figures
    export_panel_height: float = 3.0


@dataclass
class Curve:
    """
    A curve as displayed in one plot.
    """
    label: str
    color: QColor
    x: np.ndarray
    y: np.ndarray
    style: Qt.PenStyle = Qt.SolidLine
    width_factor: float = 1.0
    symbol: Optional[str] = None


def x_values(options: PlotOptions, function: str, x: np.ndarray) -> np.ndarray:
    if function == 'i' and options.iq_two_theta:
        return q_to_two_theta(x, options.wavelength)
    return x


def x_label(options: PlotOptions, function: str) -> str:
    return TWO_THETA_LABEL if function == 'i' and options.iq_two_theta else FUNCTIONS[function][2]


def _normalized(y: np.ndarray, options: PlotOptions) -> Tuple[np.ndarray, float]:
    peak = np.nanmax(np.abs(y)) if y.size else 0.0
    if options.normalize and peak > 0:
        return y / peak, peak
    return y, 1.0


def build_curves(entries: List[PlotEntry], data: List[DataEntry], options: PlotOptions, function: str,
                 indices: List[int], data_color: str) -> List[Curve]:
    """
    Curves of one plot: calculated patterns of the given entries (with partials), experimental data and
    difference curves, normalised and offset as displayed.
    """
    x_attr = FUNCTIONS[function][1]
    log_y = function == 'i' and options.log_iq
    stacked = options.mode == 'Stacked'

    base = {}
    for k in indices:
        y = np.asarray(getattr(entries[k].result, function), dtype=float) * entries[k].scales.get(function, 1.0)
        base[k] = _normalized(y, options)
    amplitude = max((np.nanmax(np.abs(y)) for y, _ in base.values() if y.size), default=1.0) or 1.0

    def shift(y, position):
        if not stacked or position == 0:
            return y
        return y * 10 ** (position * options.offset) if log_y else y + position * options.offset * amplitude

    curves: List[Curve] = []
    position_of = {k: position for position, k in enumerate(indices)}
    for k in indices:
        entry = entries[k]
        y, norm = base[k]
        x = x_values(options, function, getattr(entry.result, x_attr))
        symbol = 'o' if options.markers else None
        if entry.show_total:
            curves.append(Curve(entry.label, entry.color, x, shift(y, position_of[k]), symbol=symbol))
        for number, (pair, values) in enumerate(entry.result.partials.items() if entry.show_partials else ()):
            partial_y = np.asarray(values[function], dtype=float) * entry.scales.get(function, 1.0) / norm
            shade = entry.color.darker(140) if number % 2 == 0 else entry.color.lighter(135 + 15 * number)
            curves.append(Curve(f'{entry.label} · {pair}', shade, x, shift(partial_y, position_of[k]),
                                style=PARTIAL_STYLES[number % len(PARTIAL_STYLES)], width_factor=0.8))

    # In separate mode, data compared with a structure appears only in that structure's row
    shown_data = [d for d in data if d.function == function and
                  (options.mode != 'Separate' or d.compare_index is None or d.compare_index in indices)]
    lowest = min((np.nanmin(c.y) for c in curves if c.y.size), default=0.0)
    for d in shown_data:
        y, norm = _normalized(np.asarray(d.y, dtype=float), options)
        position = position_of.get(d.compare_index, 0)
        color = QColor(data_color) if d.color is None else d.color
        curves.append(Curve(d.label, color, x_values(options, function, d.x), shift(y, position),
                            width_factor=0.9, symbol='o' if options.markers else None))
        lowest = min(lowest, np.nanmin(shift(y, position)))
        if d.difference_x is not None and not log_y and d.compare_index in indices:
            difference = d.difference_y / norm
            spread = amplitude if not options.normalize else 1.0
            offset = lowest - 0.15 * spread - np.nanmax(difference)
            curves.append(Curve(f'{d.label} − calc.', QColor('#2ca02c'), x_values(options, function, d.difference_x),
                                difference + offset, width_factor=0.8))
    return curves


class PlotPanel(pg.GraphicsLayoutWidget):
    cursorMoved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.options = PlotOptions()
        self.dark = False
        self._entries: List[PlotEntry] = []
        self._data: List[DataEntry] = []
        self._layout_key = None
        self._plots: Dict[tuple, pg.PlotItem] = {}
        self._curves: Dict[tuple, pg.PlotDataItem] = {}
        self._curve_data: Dict[tuple, Curve] = {}
        self._legends: Dict[tuple, pg.LegendItem] = {}
        self._cursor_lines: Dict[tuple, pg.InfiniteLine] = {}
        self.setBackground(THEMES[False]['background'])
        self._mouse_proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    # -- public ------------------------------------------------------------------------------------------------------

    def set_options(self, options: PlotOptions) -> None:
        self.options = options
        self._redraw()

    def set_entries(self, entries: List[PlotEntry], data: Optional[List[DataEntry]] = None) -> None:
        self._entries = entries
        self._data = data or []
        self._redraw()

    def set_dark(self, dark: bool) -> None:
        self.dark = dark
        self._layout_key = None
        self._redraw()

    def reset_view(self) -> None:
        for plot in self._plots.values():
            plot.enableAutoRange()

    def export_image(self, path: str) -> None:
        """
        Write the current plots with matplotlib (format from the file suffix, e.g. .png, .svg, .pdf).
        """
        export_figure(self._entries, self._data, self.options, path)

    # -- layout ------------------------------------------------------------------------------------------------------

    def _rows(self) -> List[Tuple[int, str]]:
        """
        Plot keys: (row, function). In separate mode, row is the entry index; otherwise 0.
        """
        functions = [f for f in FUNCTIONS if f in self.options.functions]
        if self.options.mode == 'Separate':
            return [(row, f) for row in range(len(self._entries)) for f in functions]
        return [(0, f) for f in functions]

    def _style_plot(self, plot: pg.PlotItem, title: str) -> None:
        theme = THEMES[self.dark]
        foreground = theme['foreground']
        for name in ('left', 'bottom', 'top', 'right'):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(foreground))
            axis.setTextPen(pg.mkPen(foreground))
            axis.enableAutoSIPrefix(False)
        plot.getAxis('left').setWidth(LEFT_AXIS_WIDTH)
        plot.showGrid(x=self.options.grid, y=self.options.grid, alpha=theme['grid_alpha'])
        if title:
            plot.setTitle(title, size='10pt', color=foreground)

    def _build_layout(self) -> None:
        self.clear()
        self._plots.clear()
        self._curves.clear()
        self._curve_data.clear()
        self._legends.clear()
        self._cursor_lines.clear()
        theme = THEMES[self.dark]
        self.setBackground(theme['background'])
        separate = self.options.mode == 'Separate'
        functions = [f for f in FUNCTIONS if f in self.options.functions]
        columns = len(functions) if separate else max(1, min(self.options.columns, len(functions)))
        foreground = theme['foreground']

        first_in_column: Dict[str, pg.PlotItem] = {}
        for index, (row, function) in enumerate(self._rows()):
            title, _, _, y_label = FUNCTIONS[function]
            if separate:
                grid_row, grid_col = row, functions.index(function)
            else:
                grid_row, grid_col = divmod(index, columns)
            plot = self.addPlot(row=grid_row, col=grid_col)
            label_style = {'color': foreground}
            plot.setLabel('bottom', x_label(self.options, function), **label_style)
            if separate:
                # Function names on the top row; the structure label joins the y-label of the first column
                self._style_plot(plot, title if row == 0 else '')
                y_text = f'{self._entries[row].label}<br>{y_label}' if grid_col == 0 else y_label
                plot.setLabel('left', y_text, **label_style)
                if grid_col == 0:
                    # Room for the second label line
                    plot.getAxis('left').setWidth(LEFT_AXIS_WIDTH + 18)
                if function in first_in_column:
                    plot.setXLink(first_in_column[function])
                else:
                    first_in_column[function] = plot
            else:
                self._style_plot(plot, title)
                plot.setLabel('left', y_label, **label_style)
            self._plots[(row, function)] = plot
            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme['cursor'], style=Qt.DashLine))
            line.setVisible(False)
            plot.addItem(line, ignoreBounds=True)
            self._cursor_lines[(row, function)] = line
            if self.options.legend and not separate and index == 0:
                legend = plot.addLegend(offset=LEGEND_POSITIONS.get(self.options.legend_position, (-10, 10)))
                legend.setLabelTextColor(foreground)
                self._legends[(row, function)] = legend

    # -- drawing -----------------------------------------------------------------------------------------------------

    def _redraw(self) -> None:
        o = self.options
        layout_key = (o.mode, tuple(o.functions), o.columns, o.legend, o.legend_position, o.grid, self.dark,
                      o.iq_two_theta,
                      len(self._entries) if o.mode == 'Separate' else 0,
                      tuple(e.label for e in self._entries) if o.mode == 'Separate' else ())
        if layout_key != self._layout_key:
            self._build_layout()
            self._layout_key = layout_key

        for (row, function), plot in self._plots.items():
            is_iq = function == 'i'
            plot.setLogMode(x=is_iq and o.log_q, y=is_iq and o.log_iq)
            if o.auto_range:
                plot.enableAutoRange()
            else:
                plot.disableAutoRange()

        separate = o.mode == 'Separate'
        data_color = THEMES[self.dark]['data']
        wanted = set()
        for (row, function), plot in self._plots.items():
            indices = [row] if separate else list(range(len(self._entries)))
            curves = build_curves(self._entries, self._data, o, function, indices, data_color)
            log_y = function == 'i' and o.log_iq
            log_x = function == 'i' and o.log_q
            for number, curve in enumerate(curves):
                x, y = curve.x, curve.y
                if log_y:
                    y = np.where(y > 0, y, np.nan)
                keep = np.isfinite(x) & (x > 0 if log_x else True)
                x, y = x[keep], y[keep]
                key = (row, function, number)
                wanted.add(key)
                pen = pg.mkPen(curve.color, width=o.line_width * curve.width_factor, style=curve.style)
                style = dict(pen=pen, connect='finite', symbol=curve.symbol, symbolSize=4, symbolPen=None,
                             symbolBrush=curve.color)
                item = self._curves.get(key)
                if item is None:
                    item = plot.plot(x, y, name=curve.label, **style)
                    self._curves[key] = item
                else:
                    item.setData(x, y, **style)
                    item.opts['name'] = curve.label
                self._curve_data[key] = curve

        for key in list(self._curves):
            if key not in wanted:
                row, function, _ = key
                plot = self._plots.get((row, function))
                if plot is not None:
                    plot.removeItem(self._curves[key])
                del self._curves[key]
                self._curve_data.pop(key, None)

        for (row, function), legend in self._legends.items():
            legend.clear()
            for key, item in sorted(self._curves.items()):
                if key[:2] == (row, function):
                    legend.addItem(item, self._curve_data[key].label)

    # -- cursor ------------------------------------------------------------------------------------------------------

    def _mouse_moved(self, event) -> None:
        position = event[0]
        hovered = None
        for key, plot in self._plots.items():
            if plot.sceneBoundingRect().contains(position) and plot.vb.sceneBoundingRect().contains(position):
                hovered = key
        for key, line in self._cursor_lines.items():
            line.setVisible(self.options.cursor and key == hovered)
        if hovered is None or not self.options.cursor:
            self.cursorMoved.emit('')
            return

        row, function = hovered
        plot = self._plots[hovered]
        view_x = plot.vb.mapSceneToView(position).x()
        self._cursor_lines[hovered].setPos(view_x)
        x = 10 ** view_x if function == 'i' and self.options.log_q else view_x
        values = []
        for key, curve in self._curve_data.items():
            if key[:2] != hovered or curve.x.size < 2:
                continue
            finite = np.isfinite(curve.x) & np.isfinite(curve.y)
            cx, cy = curve.x[finite], curve.y[finite]
            if cx.size < 2 or not (cx.min() <= x <= cx.max()):
                continue
            order = np.argsort(cx)
            values.append(f'{curve.label}: {np.interp(x, cx[order], cy[order]):.5g}')
        name = x_label(self.options, function).split(' ')[0]
        self.cursorMoved.emit(f'{FUNCTIONS[function][0]}  {name} = {x:.4g}   ' + '   '.join(values))


def export_figure(entries: List[PlotEntry], data: List[DataEntry], options: PlotOptions, path: str) -> None:
    import matplotlib
    from matplotlib.figure import Figure

    functions = [f for f in FUNCTIONS if f in options.functions]
    if not functions or not entries:
        raise ValueError('Nothing to export')
    separate = options.mode == 'Separate'
    if separate:
        n_rows, n_cols = len(entries), len(functions)
    else:
        n_cols = max(1, min(options.columns, len(functions)))
        n_rows = int(np.ceil(len(functions) / n_cols))

    line_styles = {Qt.SolidLine: '-', Qt.DashLine: '--', Qt.DotLine: ':', Qt.DashDotLine: '-.',
                   Qt.DashDotDotLine: (0, (3, 1, 1, 1, 1, 1))}
    legend_locations = {'Top right': 'upper right', 'Top left': 'upper left',
                        'Bottom right': 'lower right', 'Bottom left': 'lower left'}
    with matplotlib.rc_context({'font.size': 9}):
        fig = Figure(figsize=(options.export_panel_width * n_cols, options.export_panel_height * n_rows),
                     constrained_layout=True)
        axes = np.atleast_2d(fig.subplots(n_rows, n_cols, squeeze=False))
        used = set()
        for index, function in enumerate(functions):
            title, _, _, y_label = FUNCTIONS[function]
            rows = range(len(entries)) if separate else [0]
            for row in rows:
                if separate:
                    ax = axes[row, index]
                    indices = [row]
                else:
                    ax = axes[divmod(index, n_cols)]
                    indices = list(range(len(entries)))
                used.add(id(ax))
                for curve in build_curves(entries, data, options, function, indices, '#202020'):
                    x, y = curve.x, curve.y
                    keep = np.isfinite(x) & ((x > 0) if (function == 'i' and options.log_q) else True)
                    ax.plot(x[keep], y[keep], color=curve.color.name(), label=curve.label,
                            linewidth=options.line_width * 0.8 * curve.width_factor,
                            linestyle=line_styles.get(curve.style, '-'),
                            marker='o' if curve.symbol else None, markersize=2)
                if function == 'i':
                    if options.log_iq:
                        ax.set_yscale('log')
                    if options.log_q:
                        ax.set_xscale('log')
                ax.set_xlabel(x_label(options, function))
                ax.set_ylabel(f'{entries[row].label}\n{y_label}' if separate and index == 0 else y_label)
                if not separate or row == 0:
                    ax.set_title(title)
                if options.grid:
                    ax.grid(alpha=0.3)
                if options.legend and not separate and index == 0:
                    ax.legend(fontsize=8, frameon=False, loc=legend_locations.get(options.legend_position, 'best'))
        for ax in axes.flat:
            if id(ax) not in used:
                ax.set_visible(False)
        fig.savefig(path, dpi=options.export_dpi)
