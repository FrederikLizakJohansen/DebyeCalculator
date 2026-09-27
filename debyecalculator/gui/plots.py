"""
Plot area of the desktop GUI, based on pyqtgraph.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor

from debyecalculator.gui.engine import Result

# key: (title, x attribute, x label, y label)
FUNCTIONS = {
    'i': ('I(Q)', 'q', 'Q [Å⁻¹]', 'I(Q) [counts]'),
    's': ('S(Q)', 'q', 'Q [Å⁻¹]', 'S(Q)'),
    'f': ('F(Q)', 'q', 'Q [Å⁻¹]', 'F(Q) [Å⁻¹]'),
    'g': ('G(r)', 'r', 'r [Å]', 'G(r) [Å⁻²]'),
}

MODES = ('Overlay', 'Stacked', 'Separate')


@dataclass
class PlotEntry:
    label: str
    color: QColor
    result: Result


@dataclass
class PlotOptions:
    functions: Tuple[str, ...] = ('i', 's', 'f', 'g')
    mode: str = 'Overlay'
    offset: float = 1.0          # stacked mode: spacing in units of the largest curve amplitude
    normalize: bool = False      # scale each curve to a maximum absolute value of 1
    log_iq: bool = False         # logarithmic y-axis for I(Q)
    log_q: bool = False          # logarithmic Q-axis for I(Q)
    legend: bool = True
    line_width: float = 1.5
    columns: int = 2             # overlay and stacked mode: plots per row


class PlotPanel(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.options = PlotOptions()
        self._entries: List[PlotEntry] = []
        self._layout_key = None
        self._plots: Dict[tuple, pg.PlotItem] = {}
        self._curves: Dict[tuple, pg.PlotDataItem] = {}
        self._legends: Dict[tuple, pg.LegendItem] = {}

    # -- public ------------------------------------------------------------------------------------------------------

    def set_options(self, options: PlotOptions) -> None:
        self.options = options
        self._redraw()

    def set_entries(self, entries: List[PlotEntry]) -> None:
        self._entries = entries
        self._redraw()

    def reset_view(self) -> None:
        for plot in self._plots.values():
            plot.enableAutoRange()

    def export_image(self, path: str) -> None:
        """
        Write the current plots with matplotlib (format from the file suffix, e.g. .png, .svg, .pdf).
        """
        export_figure(self._entries, self.options, path)

    # -- layout ------------------------------------------------------------------------------------------------------

    def _rows(self) -> List[Tuple[int, str]]:
        """
        Plot keys: (row, function). In separate mode, row is the entry index; otherwise 0.
        """
        functions = [f for f in FUNCTIONS if f in self.options.functions]
        if self.options.mode == 'Separate':
            return [(row, f) for row in range(len(self._entries)) for f in functions]
        return [(0, f) for f in functions]

    def _build_layout(self) -> None:
        self.clear()
        self._plots.clear()
        self._curves.clear()
        self._legends.clear()
        separate = self.options.mode == 'Separate'
        functions = [f for f in FUNCTIONS if f in self.options.functions]
        columns = len(functions) if separate else max(1, min(self.options.columns, len(functions)))

        first_in_column: Dict[str, pg.PlotItem] = {}
        for index, (row, function) in enumerate(self._rows()):
            title, _, x_label, y_label = FUNCTIONS[function]
            if separate:
                grid_row, grid_col = row, functions.index(function)
            else:
                grid_row, grid_col = divmod(index, columns)
            plot = self.addPlot(row=grid_row, col=grid_col)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel('bottom', x_label)
            plot.setLabel('left', y_label)
            for axis in ('left', 'bottom'):
                plot.getAxis(axis).enableAutoSIPrefix(False)
            if separate:
                # Function names on the top row; the structure label replaces the y-label of the first column
                if row == 0:
                    plot.setTitle(title, size='10pt')
                if grid_col == 0:
                    plot.setLabel('left', f'{self._entries[row].label}<br>{y_label}')
                if function in first_in_column:
                    plot.setXLink(first_in_column[function])
                else:
                    first_in_column[function] = plot
            else:
                plot.setTitle(title, size='10pt')
            self._plots[(row, function)] = plot
            if self.options.legend and (separate is False) and index == 0:
                self._legends[(row, function)] = plot.addLegend(offset=(-10, 10))

    # -- drawing -----------------------------------------------------------------------------------------------------

    def _redraw(self) -> None:
        layout_key = (self.options.mode, tuple(self.options.functions), self.options.columns, self.options.legend,
                      len(self._entries) if self.options.mode == 'Separate' else 0,
                      tuple(e.label for e in self._entries) if self.options.mode == 'Separate' else ())
        if layout_key != self._layout_key:
            self._build_layout()
            self._layout_key = layout_key

        for (row, function), plot in self._plots.items():
            is_iq = function == 'i'
            plot.setLogMode(x=is_iq and self.options.log_q, y=is_iq and self.options.log_iq)

        separate = self.options.mode == 'Separate'
        stacked = self.options.mode == 'Stacked'
        wanted = set()
        for (row, function), plot in self._plots.items():
            _, x_attr, _, _ = FUNCTIONS[function]
            indices = [row] if separate else range(len(self._entries))
            ys = {k: self._curve_values(self._entries[k].result, function) for k in indices}
            amplitude = max((np.nanmax(np.abs(y)) for y in ys.values() if y.size), default=1.0) or 1.0
            log_y = function == 'i' and self.options.log_iq

            for position, k in enumerate(indices):
                entry = self._entries[k]
                x = getattr(entry.result, x_attr)
                y = ys[k]
                if stacked and position > 0:
                    # Multiplicative spacing on a logarithmic axis, additive otherwise
                    y = y * 10 ** (position * self.options.offset) if log_y else y + position * self.options.offset * amplitude
                if log_y:
                    y = np.where(y > 0, y, np.nan)
                if function == 'i' and self.options.log_q:
                    positive = x > 0
                    x, y = x[positive], y[positive]
                key = (row, function, k)
                wanted.add(key)
                pen = pg.mkPen(entry.color, width=self.options.line_width)
                curve = self._curves.get(key)
                if curve is None:
                    curve = plot.plot(x, y, pen=pen, name=entry.label, connect='finite')
                    self._curves[key] = curve
                else:
                    curve.setData(x, y, connect='finite')
                    curve.setPen(pen)
                    curve.opts['name'] = entry.label

        for key in list(self._curves):
            if key not in wanted:
                row, function, _ = key
                plot = self._plots.get((row, function))
                if plot is not None:
                    plot.removeItem(self._curves[key])
                del self._curves[key]

        for (row, function), legend in self._legends.items():
            legend.clear()
            for (r, f, k), curve in sorted(self._curves.items()):
                if (r, f) == (row, function):
                    legend.addItem(curve, self._entries[k].label)

    def _curve_values(self, result: Result, function: str) -> np.ndarray:
        y = np.asarray(getattr(result, function), dtype=float)
        if self.options.normalize and y.size:
            peak = np.nanmax(np.abs(y))
            if peak > 0:
                y = y / peak
        return y


def _stacked_values(entries: List[PlotEntry], options: PlotOptions, function: str, indices: List[int]) -> dict:
    """
    Curve values as displayed: normalised if requested and shifted in stacked mode.
    """
    values = {}
    for k in indices:
        y = np.asarray(getattr(entries[k].result, function), dtype=float)
        if options.normalize and y.size and np.nanmax(np.abs(y)) > 0:
            y = y / np.nanmax(np.abs(y))
        values[k] = y
    amplitude = max((np.nanmax(np.abs(y)) for y in values.values() if y.size), default=1.0) or 1.0
    log_y = function == 'i' and options.log_iq
    if options.mode == 'Stacked':
        for position, k in enumerate(indices):
            if position > 0:
                values[k] = (values[k] * 10 ** (position * options.offset) if log_y
                             else values[k] + position * options.offset * amplitude)
    return values


def export_figure(entries: List[PlotEntry], options: PlotOptions, path: str) -> None:
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

    with matplotlib.rc_context({'font.size': 9}):
        fig = Figure(figsize=(4.2 * n_cols, 3.0 * n_rows), constrained_layout=True)
        axes = np.atleast_2d(fig.subplots(n_rows, n_cols, squeeze=False))
        used = set()
        for index, function in enumerate(functions):
            title, x_attr, x_label, y_label = FUNCTIONS[function]
            rows = range(len(entries)) if separate else [0]
            for row in rows:
                if separate:
                    ax = axes[row, index]
                    indices = [row]
                else:
                    ax = axes[divmod(index, n_cols)]
                    indices = list(range(len(entries)))
                used.add(id(ax))
                values = _stacked_values(entries, options, function, indices)
                for k in indices:
                    entry = entries[k]
                    x, y = getattr(entry.result, x_attr), values[k]
                    if function == 'i' and options.log_q:
                        x, y = x[x > 0], y[x > 0]
                    ax.plot(x, y, color=entry.color.name(), linewidth=options.line_width * 0.8, label=entry.label)
                if function == 'i':
                    if options.log_iq:
                        ax.set_yscale('log')
                    if options.log_q:
                        ax.set_xscale('log')
                ax.set_xlabel(x_label)
                ax.set_ylabel(f'{entries[row].label}\n{y_label}' if separate and index == 0 else y_label)
                if not separate or row == 0:
                    ax.set_title(title)
                ax.grid(alpha=0.3)
                if options.legend and not separate and index == 0:
                    ax.legend(fontsize=8, frameon=False)
        for ax in axes.flat:
            if id(ax) not in used:
                ax.set_visible(False)
        fig.savefig(path, dpi=300)
