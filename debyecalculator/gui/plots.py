"""
Plot area of the desktop GUI, based on pyqtgraph, and figure export with matplotlib.
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
LEGEND_POSITIONS = {'Top right': (-10, 10), 'Top left': (10, 10), 'Bottom right': (-10, -10), 'Bottom left': (10, -10)}

THEMES = {
    False: dict(background='#ffffff', foreground='#202020', grid_alpha=0.25),
    True: dict(background='#1e1f22', foreground='#d4d4d4', grid_alpha=0.18),
}

# Fixed width of the left axis, so that tick labels of changing magnitude do not resize the plots
LEFT_AXIS_WIDTH = 72


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
    legend_position: str = 'Top right'
    grid: bool = True
    markers: bool = False
    auto_range: bool = True      # rescale the axes to the data on every update
    line_width: float = 1.5
    columns: int = 2             # overlay and stacked mode: plots per row
    export_dpi: int = 300
    export_panel_width: float = 4.2   # inches per plot in exported figures
    export_panel_height: float = 3.0


class PlotPanel(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.options = PlotOptions()
        self.dark = False
        self._entries: List[PlotEntry] = []
        self._layout_key = None
        self._plots: Dict[tuple, pg.PlotItem] = {}
        self._curves: Dict[tuple, pg.PlotDataItem] = {}
        self._legends: Dict[tuple, pg.LegendItem] = {}
        self.setBackground(THEMES[False]['background'])

    # -- public ------------------------------------------------------------------------------------------------------

    def set_options(self, options: PlotOptions) -> None:
        self.options = options
        self._redraw()

    def set_entries(self, entries: List[PlotEntry]) -> None:
        self._entries = entries
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
        self._legends.clear()
        self.setBackground(THEMES[self.dark]['background'])
        separate = self.options.mode == 'Separate'
        functions = [f for f in FUNCTIONS if f in self.options.functions]
        columns = len(functions) if separate else max(1, min(self.options.columns, len(functions)))
        foreground = THEMES[self.dark]['foreground']

        first_in_column: Dict[str, pg.PlotItem] = {}
        for index, (row, function) in enumerate(self._rows()):
            title, _, x_label, y_label = FUNCTIONS[function]
            if separate:
                grid_row, grid_col = row, functions.index(function)
            else:
                grid_row, grid_col = divmod(index, columns)
            plot = self.addPlot(row=grid_row, col=grid_col)
            label_style = {'color': foreground}
            plot.setLabel('bottom', x_label, **label_style)
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
            if self.options.legend and not separate and index == 0:
                legend = plot.addLegend(offset=LEGEND_POSITIONS.get(self.options.legend_position, (-10, 10)))
                legend.setLabelTextColor(foreground)
                self._legends[(row, function)] = legend

    # -- drawing -----------------------------------------------------------------------------------------------------

    def _redraw(self) -> None:
        o = self.options
        layout_key = (o.mode, tuple(o.functions), o.columns, o.legend, o.legend_position, o.grid, self.dark,
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
        wanted = set()
        for (row, function), plot in self._plots.items():
            _, x_attr, _, _ = FUNCTIONS[function]
            indices = [row] if separate else list(range(len(self._entries)))
            values = displayed_values(self._entries, o, function, indices)
            log_y = function == 'i' and o.log_iq

            for k in indices:
                entry = self._entries[k]
                x = getattr(entry.result, x_attr)
                y = values[k]
                if log_y:
                    y = np.where(y > 0, y, np.nan)
                if function == 'i' and o.log_q:
                    positive = x > 0
                    x, y = x[positive], y[positive]
                key = (row, function, k)
                wanted.add(key)
                pen = pg.mkPen(entry.color, width=o.line_width)
                style = dict(pen=pen, connect='finite',
                             symbol='o' if o.markers else None, symbolSize=4,
                             symbolPen=None, symbolBrush=entry.color)
                curve = self._curves.get(key)
                if curve is None:
                    curve = plot.plot(x, y, name=entry.label, **style)
                    self._curves[key] = curve
                else:
                    curve.setData(x, y, **style)
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


def displayed_values(entries: List[PlotEntry], options: PlotOptions, function: str, indices: List[int]) -> dict:
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
                # Multiplicative spacing on a logarithmic axis, additive otherwise
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

    legend_locations = {'Top right': 'upper right', 'Top left': 'upper left',
                        'Bottom right': 'lower right', 'Bottom left': 'lower left'}
    with matplotlib.rc_context({'font.size': 9}):
        fig = Figure(figsize=(options.export_panel_width * n_cols, options.export_panel_height * n_rows),
                     constrained_layout=True)
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
                values = displayed_values(entries, options, function, indices)
                for k in indices:
                    entry = entries[k]
                    x, y = getattr(entry.result, x_attr), values[k]
                    if function == 'i' and options.log_q:
                        x, y = x[x > 0], y[x > 0]
                    ax.plot(x, y, color=entry.color.name(), linewidth=options.line_width * 0.8, label=entry.label,
                            marker='o' if options.markers else None, markersize=2)
                if function == 'i':
                    if options.log_iq:
                        ax.set_yscale('log')
                    if options.log_q:
                        ax.set_xscale('log')
                ax.set_xlabel(x_label)
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
