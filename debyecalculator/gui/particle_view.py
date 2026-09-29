"""
3D view of a particle, drawn as depth-sorted, depth-shaded spheres in a 2D pyqtgraph scene.
It needs no OpenGL, so it works on every platform and remote desktop.
Left-drag rotates, the mouse wheel zooms, right-drag pans (pyqtgraph default).
"""

from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from debyecalculator.utility.generate import load_elements_info

# Jmol colours for common elements; others get a colour derived from the atomic number
ELEMENT_COLORS = {
    'H': '#ffffff', 'He': '#d9ffff', 'Li': '#cc80ff', 'Be': '#c2ff00', 'B': '#ffb5b5', 'C': '#909090',
    'N': '#3050f8', 'O': '#ff0d0d', 'F': '#90e050', 'Na': '#ab5cf2', 'Mg': '#8aff00', 'Al': '#bfa6a6',
    'Si': '#f0c8a0', 'P': '#ff8000', 'S': '#ffff30', 'Cl': '#1ff01f', 'K': '#8f40d4', 'Ca': '#3dff00',
    'Ti': '#bfc2c7', 'V': '#a6a6ab', 'Cr': '#8a99c7', 'Mn': '#9c7ac7', 'Fe': '#e06633', 'Co': '#f090a0',
    'Ni': '#50d050', 'Cu': '#c88033', 'Zn': '#7d80b0', 'Ga': '#c28f8f', 'Ge': '#668f8f', 'Se': '#ffa100',
    'Br': '#a62929', 'Sr': '#00ff00', 'Y': '#94ffff', 'Zr': '#94e0e0', 'Nb': '#73c2c9', 'Mo': '#54b5b5',
    'Ru': '#248f8f', 'Rh': '#0a7d8c', 'Pd': '#006985', 'Ag': '#c0c0c0', 'Cd': '#ffd98f', 'In': '#a67573',
    'Sn': '#668080', 'Sb': '#9e63b5', 'Te': '#d47a00', 'I': '#940094', 'Ba': '#00c900', 'La': '#70d4ff',
    'Ce': '#ffffc7', 'W': '#2194d6', 'Ir': '#175487', 'Pt': '#d0d0e0', 'Au': '#ffd123', 'Pb': '#575961',
    'Bi': '#9e4fb5',
}


def element_color(element: str) -> QColor:
    if element in ELEMENT_COLORS:
        return QColor(ELEMENT_COLORS[element])
    color = QColor()
    color.setHsv(hash(element) % 360, 140, 220)
    return color


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


class _RotatingViewBox(pg.ViewBox):
    def __init__(self, view: 'ParticleView'):
        super().__init__(lockAspect=True, enableMenu=False)
        self._view = view

    def mouseDragEvent(self, event, axis=None):
        if event.button() != Qt.LeftButton:
            super().mouseDragEvent(event, axis)
            return
        event.accept()
        delta = event.pos() - event.lastPos()
        self._view.rotate(delta.x(), delta.y())


class ParticleView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._radii_table = {key: value[13] for key, value in load_elements_info().items()}
        self._elements: List[str] = []
        self._xyz = np.zeros((0, 3))
        self._cell_edges = np.zeros((0, 2, 3))
        self._atom_count = 0
        self._rotation = rotation(np.array([1.0, 0.0, 0.0]), -0.35) @ rotation(np.array([0.0, 1.0, 0.0]), 0.5)
        self._background = QColor('#ffffff')
        self._key = None

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        self.info.setMinimumWidth(0)
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        hint = QLabel('Drag to rotate, scroll to zoom, right-drag to pan')
        hint.setEnabled(False)
        hint.setWordWrap(True)
        self.canvas = pg.GraphicsLayoutWidget()
        self.viewbox = _RotatingViewBox(self)
        self.plot = self.canvas.addPlot(viewBox=self.viewbox)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.cell_back = pg.PlotCurveItem(connect='finite')
        self.cell_back.setZValue(-1)
        self.plot.addItem(self.cell_back)
        self.scatter = pg.ScatterPlotItem(pxMode=False)
        self.plot.addItem(self.scatter)
        self.cell_front = pg.PlotCurveItem(connect='finite')
        self.cell_front.setZValue(1)
        self.plot.addItem(self.cell_front)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.info)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(hint)
        self.set_dark(False)

    def set_dark(self, dark: bool) -> None:
        self._background = QColor('#1e1f22' if dark else '#ffffff')
        self.canvas.setBackground(self._background)
        self._draw()

    def set_particle(self, elements: Optional[List[str]], xyz: Optional[np.ndarray], label: str = '') -> None:
        self.set_structure(elements, xyz, label)

    def set_structure(self, elements: Optional[List[str]], xyz: Optional[np.ndarray], label: str = '',
                      lattice: Optional[np.ndarray] = None, atom_count: Optional[int] = None) -> None:
        key = (label, len(elements or []), None if xyz is None else float(np.sum(xyz)),
               None if lattice is None else tuple(np.asarray(lattice).flat))
        new_structure = key != self._key
        self._key = key
        self._elements = list(elements or [])
        self._atom_count = atom_count if atom_count is not None else len(self._elements)
        coordinates = np.zeros((0, 3)) if xyz is None else np.asarray(xyz, dtype=float)
        if lattice is None:
            center = np.mean(coordinates, axis=0) if len(coordinates) else np.zeros(3)
            self._cell_edges = np.zeros((0, 2, 3))
        else:
            lattice = np.asarray(lattice, dtype=float)
            center = np.sum(lattice, axis=0) / 2
            corners = np.asarray([i * lattice[0] + j * lattice[1] + k * lattice[2]
                                  for i in (0, 1) for j in (0, 1) for k in (0, 1)]) - center
            corner = lambda i, j, k: corners[4 * i + 2 * j + k]
            self._cell_edges = np.asarray([
                (corner(0, j, k), corner(1, j, k)) for j in (0, 1) for k in (0, 1)
            ] + [
                (corner(i, 0, k), corner(i, 1, k)) for i in (0, 1) for k in (0, 1)
            ] + [
                (corner(i, j, 0), corner(i, j, 1)) for i in (0, 1) for j in (0, 1)
            ])
        self._xyz = coordinates - center
        unique = sorted(set(self._elements))
        legend = '&nbsp;&nbsp;'.join(f'<span style="color:{element_color(e).name()}">●</span>&nbsp;{e}' for e in unique)
        count = f'{self._atom_count:,} atoms' if self._elements else 'No structure selected'
        self.info.setText(f'<b>{label}</b><br>{count}&nbsp;&nbsp;&nbsp;{legend}')
        self._draw()
        if new_structure:
            self.center_view()

    def set_message(self, label: str, message: str) -> None:
        self._key = None
        self._elements = []
        self._xyz = np.zeros((0, 3))
        self._cell_edges = np.zeros((0, 2, 3))
        self.info.setText(f'<b>{label}</b><br>{message}')
        self._draw()

    def center_view(self) -> None:
        self.viewbox.autoRange(padding=0.05)

    def rotate(self, dx: float, dy: float) -> None:
        self._rotation = (rotation(np.array([0.0, 1.0, 0.0]), dx * 0.01)
                          @ rotation(np.array([1.0, 0.0, 0.0]), dy * 0.01) @ self._rotation)
        self._draw()

    def _draw(self) -> None:
        if len(self._cell_edges):
            projected_edges = self._cell_edges @ self._rotation.T
            front = np.mean(projected_edges[:, :, 2], axis=1) >= 0
            cell_color = '#a6a6a6' if self._background.lightness() < 128 else '#666666'
            pen = pg.mkPen(cell_color, width=1.5)
            self._draw_edges(self.cell_back, projected_edges[~front], pen)
            self._draw_edges(self.cell_front, projected_edges[front], pen)
        else:
            self.cell_back.clear()
            self.cell_front.clear()
        if len(self._elements) == 0:
            self.scatter.clear()
            return
        projected = self._xyz @ self._rotation.T
        order = np.argsort(projected[:, 2])  # back to front
        depth = projected[order, 2]
        span = np.ptp(depth) or 1.0
        shade = 0.45 + 0.55 * (depth - depth.min()) / span  # far atoms fade into the background

        # Depth shading in 16 levels per element, so that brushes can be shared between atoms
        elements = np.asarray(self._elements)[order]
        levels = np.round(shade * 15).astype(int)
        cache = {}
        brushes, pens = [], []
        for element, level in zip(elements, levels):
            key = (element, level)
            if key not in cache:
                t = level / 15
                base, background = element_color(element), self._background
                mixed = QColor(int(background.red() + (base.red() - background.red()) * t),
                               int(background.green() + (base.green() - background.green()) * t),
                               int(background.blue() + (base.blue() - background.blue()) * t))
                cache[key] = (pg.mkBrush(mixed), pg.mkPen(mixed.darker(160), width=0))
            brush, pen = cache[key]
            brushes.append(brush)
            pens.append(pen)
        sizes = np.array([2 * 0.75 * self._radii_table.get(e, 1.0) for e in elements])
        self.scatter.setData(x=projected[order, 0], y=projected[order, 1], size=sizes, brush=brushes, pen=pens)

    @staticmethod
    def _draw_edges(curve: pg.PlotCurveItem, edges: np.ndarray, pen) -> None:
        if not len(edges):
            curve.clear()
            return
        separated = np.concatenate((edges, np.full((len(edges), 1, 3), np.nan)), axis=1)
        points = separated.reshape(-1, 3)
        curve.setData(points[:, 0], points[:, 1], pen=pen, connect='finite')
