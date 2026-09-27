"""
Small reusable widgets for the desktop GUI.
"""

import math
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget,
)


class FloatSlider(QWidget):
    """
    A slider coupled to a spin box, emitting valueChanged(float) while the slider is dragged.
    With log=True the slider position maps logarithmically onto [minimum, maximum] (minimum > 0).
    """

    valueChanged = Signal(float)

    STEPS = 1000

    def __init__(self, label: str, minimum: float, maximum: float, value: float, decimals: int = 3,
                 step: Optional[float] = None, log: bool = False, unit: str = '', label_width: int = 70, parent=None):
        super().__init__(parent)
        self._minimum, self._maximum, self._log = minimum, maximum, log

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.label.setMinimumWidth(label_width)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step if step is not None else (maximum - minimum) / 100)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(80)
        if unit:
            self.spin.setSuffix(f' {unit}')
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)
        self.setValue(value)

    def _to_position(self, value: float) -> int:
        lo, hi = self._minimum, self._maximum
        if self._log:
            fraction = (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            fraction = (value - lo) / (hi - lo) if hi > lo else 0.0
        return int(round(min(max(fraction, 0.0), 1.0) * self.STEPS))

    def _from_position(self, position: int) -> float:
        lo, hi = self._minimum, self._maximum
        fraction = position / self.STEPS
        if self._log:
            return math.exp(math.log(lo) + fraction * (math.log(hi) - math.log(lo)))
        return lo + fraction * (hi - lo)

    def _slider_moved(self, position: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(self._from_position(position))
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _spin_changed(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_position(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_position(self.spin.value()))
        self.slider.blockSignals(False)

    def setRange(self, minimum: float, maximum: float) -> None:
        value = self.value()
        self._minimum, self._maximum = minimum, maximum
        self.spin.setRange(minimum, maximum)
        self.setValue(min(max(value, minimum), maximum))


class ColorButton(QPushButton):
    """
    A button showing a color swatch; clicking opens a color dialog. Emits colorChanged(QColor).
    """

    colorChanged = Signal(QColor)

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self._color = QColor(color)
        self._update_style()
        self.clicked.connect(self._choose)

    def color(self) -> QColor:
        return QColor(self._color)

    def setColor(self, color: QColor) -> None:
        self._color = QColor(color)
        self._update_style()

    def _update_style(self) -> None:
        self.setStyleSheet(f'background-color: {self._color.name()}; border: 1px solid #888; border-radius: 3px;')

    def _choose(self) -> None:
        color = QColorDialog.getColor(self._color, self, 'Curve color')
        if color.isValid():
            self.setColor(color)
            self.colorChanged.emit(color)
