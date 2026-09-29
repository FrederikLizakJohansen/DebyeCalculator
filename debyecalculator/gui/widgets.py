"""
Small reusable widgets for the desktop GUI.
"""

import math
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QToolButton, QWidget,
)


class FloatSlider(QWidget):
    """
    A slider coupled to a spin box, emitting valueChanged(float).

    The slider covers [minimum, maximum]; with hard_maximum set, the spin box accepts values up to hard_maximum,
    a typed value beyond the slider range extends it, and a button doubles the slider range.
    With log=True the slider position maps logarithmically onto the range (minimum > 0).
    """

    valueChanged = Signal(float)

    STEPS = 1000

    def __init__(self, label: str, minimum: float, maximum: float, value: float, decimals: int = 3,
                 step: Optional[float] = None, log: bool = False, unit: str = '', label_width: int = 60,
                 hard_maximum: Optional[float] = None, parent=None):
        super().__init__(parent)
        self._minimum, self._maximum, self._log = minimum, maximum, log
        self._default_maximum = maximum
        self._hard_maximum = hard_maximum if hard_maximum is not None else maximum

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.label.setMinimumWidth(label_width)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setMinimumWidth(100)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, self._hard_maximum)
        self.spin.setSingleStep(step if step is not None else (maximum - minimum) / 100)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(95)
        if unit:
            self.spin.setSuffix(f' {unit}')
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.extend_button = None
        if self._hard_maximum > maximum:
            self.extend_button = QToolButton()
            self.extend_button.setText('»')
            self.extend_button.setAutoRaise(True)
            self.extend_button.setToolTip('Extend the slider range (double the maximum).\n'
                                          'Right-click to restore the default range.')
            self.extend_button.clicked.connect(self.extend)
            self.extend_button.setContextMenuPolicy(Qt.CustomContextMenu)
            self.extend_button.customContextMenuRequested.connect(lambda _pos: self.reset_range())
            layout.addWidget(self.extend_button)
            self._update_tooltip()

        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderMoved.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)
        self.setValue(value)

    # -- mapping -----------------------------------------------------------------------------------------------------

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

    # -- signals -----------------------------------------------------------------------------------------------------

    def _slider_moved(self, position: int) -> None:
        # Keeps the spin box in step while dragging when tracking is off
        if not self.slider.hasTracking():
            self.spin.blockSignals(True)
            self.spin.setValue(self._from_position(position))
            self.spin.blockSignals(False)

    def _slider_changed(self, position: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(self._from_position(position))
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _spin_changed(self, value: float) -> None:
        if value > self._maximum:
            self._set_slider_maximum(value)
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_position(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    # -- public ------------------------------------------------------------------------------------------------------

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        if value > self._maximum:
            self._set_slider_maximum(value)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_position(self.spin.value()))
        self.slider.blockSignals(False)

    def setTracking(self, tracking: bool) -> None:
        """
        With tracking off, valueChanged is emitted when the slider is released instead of while it moves.
        """
        self.slider.setTracking(tracking)

    def extend(self) -> None:
        self._set_slider_maximum(min(self._maximum * 2, self._hard_maximum))

    def reset_range(self) -> None:
        self._set_slider_maximum(max(self._default_maximum, min(self.value(), self._hard_maximum)))

    def _set_slider_maximum(self, maximum: float) -> None:
        self._maximum = min(max(maximum, self._minimum), self._hard_maximum)
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_position(self.spin.value()))
        self.slider.blockSignals(False)
        self._update_tooltip()

    def _update_tooltip(self) -> None:
        self.slider.setToolTip(f'Slider range {self._minimum:g}–{self._maximum:g}; type a value up to '
                               f'{self._hard_maximum:g} in the box')
        if self.extend_button is not None:
            self.extend_button.setEnabled(self._maximum < self._hard_maximum)


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
