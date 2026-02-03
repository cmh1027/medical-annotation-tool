from PyQt5.QtCore import pyqtSignal
from modules.constant import normalFont, boldFont
from PyQt5.QtWidgets import QPushButton, QSlider
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt
class QToggleButton(QPushButton):
    clicked_after_bold = pyqtSignal()
    def __init__(self, text="", parent=None, toggle_group=None, toggle_text=None, switch_bold=True):
        super().__init__(text, parent)
        self._bold = False  # initial state
        self.switch_bold = switch_bold
        self.clicked.connect(self.toggle_bold)
        self.toggle_group = None
        if toggle_group is not None:
            toggle_group.append(self)
            self.toggle_group = toggle_group
        self.current_text = text
        self.toggle_text = toggle_text
        if toggle_text is None:
            self.toggle_text = text

    def toggle_bold(self):
        self._bold = not self._bold
        self.current_text, self.toggle_text = self.toggle_text, self.current_text
        self.setText(self.current_text)
        if self.toggle_group is not None:
            for button in self.toggle_group:
                button.setFont(normalFont)
        if self.switch_bold:
            if self._bold:
                self.setFont(boldFont)
            else:
                self.setFont(normalFont)
        self.clicked_after_bold.emit()

class QToggleButtonGroup(list):
    pass

from PyQt5.QtWidgets import QSlider, QStyle, QStyleOptionSlider
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush

class MarkerSlider(QSlider):
    def __init__(self, *args, indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indices = indices or []
        self.marker_color = QColor("#ff4444")
        self.marker_size = 6   # diameter of marker dot

    def setMarkedIndices(self, indices):
        self.indices = indices
        self.update()

    def paintEvent(self, event):
        # Draw the normal slider first (groove, handle, ticks)
        super().paintEvent(event)

        if self.maximum() <= self.minimum():
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self.marker_color))
        p.setPen(Qt.NoPen)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        groove_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        vmin, vmax = self.minimum(), self.maximum()
        vrange = vmax - vmin
        if vrange <= 0:
            return

        handle_length = self.style().pixelMetric(QStyle.PM_SliderLength, opt, self)

        if self.orientation() == Qt.Horizontal:
            pixel_range = max(0, groove_rect.width() - handle_length)
            for idx in self.indices:
                pos = QStyle.sliderPositionFromValue(vmin, vmax, idx, pixel_range)
                x = groove_rect.x() + pos + handle_length // 2
                y = groove_rect.center().y()
                r = self.marker_size // 2
                p.drawEllipse(QPoint(x, y), r, r)

        p.end()
