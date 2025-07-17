from PyQt5.QtCore import pyqtSignal
from modules.constant import normalFont, boldFont
from PyQt5.QtWidgets import QPushButton
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
