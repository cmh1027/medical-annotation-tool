from PyQt5.QtGui import QFont

class ColorPalette(list):
    def __init__(self, parent):
        self._annotation_palette = [
            (0, 0, 0), # dummy
            (228, 26, 28),    # Red
            (55, 126, 184),   # Blue
            (77, 175, 74),    # Green
            (255, 127, 0),    # Orange
            (152, 78, 163),   # Purple
            (166, 86, 40),    # Brown
            (255, 0, 144),    # Pink
            (0, 206, 209),    # Cyan
            (255, 255, 51),   # Yellow
        ]
        self.out_of_boundary = (127, 127, 127)
        self.parent = parent
        self.warning = False
    
    def __getitem__(self, index):
        if index < len(self._annotation_palette):
            return self._annotation_palette[index]
        else:
            if not self.warning:
                try:
                    self.parent.logging(f"Warning : Annotation number > 9 has been detected.\n Their color will be identical as RGB {self.out_of_boundary}")
                    self.warning = True
                except AttributeError:
                    pass
            return self.out_of_boundary

normalFont = QFont()
boldFont = QFont()
boldFont.setBold(True)  