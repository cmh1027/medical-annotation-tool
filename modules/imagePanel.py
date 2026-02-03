from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QWheelEvent, QPainter, QColor, QPen, QBrush
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.scale_factor = 1.0
        self.original_pixmap = None
        self.cursor_pos = None
        self.parent = parent
        self.left_dragging = False
        self.right_dragging = False
        self.left_clicked = False
        self.last_cursor_pos = None

    def setPixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def isMouseInsidePixmap(self, event=None, x=None, y=None):
        if event is not None:
            x, y = self.get_image_coordinates(event)
        return 0 <= x < self.original_pixmap.width() and 0 <= y < self.original_pixmap.height()

    def update_scaled_pixmap(self):
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def wheelEvent(self, event: QWheelEvent):
        modifiers = event.modifiers()
        angle_delta = event.angleDelta().y()
        if modifiers == Qt.ControlModifier:
            # Ctrl + Scroll: zoom
            if angle_delta > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor /= 1.1
            self.scale_factor = max(0.1, min(10.0, self.scale_factor))
            self.update_scaled_pixmap()
            event.accept()

        else:
            current_val = self.parent.slice_slider.value()
            max_val = self.parent.slice_slider.maximum()
            min_val = self.parent.slice_slider.minimum()
            step = -1 if angle_delta > 0 else 1
            new_val = current_val + step
            new_val = max(min_val, min(max_val, new_val))
            self.parent.slice_slider.setValue(new_val)
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.right_dragging = False
            x, y = self.get_image_coordinates(event)
            if self.parent.probe_mode:
                self.parent.set_volume_intensity(y, x)
                self.parent.probe_button.toggle_bold()
                self.parent.probe_mode = False
            else:
                mode = self.parent.annotation_mode
                if self.isMouseInsidePixmap(x=x, y=y):
                    if self.parent.brush_mode == 'Auto':
                        if event.modifiers() & Qt.AltModifier:
                            self.parent.remove_annotation_auto(y, x, mode)
                        else:
                            self.parent.annotate_pixel_auto(y, x, mode)
                    elif self.parent.brush_mode == 'Range':
                        if event.modifiers() & Qt.AltModifier:
                            self.parent.remove_annotation_range(y, x)
                        else:
                            self.parent.annotate_pixel_range(y, x, mode)
                        self.left_dragging = True
                    elif self.parent.brush_mode == 'Change':
                        self.parent.annotate_pixel_change(y, x)
                    elif self.parent.brush_mode == 'Line':
                        if self.left_clicked:
                            src = self.get_image_coordinates(self.last_cursor_pos)
                            dst = self.get_image_coordinates(event.pos())
                            self.parent.annotate_pixel_line(src, dst, mode)
                            self.left_clicked = False
                            self.cursor_pos = None
                            self.update()
                        else:
                            self.left_clicked = True
        elif event.button() == Qt.RightButton:
            self.left_clicked = False
            self.left_dragging = False
            self.right_dragging = True
        self.last_cursor_pos = event.pos()

    def mouseMoveEvent(self, event):
        x, y = self.get_image_coordinates(event)
        if self.isMouseInsidePixmap(x=x, y=y):
            if self.parent.brush_mode == "Range":
                self.cursor_pos = event.pos()
                self.update()
                if self.left_dragging:
                    mode = self.parent.annotation_mode
                    if self.isMouseInsidePixmap(x=x, y=y):
                        if event.modifiers() & Qt.AltModifier:
                            self.parent.remove_annotation_range(y, x, drag=True)
                        else:
                            self.parent.annotate_pixel_range(y, x, mode, drag=True)
            elif self.parent.brush_mode == "Line":
                if self.left_clicked:
                    self.cursor_pos = event.pos()
                    self.update()
            
            elif self.parent.brush_mode == "Auto":
                self.cursor_pos = event.pos()
                self.update()

            if self.right_dragging:
                if self.last_cursor_pos:
                    delta = event.pos() - self.last_cursor_pos
                    self.last_cursor_pos = event.pos()
                    h_bar = self.parent.image_area.horizontalScrollBar()
                    v_bar = self.parent.image_area.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())
        else:
            self.left_dragging = False
            self.right_dragging = False
            self.left_clicked = False
            self.cursor_pos = None
            self.last_cursor_pos = None
            self.update()


    def paintEvent(self, event):
        super().paintEvent(event)  # Draw image normally
        if self.parent.probe_mode: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r, g, b = self.parent.annotation_palette[self.parent.annotation_number]
        sf = self.scale_factor
        if self.parent.brush_mode == "Range" and self.cursor_pos:
            color = QColor(r, g, b, 100)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            radius = int(self.parent.brush_size * sf)
            x = self.cursor_pos.x() - radius
            y = self.cursor_pos.y() - radius
            painter.drawEllipse(x, y, 2 * radius, 2 * radius)
        elif self.parent.brush_mode == "Line" and self.cursor_pos:
            color = QColor(r, g, b, 100)
            pen = QPen(color)
            pen.setWidth(int(self.parent.brush_size * sf))  # Set thickness
            painter.setPen(pen)
            start = self.last_cursor_pos
            end = self.cursor_pos
            painter.drawLine(start, end)
        elif self.parent.brush_mode == "Auto" and self.cursor_pos:
            color = QColor(r, g, b, 30)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            radius = int(self.parent.brush_size * sf) * 2
            x = self.cursor_pos.x() - radius
            y = self.cursor_pos.y() - radius
            painter.drawEllipse(x, y, 2 * radius, 2 * radius)
        painter.end()
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_dragging = False
            self.parent.update_slice(None)
        elif event.button() == Qt.RightButton:
            self.right_dragging = False
            self.last_cursor_pos = None

    def get_image_coordinates(self, event):
        # Calculate click position relative to image coordinates
        label_w, label_h = self.width(), self.height()
        pixmap_w = self.original_pixmap.width() * self.scale_factor
        pixmap_h = self.original_pixmap.height() * self.scale_factor
        offset_x = (label_w - pixmap_w) / 2
        offset_y = (label_h - pixmap_h) / 2

        x = (event.x() - offset_x) / self.scale_factor
        y = (event.y() - offset_y) / self.scale_factor
        x, y = int(x), int(y)
        return x, y