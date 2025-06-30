import sys
import os
import numpy as np
import SimpleITK as sitk

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSlider,
    QPushButton, QHBoxLayout, QFileDialog, QScrollArea,
    QLineEdit, QFormLayout, QMainWindow, QAction
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QColor
from PyQt5.QtWidgets import QShortcut, QSizePolicy
from PyQt5.QtGui import QKeySequence
from qtrangeslider import QRangeSlider
from scipy.ndimage import label

from scipy.ndimage import binary_fill_holes, binary_closing
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_dilation
import nibabel as nib
import pydicom
from glob import glob
import cv2

annotation_palette = [
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

def read_dicom(dcm_path, rescale=False):
    dcm = pydicom.dcmread(dcm_path,force=True)
    if not hasattr(dcm.file_meta,'TransferSyntaxUID'):
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # type: ignore
    required_elements = ['PixelData', 'BitsAllocated', 'Rows', 'Columns',
                     'PixelRepresentation', 'SamplesPerPixel','PhotometricInterpretation',
                        'BitsStored','HighBit']
    missing = [elem for elem in required_elements if elem not in dcm]
    for elem in required_elements:
        if elem not in dcm:
            if elem== 'BitsAllocated':
                dcm.BitsAllocated = 16
            if elem== 'PixelRepresentation':
                dcm.PixelRepresentation = 1
            if elem== 'SamplesPerPixel':
                dcm.SamplesPerPixel = 1
            if elem== 'PhotometricInterpretation':
                dcm.PhotometricInterpretation = 'MONOCHROME2'
            if elem== 'BitsStored':
                dcm.BitsStored = 12
            if elem== 'BitsStored':
                dcm.BitsStored = 11
    if rescale: # For CT
        arr = dcm.pixel_array
        arr_hu = (arr * dcm.RescaleSlope + dcm.RescaleIntercept).astype(np.int16)
        dcm.PixelRepresentation = 1
        dcm.PixelData = arr_hu.tobytes()
    else:
        if dcm.PixelRepresentation == 1:
            arr = dcm.pixel_array
            overflow_threshold = 1 << (dcm.BitsStored-1)
            arr[arr >= overflow_threshold] = 0
            dcm.PixelData = arr.tobytes()
    return dcm

def read_dicoms(dcm_path, 
                slice_first=False, 
                autoflip=False, 
                return_metadata=False,
                metadata_list=[],
                rescale=False):
    dcm_list = sorted(glob(os.path.join(dcm_path, "*.dcm")))
    dcm_array = []
    instance_num = []
    sl_loc = []
    dcm_infos = []
    for sl in range(len(dcm_list)):
        dcm_info = read_dicom(dcm_list[sl], rescale=rescale)
        dcm_infos.append(dcm_info)
        dcm_array.append(dcm_info.pixel_array)
        instance_num.append(dcm_info.InstanceNumber)
        sl_loc.append(np.float16(dcm_info.ImagePositionPatient[2]))

    dcm_array  = np.array(dcm_array)
    dcm_array[np.isnan(dcm_array)] = 0
    sort_idx = np.argsort(instance_num)
    dcm_array = np.array(dcm_array)[sort_idx]
    sl_loc = np.array(sl_loc)[sort_idx]
    dcm_infos = np.array(dcm_infos)[sort_idx]

    flip_idx = sl_loc[0] > sl_loc[-1]
    if flip_idx and autoflip:
        dcm_array = np.flip(dcm_array,0)
    
    if not slice_first:
        dcm_array = np.transpose(dcm_array,(1,2,0))

    pixel_spacing = dcm_infos[0].PixelSpacing
    if hasattr(dcm_infos[0],'SpacingBetweenSlices'):
        slice_spacing = dcm_infos[0].SpacingBetweenSlices
    else:
        slice_spacing = abs(dcm_infos[1].ImagePositionPatient[2] - dcm_infos[0].ImagePositionPatient[2])
    if hasattr(dcm_infos[0], 'SliceThickness'):
        thickness = dcm_infos[0].SliceThickness
    else:
        thickness  = dcm_infos[0].SpacingBetweenSlices

    return_array = dcm_array

    if return_metadata:
        metadata = {
            "pixel_spacing" : (float(pixel_spacing[0]), float(pixel_spacing[1])),
            "slice_spacing" : float(slice_spacing),
            "thickness" : float(thickness),
            "flip_idx": flip_idx
        }
        for tag in metadata_list:
            L = []
            for dcm_info in dcm_infos:
                L.append(getattr(dcm_info, tag, None))
            metadata[tag] = L
        return return_array, metadata
    else:
        return return_array 


def apply_windowing(image_2d, center, width):
    img = image_2d.astype(np.float32)
    min_val = center - (width / 2)
    max_val = center + (width / 2)
    windowed = np.clip((img - min_val) / (max_val - min_val) * 255.0, 0, 255)
    return windowed.astype(np.uint8)


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.scale_factor = 1.0
        self.original_pixmap = None
        self.cursor_pos = (0,0)
        self.parent = parent
        self.left_dragging = False
        self.right_dragging = False
        self.last_drag_pos = None

    def setPixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

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
            # Shift + Scroll: change slice
            current_val = self.parent.slider.value()
            max_val = self.parent.slider.maximum()
            min_val = self.parent.slider.minimum()
            step = -1 if angle_delta > 0 else 1
            new_val = current_val + step
            new_val = max(min_val, min(max_val, new_val))
            self.parent.slider.setValue(new_val)
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_dragging = True
            x, y = self.get_image_coordinates(event)
            mode = self.parent.annotation_mode
            if 0 <= x < self.original_pixmap.width() and 0 <= y < self.original_pixmap.height():
                if self.parent.brush_mode == 'Auto':
                    if event.modifiers() & Qt.AltModifier:
                        self.parent.remove_annotation_auto(y, x, mode)
                    else:
                        self.parent.annotate_pixel_auto(y, x, mode)
                else:
                    if event.modifiers() & Qt.AltModifier:
                        self.parent.remove_annotation_range(y, x)
                    else:
                        self.parent.annotate_pixel_range(y, x)
                        
        elif event.button() == Qt.RightButton:
            self.right_dragging = True
            self.last_drag_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.parent.brush_mode == "Range":
            self.cursor_pos = event.pos()
            self.update()
            if self.left_dragging:
                x, y = self.get_image_coordinates(event)
                if 0 <= x < self.original_pixmap.width() and 0 <= y < self.original_pixmap.height():
                    if event.modifiers() & Qt.AltModifier:
                        self.parent.remove_annotation_range(y, x)
                    else:
                        self.parent.annotate_pixel_range(y, x)

        if self.right_dragging:
            if self.last_drag_pos:
                delta = event.pos() - self.last_drag_pos
                self.last_drag_pos = event.pos()
                h_bar = self.parent.image_area.horizontalScrollBar()
                v_bar = self.parent.image_area.verticalScrollBar()
                h_bar.setValue(h_bar.value() - delta.x())
                v_bar.setValue(v_bar.value() - delta.y())

    def paintEvent(self, event):
        super().paintEvent(event)  # Draw image normally
        if self.parent.brush_mode == "Range" and self.cursor_pos:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Semi-transparent red circle
            r, g, b = annotation_palette[self.parent.annotation_number]
            brush = QColor(r, g, b, 100)  # RGBA
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)

            radius = int(self.parent.brush_size * self.scale_factor)
            x = self.cursor_pos.x() - radius
            y = self.cursor_pos.y() - radius
            painter.drawEllipse(x, y, 2 * radius, 2 * radius)
            painter.end()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_dragging = False
        elif event.button() == Qt.RightButton:
            self.right_dragging = False
            self.last_drag_pos = None

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


class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume = np.random.randint(low=0, high=1000, size=(20, 200, 200), dtype=np.int16)
        self.annotation_map = np.zeros_like(self.volume, dtype=np.uint8)
        self.tolerance = 0.15
        self.annotation_mode = '-'
        self.brush_mode = 'Auto'
        self.window_center = 3000
        self.window_width = 3000
        self.intensity_min = 100
        self.intensity_max = 300
        self.flip_idx = False
        self.annotation_visible = True
        self.title = "DICOM Annotation tool"
        self.last_dirname = os.getcwd()
        self.last_basename = os.getcwd()
        self.annotation_number = 1
        self.initialize()
        self.refresh()

    def refresh(self):
        self.scale_factor = 10
        self.slice_index = self.volume.shape[0] // 2
        self.annotation_history = []
        self.annotation_map = np.zeros_like(self.volume, dtype=np.uint8)
        self.annotation_visible = True
        self.last_saved_name = None
        self.setWindowTitle(self.title)

        self.slider.setMinimum(0)
        self.slider.setMaximum(self.volume.shape[0] - 1)
        self.slider.setValue(self.slice_index)

        self.center_slider.setMinimum(0)
        self.center_slider.setMaximum(6000)
        self.center_slider.setValue(int(self.window_center))

        self.width_slider.setMinimum(0)
        self.width_slider.setMaximum(6000)
        self.width_slider.setValue(int(self.window_width))

    def initialize(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()  # Main horizontal layout
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # === Left: Controls (Vertical stack) ===
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignTop)

        # Slice slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(200)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_slice)
        controls_layout.addWidget(QLabel("Slice"))
        controls_layout.addWidget(self.slider)

        # Window Center
        self.center_slider = QSlider(Qt.Horizontal)
        self.center_slider.setMinimum(0)
        self.center_slider.setMaximum(6000)
        self.center_slider.setValue(int(self.window_center))
        self.center_slider.valueChanged.connect(self.update_windowing_slider)
        controls_layout.addWidget(QLabel("Center"))
        controls_layout.addWidget(self.center_slider)

        # Window Width
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setMinimum(0)
        self.width_slider.setMaximum(6000)
        self.width_slider.setValue(int(self.window_width))
        self.width_slider.valueChanged.connect(self.update_windowing_slider)
        controls_layout.addWidget(QLabel("Width"))
        controls_layout.addWidget(self.width_slider)

        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(50)
        self.tolerance_slider.setValue(int(self.tolerance * 100))
        self.tolerance_slider.valueChanged.connect(self.update_tolerance_from_slider)
        self.tolerance_label = QLabel(f"Tolerance: {self.tolerance:.2f}")
        controls_layout.addWidget(self.tolerance_label)
        controls_layout.addWidget(self.tolerance_slider)

        self.intensity_slider = QRangeSlider()
        self.intensity_slider.setOrientation(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(2000)
        self.intensity_slider.setValue((self.intensity_min, self.intensity_max))  # Set initial lower and upper bounds
        self.intensity_slider.valueChanged.connect(self.update_intensity_range)
        self.intensity_slider.setTracking(True)
        self.intensity_label = QLabel(f"Intensity: {(self.intensity_min, self.intensity_max)}")
        controls_layout.addWidget(self.intensity_label)
        controls_layout.addWidget(self.intensity_slider)
        self.intensity_label.setVisible(False)
        self.intensity_slider.setVisible(False)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(50)
        self.size_slider.setValue(10)
        self.size_slider.valueChanged.connect(self.update_size_label)
        self.size_label = QLabel(f"RoI Size: {self.size_slider.value()}")
        self.brush_size = self.size_slider.value()
        controls_layout.addWidget(self.size_label)
        controls_layout.addWidget(self.size_slider)

        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(1)
        self.color_slider.setMaximum(9)
        self.tolerance_slider.setValue(self.annotation_number)
        self.color_slider.valueChanged.connect(self.update_annotation_number)
        self.color_label = QLabel(f"Color ■")
        self.color_label.setStyleSheet(f"color: rgb{annotation_palette[self.annotation_number]};")  # Light blue
        controls_layout.addWidget(self.color_label)
        controls_layout.addWidget(self.color_slider)

        # Annotation Mode Button
        self.annotation_mode_button = QPushButton("-")
        self.annotation_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.annotation_mode_button.setShortcut("Tab")
        self.brush_mode_button = QPushButton("Auto")
        self.brush_mode_button.clicked.connect(self.toggle_brush_mode)
        mode_layout = QHBoxLayout()
        mode_layout.setAlignment(Qt.AlignLeft)
        mode_layout.addWidget(QLabel("Mode"))
        mode_layout.addWidget(self.annotation_mode_button)
        mode_layout.addWidget(QLabel("Brush"))
        mode_layout.addWidget(self.brush_mode_button)
        controls_layout.addLayout(mode_layout)

        # Wrap controls in a QWidget
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)

        # === Right: Image view ===
        self.image_label = ImageLabel(self)
        self.image_label.setMouseTracking(True)
        self.image_area = QScrollArea()
        self.image_area.setWidgetResizable(True)
        self.image_area.setWidget(self.image_label)
        self.image_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        

        # === Add to main layout ===
        main_layout.addWidget(controls_widget, stretch=1)
        main_layout.addWidget(self.image_area, stretch=4)  # Let image view expand
        # Keyboard shortcuts
        QShortcut(QKeySequence("1"), self).activated.connect(self.decrease_tolerance)
        QShortcut(QKeySequence("2"), self).activated.connect(self.increase_tolerance)
        QShortcut(QKeySequence("3"), self).activated.connect(self.decrease_brush)
        QShortcut(QKeySequence("4"), self).activated.connect(self.increase_brush)
        QShortcut(QKeySequence("`"), self).activated.connect(self.toggle_annotation_visibility)
        QShortcut(QKeySequence("a"), self).activated.connect(self.change_annotation_mode_left)
        QShortcut(QKeySequence("d"), self).activated.connect(self.change_annotation_mode_right)
        QShortcut(QKeySequence("w"), self).activated.connect(self.change_annotation_mode_all)
        QShortcut(QKeySequence("s"), self).activated.connect(self.change_annotation_mode_one)

        # Menu bar setup
        self.create_menu()
        self.file_mode = "dicom"

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+Q")
        open_action.triggered.connect(self.open)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_annotation)

        load_action = QAction("Load", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.load_annotation)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_annotation)

        save_action_new = QAction("Save as", self)
        save_action_new.setShortcut("Ctrl+A")
        save_action_new.triggered.connect(self.save_annotation_newname)

        file_menu.addAction(open_action)
        file_menu.addAction(undo_action)
        file_menu.addAction(load_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_action_new)

    def open(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Select DICOM Series Folder", self.last_dirname
        )
        self.dicom_folder = folder
        try:
            if folder:
                self.title = os.path.basename(folder)
                if len(list(filter(lambda k:".npy" in k, os.listdir(folder)))):
                    self.file_mode = "numpy"
                    self.volume = np.transpose(np.load(os.path.join(folder, "image.npy")), (2,0,1))
                    self.volume -= np.min(self.volume)
                    lower = np.percentile(self.volume, 0.5)
                    upper = np.percentile(self.volume, 99.5)
                    self.window_center = (upper + lower) / 2
                    self.window_width = (upper - lower)
                    self.flip_idx = 0
                    self.refresh()
                    self.load_annotation(path=os.path.join(folder, "label.npy"))
                else:
                    self.file_mode = "dicom"
                    volume, metadata = read_dicoms(folder, return_metadata=True, metadata_list=["WindowCenter", "WindowWidth"], slice_first=True, autoflip=True)
                    N = volume.shape[0]
                    self.volume = volume
                    self.window_center = metadata['WindowCenter'][N//2]
                    self.window_width = metadata['WindowWidth'][N//2]
                    try:
                        self.window_center = self.window_center[0]
                        self.window_width = self.window_width[0]
                    except:
                        pass
                    self.flip_idx = metadata['flip_idx']
                    self.refresh()
                    nii_list = glob(os.path.join(folder, "*.nii"))
                    if len(nii_list) > 0:
                        self.load_annotation(path=nii_list[0])
                self.last_dirname = os.path.dirname(folder)
                self.last_basename = folder
        except Exception as e:
            print("Failed to load file")
           
            
    def update_tolerance_from_slider(self, value):
        self.tolerance = value / 100.0
        self.tolerance_label.setText(f"Tolerance: {(self.tolerance):.2f}")

    def increase_tolerance(self):
        val = min(50, self.tolerance_slider.value() + 1)
        self.tolerance_slider.setValue(val)  # This triggers update_tolerance_from_slider
        self.update_tolerance()
        self.tolerance_label.setText(f"Tolerance: {(val / 100.0):.2f}")

    def decrease_tolerance(self):
        val = max(0, self.tolerance_slider.value() - 1)
        self.tolerance_slider.setValue(val)  # This triggers update_tolerance_from_slider
        self.update_tolerance()
        self.tolerance_label.setText(f"Tolerance: {(val / 100.0):.2f}")

    def decrease_brush(self):
        current = self.size_slider.value()
        if current > self.size_slider.minimum():
            self.size_slider.setValue(current - 1)
        size = self.size_slider.value()
        self.brush_size = size
        self.update_size_label(size)

    def increase_brush(self):
        current = self.size_slider.value()
        if current < self.size_slider.maximum():
            self.size_slider.setValue(current + 1)
        size = self.size_slider.value()
        self.brush_size = size
        self.update_size_label(size)

    def update_size_label(self, value):
        self.size_label.setText(f"RoI Size: {value}")
        self.brush_size = value

    def update_annotation_number(self, value):
        self.annotation_number = value
        self.color_label.setStyleSheet(f"color: rgb{annotation_palette[self.annotation_number]};")  # Light blue

    def update_intensity_range(self, value):
        low, high = value
        self.intensity_label.setText(f"Intensity: {(low, high)}")
        self.intensity_min, self.intensity_max = value

    def load_annotation(self, path=None):
        try:
            if path is None:
                path, _ = QFileDialog.getOpenFileName(self, "Load Annotation", self.dicom_folder, "Nifti Files (*.nii)")
            if path:
                if ".npy" in path:
                    loaded = np.transpose(np.load(path), (2,0,1))
                else:
                    loaded = np.transpose(np.array(nib.load(path).dataobj), (2,0,1))
                if self.flip_idx:
                    loaded = np.flip(loaded, 0)
                if loaded.shape == self.annotation_map.shape:
                    self.annotation_map = loaded
                    self.update_slice(self.slice_index)
                    self.last_saved_name = path
                else:
                    print(f"Shape mismatch ({loaded.shape, self.annotation_map.shape}). Annotation not loaded.")
        except Exception as e:
            print(f"Failed to load annotation: {e}") 

    def toggle_annotation_visibility(self):
        self.annotation_visible = not self.annotation_visible
        self.update_slice(self.slice_index)

    def update_tolerance(self):
        val = self.tolerance_slider.value()
        self.tolerance = val / 100.0

    def change_annotation_mode_left(self):
        self.annotation_mode = 'left'
        self.annotation_mode_button.setText('←')

    def change_annotation_mode_right(self):
        self.annotation_mode = 'right'
        self.annotation_mode_button.setText('→')

    def change_annotation_mode_all(self):
        self.annotation_mode = 'all'
        self.annotation_mode_button.setText('↔')

    def change_annotation_mode_one(self):
        self.annotation_mode = '-'
        self.annotation_mode_button.setText('-')

    def toggle_annotation_mode(self):
        if self.annotation_mode == '-':
            self.annotation_mode = 'left'
            self.annotation_mode_button.setText('←')
        elif self.annotation_mode == 'left':
            self.annotation_mode = 'right'
            self.annotation_mode_button.setText('→')
        elif self.annotation_mode == 'right':
            self.annotation_mode = 'all'
            self.annotation_mode_button.setText('↔')
        elif self.annotation_mode == 'all':
            self.annotation_mode = '-'
            self.annotation_mode_button.setText('-')
        self.annotation_mode_button.setShortcut("Tab")

    def toggle_brush_mode(self):
        if self.brush_mode == 'Auto':
            self.brush_mode = 'Range'
            self.brush_mode_button.setText('Range')
            self.intensity_label.setVisible(True)
            self.intensity_slider.setVisible(True)
            self.tolerance_label.setVisible(False)
            self.tolerance_slider.setVisible(False)

        elif self.brush_mode == 'Range':
            self.brush_mode = 'Auto'
            self.brush_mode_button.setText('Auto')
            self.intensity_label.setVisible(False)
            self.intensity_slider.setVisible(False)
            self.tolerance_label.setVisible(True)
            self.tolerance_slider.setVisible(True)

    def update_windowing_slider(self):
        # Update window center and width from slider values
        self.window_center = self.center_slider.value()
        self.window_width = self.width_slider.value()
        self.update_slice(self.slice_index)  # Refresh current slice with new windowing


    def update_slice(self, index, alpha=0.5):
        if self.volume is None: return
        self.slice_index = index
        raw_slice = self.volume[self.slice_index]
        windowed = apply_windowing(raw_slice, self.window_center, self.window_width)

        # Convert grayscale to RGB
        rgb = np.stack([windowed] * 3, axis=-1).astype(np.float32)

        if self.annotation_visible:
            ann = self.annotation_map[self.slice_index]
            for k in np.unique(ann):
                if k == 0: continue
                overlay = np.zeros_like(rgb)
                for channel in [0, 1, 2]:
                    overlay[..., channel] = annotation_palette[k][channel]  # Red channel
                mask = (ann == k)
                rgb[mask] = (1 - alpha) * rgb[mask] + alpha * overlay[mask]

        # Convert to uint8 for QImage
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.image_label.update_scaled_pixmap()


    def update_windowing(self):
        try:
            center = float(self.center_input.text())
            width = float(self.width_input.text())
            if width <= 0:
                raise ValueError("Width must be > 0")
            self.window_center = center
            self.window_width = width
            self.update_slice(self.slice_index)
        except ValueError as e:
            print("Invalid windowing values:", e)

    def go_to_previous(self):
        if self.slice_index > 0:
            self.slider.setValue(self.slice_index - 1)

    def go_to_next(self):
        if self.slice_index < self.volume.shape[0] - 1:
            self.slider.setValue(self.slice_index + 1)

    def tolerance_slider_changed(self, value):
        self.tolerance = value / 100.0  

    def annotate_pixel_range(self, y, x):
        z = self.slice_index
        R = self.brush_size

        y_min = max(0, y - R)
        y_max = min(self.volume.shape[1], y + R + 1)
        x_min = max(0, x - R)
        x_max = min(self.volume.shape[2], x + R + 1)

        # Extract ROI from volume
        roi = self.volume[z, y_min:y_max, x_min:x_max]

        # Create circular mask
        H, W = roi.shape
        Y, X = np.ogrid[0:H, 0:W]
        center_y, center_x = y - y_min, x - x_min
        dist_sq = (Y - center_y)**2 + (X - center_x)**2
        circular_mask = dist_sq <= R**2

        # Intensity threshold mask
        intensity_mask = (roi >= self.intensity_min) & (roi <= self.intensity_max)

        # Final mask: inside circle and within intensity range
        mask = circular_mask & intensity_mask

        # Save undo information and apply annotation
        undo_entry = []
        local_indices = np.argwhere(mask)
        for dy_, dx_ in local_indices:
            global_y = y_min + dy_
            global_x = x_min + dx_
            old_val = self.annotation_map[z, global_y, global_x]
            undo_entry.append((z, global_y, global_x, old_val))
            self.annotation_map[z, global_y, global_x] = self.annotation_number

        self.annotation_history.append(undo_entry)
        self.update_slice(z)

    def annotate_pixel_auto(self, y, x, direction):
        z = self.slice_index
    
        roi_radius = self.brush_size
        y_min = max(0, y - roi_radius)
        y_max = min(self.volume.shape[1], y + roi_radius + 1)
        x_min = max(0, x - roi_radius)
        x_max = min(self.volume.shape[2], x + roi_radius + 1)

        max_index_flat = np.argmin(np.where(self.annotation_map[z, y_min:y_max, x_min:x_max], 1e+8, self.volume[z, y_min:y_max, x_min:x_max]))
        local_y, local_x = np.unravel_index(max_index_flat, (y_max-y_min, x_max-x_min))
        y = y_min + local_y
        x = x_min + local_x
        intensity = self.volume[z, y, x]
        y_min = max(0, y - roi_radius)
        y_max = min(self.volume.shape[1], y + roi_radius + 1)
        x_min = max(0, x - roi_radius)
        x_max = min(self.volume.shape[2], x + roi_radius + 1)

        struct = generate_binary_structure(2, 2)  # 2D, 8-connectivity

        # Helper function: get labeled mask for one slice ROI
        def get_labeled_mask(slice_idx, upper_bound):
            local_slice = self.volume[slice_idx, y_min:y_max, x_min:x_max]
            binary_mask = local_slice <= upper_bound
            labeled, num_features = label(binary_mask, structure=struct)
            return labeled, num_features

        # Get initial labeled mask on current slice

        labeled, num_features = get_labeled_mask(z, intensity * (1 + self.tolerance))

        local_y = y - y_min
        local_x = x - x_min

        component_label = labeled[local_y, local_x]
        if component_label == 0:
            print(f"No connected component (Intensity : {intensity})")
            return

        # Initialize mask accumulator with zeros (for all slices in ROI)
        accum_mask = np.zeros((min(self.volume.shape[0], z + roi_radius + 1) - max(0, z - roi_radius),
                            y_max - y_min, x_max - x_min), dtype=bool)

        z_start = max(0, z - roi_radius)
        z_end = min(self.volume.shape[0], z + roi_radius + 1)

        # Set initial mask in current slice (relative index)
        accum_mask[z-z_start] = (labeled == component_label)
        # Function to check if two masks overlap (touch)
        def masks_touch(mask1, mask2):
            dilated = binary_dilation(mask1, structure=struct)
            return (dilated & mask2).sum()


        if direction == 'left' or direction == 'all':
            for zi in range(z - 1, z_start - 1, -1):
                tol = self.tolerance
                while tol > 0:
                    labeled_up, num_features_up = get_labeled_mask(zi, intensity * (1 + tol))
                    slice_idx = zi - z_start
                    found = False
                    before = accum_mask[slice_idx+1].sum()
                    accum_new = np.zeros_like(accum_mask[slice_idx])
                    max_overlap = 0
                    for lbl in range(1, num_features_up + 1):
                        comp_mask = (labeled_up == lbl)
                        overlap = masks_touch(comp_mask, accum_mask[slice_idx + 1])
                        if max_overlap < overlap:
                            found = True
                            max_overlap = overlap
                            accum_new = comp_mask
                    after = accum_new.sum()
                    if before * 3 < after:
                        tol = tol - 0.01
                    else:
                        accum_mask[slice_idx] = accum_new
                        break
                if not found:
                    break  # stop if no connected component touches

        if direction == 'right' or direction == 'all':
            for zi in range(z + 1, z_end):
                tol = self.tolerance
                found = False
                while tol > 0:
                    labeled_down, num_features_down = get_labeled_mask(zi, intensity * (1 + tol))
                    slice_idx = zi - z_start
                    found = False
                    before = accum_mask[slice_idx-1].sum()
                    accum_new = np.zeros_like(accum_mask[slice_idx])
                    max_overlap = 0
                    for lbl in range(1, num_features_down + 1):
                        comp_mask = (labeled_down == lbl)
                        overlap = masks_touch(comp_mask, accum_mask[slice_idx - 1])
                        if max_overlap < overlap:
                            found = True
                            max_overlap = overlap
                            accum_new = comp_mask
                    after = accum_new.sum()
                    if before * 3 < after:
                        tol = tol - 0.01
                    else:
                        accum_mask[slice_idx] = accum_new
                        break
                if not found:
                    break

        # Apply fill holes and closing to each slice mask in accum_mask
        for si in range(accum_mask.shape[0]):
            accum_mask[si] = binary_fill_holes(accum_mask[si])
            accum_mask[si] = binary_closing(accum_mask[si], structure=struct)

        # Save old values for undo and update annotation_map globally
        undo_entry = []
        for si, slice_mask in enumerate(accum_mask):
            global_z = z_start + si
            local_indices = np.argwhere(slice_mask)
            for dy_, dx_ in local_indices:
                global_y = y_min + dy_
                global_x = x_min + dx_
                old_val = self.annotation_map[global_z, global_y, global_x]
                undo_entry.append((global_z, global_y, global_x, old_val))
                self.annotation_map[global_z, global_y, global_x] = self.annotation_number

        self.annotation_history.append(undo_entry)
        self.update_slice(z)

    def remove_annotation_range(self, y, x):
        z = self.slice_index
        R = self.brush_size

        # Define bounds of the circular region
        y_min = max(0, y - R)
        y_max = min(self.volume.shape[1], y + R + 1)
        x_min = max(0, x - R)
        x_max = min(self.volume.shape[2], x + R + 1)

        # Extract ROIs
        ann_slice = self.annotation_map[z, y_min:y_max, x_min:x_max]

        # Compute circular mask
        H, W = ann_slice.shape
        Y, X = np.ogrid[0:H, 0:W]
        center_y, center_x = y - y_min, x - x_min
        dist_sq = (Y - center_y)**2 + (X - center_x)**2
        circular_mask = dist_sq <= R**2

        mask = ((ann_slice == self.annotation_number) & circular_mask)

        # Save undo info and clear annotations
        undo_entry = []
        indices_to_clear = np.argwhere(mask)
        for dy_, dx_ in indices_to_clear:
            global_y = y_min + dy_
            global_x = x_min + dx_
            old_val = self.annotation_map[z, global_y, global_x]
            undo_entry.append((z, global_y, global_x, old_val))
            self.annotation_map[z, global_y, global_x] = 0

        if undo_entry:
            self.annotation_history.append(undo_entry)

        self.update_slice(z)

    def remove_annotation_auto(self, y, x, direction):
        z = self.slice_index
        # Check if clicked voxel is annotated
        if self.annotation_map[z, y, x] == 0:
            return

        # Extract full 3D annotation map
        ann_map = self.annotation_map.copy()

        # Label connected components in annotation map (3D)
        labeled, num_features = label(ann_map)

        # Find label of component at clicked voxel
        comp_label = labeled[z, y, x]
        if comp_label == 0:
            print("No connected component found at clicked voxel.")
            return

        # Find slices to remove annotation in
        if direction == '-':
            z_min = z
            z_max = z + 1
        elif direction=='left':
            z_min = 0
            z_max = z + 1  
        elif direction=='right':
            z_min = z
            z_max = ann_map.shape[0]  
        else:
            z_min = 0
            z_max = ann_map.shape[0]  

        # Create mask for voxels to remove:
        # Only those in the connected component AND in slices in [z_min, z_max)
        mask_remove = np.zeros_like(ann_map, dtype=bool)
        for slice_idx in range(z_min, z_max):
            mask_slice = (labeled[slice_idx] == comp_label) & (self.annotation_map == self.annotation_number)
            mask_remove[slice_idx] = mask_slice

        # Save old annotation values for undo
        undo_entry = []
        indices_to_clear = np.argwhere(mask_remove)
        for (zz, yy, xx) in indices_to_clear:
            old_val = self.annotation_map[zz, yy, xx]
            if old_val != 0:
                undo_entry.append((zz, yy, xx, old_val))
                self.annotation_map[zz, yy, xx] = 0

        if undo_entry:
            self.annotation_history.append(undo_entry)

        self.update_slice(self.slice_index)

    def undo_annotation(self):
        if self.annotation_history:
            last_action = self.annotation_history.pop()
            for z, y, x, old_val in last_action:
                self.annotation_map[z, y, x] = old_val
            self.update_slice(self.slice_index)

    def save_annotation_newname(self):
        anno = self.annotation_map
        if self.flip_idx:
            anno = np.flip(anno, 0)
        if self.file_mode == "dicom":
            save_path = QFileDialog.getSaveFileName(self, caption="Save Annotation", filter="Nifti file (*.nii)", directory=os.path.join(self.last_basename, "annotation.nii"))[0]
            if save_path:
                nib.save(nib.Nifti1Image(np.transpose(anno, (1,2,0)), np.ones((4,4))), save_path)
                self.last_saved_name = save_path
        elif self.file_mode == "numpy":
            save_path = QFileDialog.getSaveFileName(self, caption="Save Annotation", filter="Numpy file (*.npy)", directory=os.path.join(self.last_basename, "label.npy"))[0]
            if save_path:
                np.save(save_path, np.transpose(anno, (1,2,0)))
                self.last_saved_name = save_path
        else:
            raise NotImplementedError

    def save_annotation(self):
        if self.last_saved_name is None:
            self.save_annotation_newname()
        else:
            anno = self.annotation_map
            if self.flip_idx:
                anno = np.flip(anno, 0)
            if self.file_mode == "dicom":
                nib.save(nib.Nifti1Image(np.transpose(anno, (1,2,0)), np.ones((4,4))), self.last_saved_name)
            elif self.file_mode == "numpy":
                np.save(self.last_saved_name, np.transpose(anno, (1,2,0)))
            else:
                raise NotImplementedError
        if self.last_saved_name:
            print(f"{self.last_saved_name} Save complete")

def main():
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.resize(1024, 768)
    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
