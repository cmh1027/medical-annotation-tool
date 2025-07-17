import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSlider,
    QPushButton, QHBoxLayout, QFileDialog, QScrollArea,
    QLineEdit, QFormLayout, QMainWindow, QAction, QDialog
)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QColor, QPen, QIcon, QFont
from PyQt5.QtWidgets import QShortcut, QTextEdit, QTreeView, QFileSystemModel, QSplitter, QSizePolicy, QMenu, QAction, QMessageBox
from PyQt5.QtGui import QKeySequence
from qtrangeslider import QRangeSlider
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes, binary_closing
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_dilation
import nibabel as nib
from glob import glob
import cv2
from modules.functions import keep_largest_component, read_dicoms, apply_windowing
from modules.constant import ColorPalette, normalFont, boldFont
from modules.customWidget import QToggleButton, QToggleButtonGroup
from modules.imagePanel import ImageLabel
from modules.dialogue.fileSetting import FileSettingsDialog
from tqdm import tqdm
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("asset/jlk_logo.png"))  # Set your logo here
        self.title = "Medical Annotation tool"
        self.volume = np.random.randint(low=0, high=1000, size=(20, 200, 200), dtype=np.int16)
        self.volume_inverse = None
        self.volume_isInverse = False
        self.annotation_map = np.zeros_like(self.volume, dtype=np.uint8)
        self.tolerance = 0.15
        self.annotation_mode = '-'
        self.brush_mode = 'Auto'
        self.window_center = 3000
        self.window_width = 3000
        self.pixel_spacing = 1.0
        self.slice_spacing = 1.0
        self.slice_thickness = 1.0
        self.intensity_min = 0
        self.intensity_max = 300
        self.intensity_intercept = 0
        self.noise_pixels = 10
        self.probe_mode = False
        self.flip_idx = False
        self.annotation_visible = True
        self.update_volume = True
        self.last_dirname = os.getcwd()
        self.last_basename = os.getcwd()
        self.annotation_number = 1
        self.keep_largest_mode = "None"
        self.nifti_default_load_name = ""
        self.numpy_default_load_name = ""
        self.nifti_default_save_name = "annotation.nii"
        self.numpy_default_save_name = "label.npy"
        self.annotation_palette = ColorPalette(self)
        self.volume_text_cache = []
        self.initialize()
        self.refresh()

    @property
    def slice_count(self):
        return self.volume.shape[0]
    
    @property
    def resolution(self):
        return self.volume.shape[1:]
    
    @property
    def volume_shape(self):
        return self.volume.shape

    def refresh(self):
        self.scale_factor = 10
        self.slice_index = self.slice_count // 2
        self.annotation_history = []
        self.annotation_map = np.zeros_like(self.volume, dtype=np.uint8)
        self.annotation_visible = True
        self.setWindowTitle(self.title)

        self.slider.setMinimum(0)
        self.slider.setMaximum(self.slice_count - 1)
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
        self.slice_widget = QLabel(f"Slice 1/1")
        controls_layout.addWidget(self.slice_widget)
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
        self.intensity_slider.setMaximum(3000)
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
        self.size_slider.valueChanged.connect(self.update_roi_size)
        self.size_label = QLabel(f"RoI Size: {self.size_slider.value()}")
        self.brush_size = self.size_slider.value()
        controls_layout.addWidget(self.size_label)
        controls_layout.addWidget(self.size_slider)

        self.propagate_slider = QSlider(Qt.Horizontal)
        self.propagate_slider.setMinimum(0)
        self.propagate_slider.setMaximum(50)
        self.propagate_slider.setValue(0)
        self.propagate_slider.valueChanged.connect(self.update_propagate_slides)
        self.propagate_label = QLabel(f"Propagating slides: {self.propagate_slider.value()}")
        self.propagete_slides = self.propagate_slider.value()
        controls_layout.addWidget(self.propagate_label)
        controls_layout.addWidget(self.propagate_slider)

        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(1)
        self.color_slider.setMaximum(9)
        self.color_slider.setValue(self.annotation_number)
        self.color_slider.valueChanged.connect(self.update_annotation_number)
        self.color_label = QLabel(f"Color ■")
        self.color_label.setStyleSheet(f"color: rgb{self.annotation_palette[self.annotation_number]};")  # Light blue
        controls_layout.addWidget(self.color_label)
        controls_layout.addWidget(self.color_slider)
        for i in range(1, 10):  # keys 1~9
            shortcut = QShortcut(QKeySequence(f"F{i}"), self)
            shortcut.activated.connect(lambda i=i: self.update_annotation_number(i))  # capture i correctly

        # Annotation Mode Button
        annotation_mode_layout = QHBoxLayout()
        annotation_mode_layout.setAlignment(Qt.AlignLeft)
        self.annotation_mode_button = QPushButton("-")
        self.annotation_mode_button.clicked.connect(self.toggle_annotation_mode)
        self.annotation_mode_button.setShortcut("Tab")
        annotation_mode_layout.addWidget(QLabel("Mode"))
        annotation_mode_layout.addWidget(self.annotation_mode_button)
        controls_layout.addLayout(annotation_mode_layout)

        brush_mode_layout = QHBoxLayout()
        brush_mode_layout.setAlignment(Qt.AlignLeft)
        brush_mode_layout.addWidget(QLabel("Brush"))
        brush_button_group = QToggleButtonGroup()
        self.brush_button_auto = QToggleButton("Auto", toggle_group=brush_button_group)
        self.brush_button_auto.setFont(boldFont)
        self.brush_button_auto.clicked.connect(lambda _: self.toggle_brush_mode("Auto"))
        brush_mode_layout.addWidget(self.brush_button_auto)
        self.brush_button_range = QToggleButton("Range", toggle_group=brush_button_group)
        self.brush_button_range.clicked.connect(lambda _: self.toggle_brush_mode("Range"))
        brush_mode_layout.addWidget(self.brush_button_range)
        self.brush_button_change = QToggleButton("Change", toggle_group=brush_button_group)
        self.brush_button_change.clicked.connect(lambda _: self.toggle_brush_mode("Change"))
        brush_mode_layout.addWidget(self.brush_button_change)
        self.brush_button_line = QToggleButton("Line", toggle_group=brush_button_group)
        self.brush_button_line.clicked.connect(lambda _: self.toggle_brush_mode("Line"))
        brush_mode_layout.addWidget(self.brush_button_line)
        controls_layout.addLayout(brush_mode_layout)

        keep_largest_layout = QHBoxLayout()
        keep_largest_layout.setAlignment(Qt.AlignLeft)
        self.keep_largest_button = QPushButton("None")
        self.keep_largest_button.clicked.connect(self.toggle_keep_largest)
        self.keep_largest_label = QLabel("Keep Largest")
        keep_largest_layout.addWidget(self.keep_largest_label)
        keep_largest_layout.addWidget(self.keep_largest_button)
        self.keep_largest_label.setVisible(False)
        self.keep_largest_button.setVisible(False)
        controls_layout.addLayout(keep_largest_layout)

        misc_layout = QVBoxLayout()

        misc1_layout = QHBoxLayout()
        misc1_layout.setAlignment(Qt.AlignLeft)
        self.probe_button = QToggleButton("Probe")
        self.probe_button.clicked.connect(self.toggle_probe)
        misc1_layout.addWidget(self.probe_button)
        self.inverse_button = QToggleButton("Inverse")
        self.inverse_button.clicked.connect(self.inverse_intensity)
        misc1_layout.addWidget(self.inverse_button)
        

        misc2_layout = QHBoxLayout()
        misc2_layout.setAlignment(Qt.AlignLeft)
        self.noise_label = QLabel(f"Noise {self.noise_pixels} / 100")
        misc2_layout.addWidget(self.noise_label)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(self.noise_pixels)
        self.noise_slider.valueChanged.connect(self.update_noise_from_slider)
        misc2_layout.addWidget(self.noise_slider)
        self.noise_removal_button = QPushButton("Remove")
        self.noise_removal_button.clicked.connect(self.remove_noise)
        misc2_layout.addWidget(self.noise_removal_button)



        misc_layout.addLayout(misc1_layout)
        misc_layout.addLayout(misc2_layout)
        controls_layout.addLayout(misc_layout)
        

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        controls_layout.addWidget(self.log_box)

        self.info_box = QLabel()
        controls_layout.addWidget(self.info_box)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)

        self.image_label = ImageLabel(self)
        self.image_label.setMouseTracking(True)
        self.image_area = QScrollArea()
        self.image_area.setWidgetResizable(True)
        self.image_area.setWidget(self.image_label)
        self.image_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_area.setMinimumWidth(500)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(os.getcwd())  # or set to a specific path

        self.navigator = QTreeView()
        self.navigator.setModel(self.file_model)
        self.navigator.setRootIndex(self.file_model.index(os.getcwd()))
        self.navigator.setColumnHidden(1, True)  # Hide Size
        self.navigator.setColumnHidden(2, True)  # Hide Type
        self.navigator.setColumnHidden(3, True)  # Hide Date Modified
        self.navigator.setContextMenuPolicy(Qt.CustomContextMenu)
        self.navigator.customContextMenuRequested.connect(self.navigator_menu)
        self.navigator.setMinimumWidth(170)
        # Create a horizontal splitter
        splitter = QSplitter(Qt.Horizontal)

        # Add widgets to splitter
        splitter.addWidget(self.navigator)
        splitter.addWidget(controls_widget)
        splitter.addWidget(self.image_area)
        splitter.setSizes([1, 5, 5])  # works like stretch


        main_layout.addWidget(splitter)


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


    def navigator_menu(self, point):
        index = self.navigator.indexAt(point)

        if not index.isValid():
            return
        
        if self.file_model.isDir(index):
            menu = QMenu()
            open_action = QAction("Open", self.navigator)
    
            def open_folder():
                folder = self.file_model.filePath(index)
                self.open(folder=folder)
            
            open_action.triggered.connect(open_folder)
            menu.addAction(open_action)
            menu.exec_(self.navigator.viewport().mapToGlobal(point))
        

    def logging(self, text):
        self.log_box.append("- " + text)

    def setInfo(self, metadata:dict):
        if metadata is not None:
            text = ""
            keys = ["SeriesDescription", "Manufacturer", "pixel_spacing", "slice_spacing", "thickness"]
            for key in keys:
                value = metadata[key]
                if key in ["SeriesDescription", "Manufacturer"]:
                    value = value[0]
                elif key == "pixel_spacing":
                    value = tuple(float('%.1f' % v) for v in value)
                try:
                    value = float(value)
                    text += f"- {key} : {'%.1f' % value}\n"
                except (TypeError, ValueError):
                    text += f"- {key} : {value}\n"
            text += f"- Resolution : {self.resolution}"
            self.info_box.setText(text)

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

        save_action_new = QAction("Save as", self)
        save_action_new.setShortcut("Ctrl+A")
        save_action_new.triggered.connect(self.save_annotation_newname)

        file_setting_action = QAction("Settings", self)
        file_setting_action.triggered.connect(self.file_setting)


        file_menu.addAction(open_action)
        file_menu.addAction(undo_action)
        file_menu.addAction(load_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_action_new)
        file_menu.addSeparator()
        file_menu.addAction(file_setting_action)


        info_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_notice)
        info_menu.addAction(about_action)

    def file_setting(self):
        dialog = FileSettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            self.nifti_default_load_name = settings['nifti_default_load']
            self.numpy_default_load_name = settings['numpy_default_load']
            self.nifti_default_save_name = settings['nifti_default_save']
            self.numpy_default_save_name = settings['numpy_default_save']

    def show_notice(self):
        notice_text = (
            "Copyright © 2025 Minhyuk Choi. All rights reserved.\n\n"
            "[Contact]\n"
            "mhchoi@jlkgroup.com\n"
            "https://github.com/cmh1027/medical-annotation-tool"
        )

        QMessageBox.information(self, "Medical Annotation tool", notice_text)

    def open(self, folder=None):
        if folder is False:
            folder = QFileDialog.getExistingDirectory(
                None, "Select DICOM Series Folder", self.last_dirname
            )
        self.dicom_folder = folder
        try:
            if folder and len(glob(os.path.join(folder, "*.dcm")) + glob(os.path.join(folder, "image.npy"))) > 0:
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
                    if self.numpy_default_load_name == "":
                        load_name = list(filter(lambda k:k != "image.npy" and k.endswith(".npy"), os.listdir(folder)))[0]
                    else:
                        load_name = self.numpy_default_load_name
                    self.load_annotation(path=os.path.join(folder, load_name))
                    self.pixel_spacing = 1.0
                    self.slice_spacing = 1.0
                    self.slice_thickness = 1.0
                    metadata = None
                else:
                    self.file_mode = "dicom"
                    metadata_list = ["WindowCenter", "WindowWidth", "SeriesDescription", "Manufacturer"]
                    volume, metadata = read_dicoms(folder, return_metadata=True, metadata_list=metadata_list, slice_first=True, autoflip=True, rescale=True)
                    self.pixel_spacing = float(metadata['pixel_spacing'][0])
                    self.slice_spacing = float(metadata['slice_spacing'])
                    self.slice_thickness = float(metadata['thickness'])
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
                    if self.nifti_default_load_name == "":
                        load_name = list(filter(lambda k:k.endswith(".nii"), os.listdir(folder)))
                        if len(load_name) > 0:
                            anno_path = os.path.join(folder, load_name[0])
                        else:
                            anno_path = "None"
                    else:
                        anno_path = os.path.join(folder, self.numpy_default_load_name)
                    if os.path.exists(anno_path):
                        self.load_annotation(path=anno_path)
                self.last_dirname = os.path.dirname(folder)
                self.last_basename = folder
                self.intensity_slider.setMaximum(self.volume.max() - self.volume.min())
                self.intensity_intercept = self.volume.min()
                self.update_intensity_range((0, self.volume.max()//3))
                self.volume_inverse = None
                self.setInfo(metadata)
                self.logging(f"Open Series : {folder}")
            else:
                self.logging(f"Series {folder} is empty")
        except Exception as e:
            self.logging("Failed to load file", e)

    def inverse_intensity(self):
        M = self.volume.max()
        if self.volume_inverse is None:
            self.volume_inverse = M - self.volume
        self.volume, self.volume_inverse = self.volume_inverse, self.volume
        self.window_center = M - self.window_center
        self.center_slider.setValue(self.window_center)
        self.update_slice(self.slice_index)
        self.volume_isInverse = not self.volume_isInverse
        if self.volume_isInverse:
            self.inverse_button.setFont(boldFont)
        else:
            self.inverse_button.setFont(normalFont)

    def toggle_probe(self):
        self.probe_mode = not self.probe_mode

    def update_noise_from_slider(self, value):
        self.noise_pixels = value
        self.noise_label.setText(f"Noise {value} / 100")

    def remove_noise(self):
        for k in tqdm(np.unique(self.annotation_map)):
            if k == 0: continue
            mask = (self.annotation_map == k)
            labeled, _ = label(mask)
            for k in tqdm(np.unique(labeled), leave=False):
                if k == 0: continue
                if (labeled==k).sum() <= self.noise_pixels:
                    self.annotation_map[labeled == k] = 0
        self.update_slice(self.slice_index)

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
        self.update_roi_size(size)

    def increase_brush(self):
        current = self.size_slider.value()
        if current < self.size_slider.maximum():
            self.size_slider.setValue(current + 1)
        size = self.size_slider.value()
        self.brush_size = size
        self.update_roi_size(size)

    def update_roi_size(self, value):
        self.size_label.setText(f"RoI Size: {value}")
        self.brush_size = value

    def update_propagate_slides(self, value):
        self.propagate_label.setText(f"Propagating slides: {value}")
        self.propagete_slides = value

    def update_annotation_number(self, value):
        self.annotation_number = value
        self.color_label.setStyleSheet(f"color: rgb{self.annotation_palette[self.annotation_number]};")  # Light blue
        self.color_slider.setValue(self.annotation_number)

    def update_intensity_range(self, value):
        low, high = value
        low = int(low + self.intensity_intercept)
        high = int(high + self.intensity_intercept)
        self.intensity_label.setText(f"Intensity: {(low, high)}")
        self.intensity_min, self.intensity_max = low, high

    def load_annotation(self, path=None):
        try:
            if not path:
                path, _ = QFileDialog.getOpenFileName(self, "Load Annotation", self.dicom_folder, "Nifti Files (*.nii)")
            if path:
                if not os.path.exists(path): return
                if ".npy" in path:
                    loaded = np.transpose(np.load(path), (2,0,1))
                else:
                    loaded = np.transpose(np.array(nib.load(path).dataobj), (2,0,1))
                if self.flip_idx:
                    loaded = np.flip(loaded, 0)
                if loaded.shape == self.volume_shape:
                    self.annotation_map = loaded
                    self.update_slice(self.slice_index)
                    self.logging(f"Open Annotation : {path}")
                else:
                    self.logging(f"Shape mismatch ({loaded.shape, self.volume_shape}). Annotation not loaded.")
        except Exception as e:
            self.logging(f"Failed to load annotation: {e}") 

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

    def toggle_brush_mode(self, mode):
        if mode == 'Range':
            self.brush_mode = 'Range'
            self.intensity_label.setVisible(True)
            self.intensity_slider.setVisible(True)
            self.tolerance_label.setVisible(False)
            self.tolerance_slider.setVisible(False)
            self.propagate_label.setVisible(True)
            self.propagate_slider.setVisible(True)
            self.keep_largest_label.setVisible(True)
            self.keep_largest_button.setVisible(True)
            self.brush_button_auto.setFont(normalFont)
            self.brush_button_range.setFont(boldFont)
            self.brush_button_change.setFont(normalFont)
            self.brush_button_line.setFont(normalFont)

        elif mode == 'Change':
            self.brush_mode = 'Change'
            self.intensity_label.setVisible(False)
            self.intensity_slider.setVisible(False)
            self.tolerance_label.setVisible(False)
            self.tolerance_slider.setVisible(False)
            self.propagate_label.setVisible(False)
            self.propagate_slider.setVisible(False)
            self.keep_largest_label.setVisible(False)
            self.keep_largest_button.setVisible(False)
            self.brush_button_auto.setFont(normalFont)
            self.brush_button_range.setFont(normalFont)
            self.brush_button_change.setFont(boldFont)
            self.brush_button_line.setFont(normalFont)


        elif mode == 'Auto':
            self.brush_mode = 'Auto'
            self.intensity_label.setVisible(False)
            self.intensity_slider.setVisible(False)
            self.tolerance_label.setVisible(True)
            self.tolerance_slider.setVisible(True)
            self.propagate_label.setVisible(True)
            self.propagate_slider.setVisible(True)
            self.keep_largest_label.setVisible(False)
            self.keep_largest_button.setVisible(False)
            self.brush_button_auto.setFont(boldFont)
            self.brush_button_range.setFont(normalFont)
            self.brush_button_change.setFont(normalFont)
            self.brush_button_line.setFont(normalFont)

        elif mode == "Line":
            self.brush_mode = 'Line'
            self.intensity_label.setVisible(True)
            self.intensity_slider.setVisible(True)
            self.tolerance_label.setVisible(False)
            self.tolerance_slider.setVisible(False)
            self.propagate_label.setVisible(True)
            self.propagate_slider.setVisible(True)
            self.keep_largest_label.setVisible(True)
            self.keep_largest_button.setVisible(True)
            self.brush_button_auto.setFont(normalFont)
            self.brush_button_range.setFont(normalFont)
            self.brush_button_change.setFont(normalFont)
            self.brush_button_line.setFont(boldFont)

    def toggle_keep_largest(self):
        if self.keep_largest_mode == "None":
            self.keep_largest_button.setText("3D")
            self.keep_largest_mode = "3D"
        elif self.keep_largest_mode == "3D":
            self.keep_largest_button.setText("2D")
            self.keep_largest_mode = "2D"
        elif self.keep_largest_mode == "2D":
            self.keep_largest_button.setText("None")
            self.keep_largest_mode = "None"

    def update_windowing_slider(self):
        # Update window center and width from slider values
        self.window_center = self.center_slider.value()
        self.window_width = self.width_slider.value()
        self.update_slice(self.slice_index)  # Refresh current slice with new windowing


    def update_slice(self, index, alpha=0.5, no_volume_update=False):
        if index is None:
            index = self.slice_index
        if self.volume is None: return
        self.slice_widget.setText(f"Slice {index} / {self.slice_count}")
        self.slice_index = index
        ann = self.annotation_map[self.slice_index]
        raw_slice = self.volume[self.slice_index]
        windowed = apply_windowing(raw_slice, self.window_center, self.window_width)

        # Convert grayscale to RGB
        rgb = np.stack([windowed] * 3, axis=-1).astype(np.float32)

        if self.annotation_visible:
            for k in np.unique(ann):
                if k == 0: continue
                overlay = np.zeros_like(rgb)
                for channel in [0, 1, 2]:
                    overlay[..., channel] = self.annotation_palette[k][channel]  # Red channel
                mask = (ann == k)
                rgb[mask] = (1 - alpha) * rgb[mask] + alpha * overlay[mask]

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        coef = np.sqrt(self.resolution[0] / 676)
        painter.setFont(QFont("Arial", int(11 * coef)))

        texts = []
        if self.update_volume and not no_volume_update:
            for k in np.unique(ann):
                if k == 0: continue
                r, g, b = self.annotation_palette[k]
                if self.file_mode == "dicom":
                    volume = (self.pixel_spacing ** 2) * self.slice_thickness * (ann == k).sum() / 1000
                    text = (f"Volume: {'%.4f' % volume}mL", QColor(r, g, b))
                else:
                    voxel = (ann == k).sum()
                    text = (f"Volume: {'%.4f' % voxel} voxels", QColor(r, g, b))
                texts.append(text)
            self.volume_text_cache = texts
        else:
            texts = self.volume_text_cache
        y_offset = 15
        for text, color in texts:
            painter.setPen(color)
            painter.drawText(5, y_offset, text)
            y_offset += int(15 * coef)
        painter.end()

        self.image_label.setPixmap(pixmap)
        self.image_label.update_scaled_pixmap()


    def update_windowing(self):
        try:
            center = float(self.center_input.text())
            width = float(self.width_input.text())
            self.window_center = center
            self.window_width = width
            self.update_slice(self.slice_index)
        except ValueError as e:
            self.logging("Invalid windowing values:", e)

    def go_to_previous(self):
        if self.slice_index > 0:
            self.slider.setValue(self.slice_index - 1)

    def go_to_next(self):
        if self.slice_index < self.slice_count - 1:
            self.slider.setValue(self.slice_index + 1)

    def tolerance_slider_changed(self, value):
        self.tolerance = value / 100.0  

    def set_volume_intensity(self, y, x):
        z = self.slice_index
        value = int(self.volume[z, y, x])
        m, M = self.intensity_slider.value()
        m_ = m + self.intensity_intercept
        M_ = M + self.intensity_intercept

        if value < m_:
            new_m = value - self.intensity_intercept
            new_M = M
        else:
            new_m = m
            new_M = value - self.intensity_intercept
        
        self.update_intensity_range((new_m, new_M))
        self.intensity_slider.setValue((new_m, new_M))  

    def annotate_pixel_range(self, y, x, direction, drag=False):
        if drag is False:
            self.annotation_map_backup = self.annotation_map.copy()
        z = self.slice_index
        R = self.brush_size

        if direction == '-':
            z_min = z
            z_max = z + 1
        elif direction=='left':
            z_min = z - self.propagete_slides
            z_max = z + 1  
        elif direction=='right':
            z_min = z
            z_max = z + self.propagete_slides + 1
        else:
            z_min = z - self.propagete_slides
            z_max = z + self.propagete_slides + 1
        


        y_min = max(0, y - R)
        y_max = min(self.volume_shape[1], y + R + 1)
        x_min = max(0, x - R)
        x_max = min(self.volume_shape[2], x + R + 1)

        # Extract ROI from volume
        roi = self.volume[z_min:z_max, y_min:y_max, x_min:x_max]

        # Create circular mask
        _, H, W = roi.shape
        Y, X = np.ogrid[0:H, 0:W]
        center_y, center_x = y - y_min, x - x_min
        dist_sq = (Y - center_y)**2 + (X - center_x)**2
        circular_mask_slice = dist_sq <= R**2
        circular_mask = np.repeat(circular_mask_slice[None], z_max-z_min, axis=0)
        # Intensity threshold mask
        intensity_mask = (roi >= self.intensity_min) & (roi <= self.intensity_max)
        print(self.intensity_min, self.intensity_max, np.unique(roi))
        # Final mask: inside circle and within intensity range
        mask = circular_mask & intensity_mask
        mask = keep_largest_component(mask, self.keep_largest_mode, direction=direction)
            
        # Save undo information and apply annotation
        undo_entry = []
        local_indices = np.argwhere(mask)
        for dz_, dy_, dx_ in local_indices:
            global_z = z_min + dz_
            global_y = y_min + dy_
            global_x = x_min + dx_
            old_val = self.annotation_map_backup[global_z, global_y, global_x]
            undo_entry.append((global_z, global_y, global_x, old_val))
            self.annotation_map[global_z, global_y, global_x] = self.annotation_number
        if drag and len(self.annotation_history) > 0:
            self.annotation_history[-1].extend(undo_entry)
        else:
            self.annotation_history.append(undo_entry)
        self.update_slice(z, no_volume_update=drag)

    def annotate_pixel_change(self, y, x):
        z = self.slice_index
        current_anno = self.annotation_map[z, y, x]
        if current_anno == 0: 
            self.logging("Background has been clicked")
            return
        labeled, _ = label(self.annotation_map == current_anno)
        mask = None
        for k in np.unique(labeled):
            if k == 0: continue
            if (labeled == k)[z, y, x] > 0:
                mask = (labeled == k)
                break

        # Save undo information and apply annotation
        if mask is not None:
            undo_entry = []
            global_indices = np.argwhere(mask)
            for global_z, global_y, global_x in global_indices:
                old_val = self.annotation_map[global_z, global_y, global_x]
                undo_entry.append((global_z, global_y, global_x, old_val))
                self.annotation_map[global_z, global_y, global_x] = self.annotation_number

            self.annotation_history.append(undo_entry)
            self.update_slice(z)
        else:
            self.logging("There is no cluster")

    def annotate_pixel_line(self, src, dst, direction):
        x1, y1 = src
        x2, y2 = dst
        z = self.slice_index

        if direction == '-':
            z_min = z
            z_max = z + 1
        elif direction=='left':
            z_min = z - self.propagete_slides
            z_max = z + 1  
        elif direction=='right':
            z_min = z
            z_max = z + self.propagete_slides + 1
        else:
            z_min = z - self.propagete_slides
            z_max = z + self.propagete_slides + 1


        line_mask_slice = np.zeros_like(self.volume[0])
        cv2.line(line_mask_slice, (x1, y1), (x2, y2), color=1, thickness=self.brush_size)
        line_mask = np.repeat(line_mask_slice[None], z_max-z_min, axis=0)

        roi = self.volume[z_min:z_max]
        intensity_mask = (roi >= self.intensity_min) & (roi <= self.intensity_max)
        mask = line_mask & intensity_mask

        mask = keep_largest_component(mask, self.keep_largest_mode, direction=direction)

        undo_entry = []
        local_indices = np.argwhere(mask)
        for dz_, global_y, global_x in local_indices:
            global_z = z_min + dz_
            old_val = self.annotation_map[global_z, global_y, global_x]
            undo_entry.append((global_z, global_y, global_x, old_val))
            self.annotation_map[global_z, global_y, global_x] = self.annotation_number

        self.annotation_history.append(undo_entry)
        self.update_slice(z)

    def annotate_pixel_auto(self, y, x, direction):
        z = self.slice_index
    
        roi_radius = self.brush_size
        y_min = max(0, y - roi_radius)
        y_max = min(self.volume_shape[1], y + roi_radius + 1)
        x_min = max(0, x - roi_radius)
        x_max = min(self.volume_shape[2], x + roi_radius + 1)

        max_index_flat = np.argmin(np.where(self.annotation_map[z, y_min:y_max, x_min:x_max], 1e+8, self.volume[z, y_min:y_max, x_min:x_max]))
        local_y, local_x = np.unravel_index(max_index_flat, (y_max-y_min, x_max-x_min))
        y = y_min + local_y
        x = x_min + local_x
        intensity = self.volume[z, y, x]
        y_min = max(0, y - roi_radius)
        y_max = min(self.volume_shape[1], y + roi_radius + 1)
        x_min = max(0, x - roi_radius)
        x_max = min(self.volume_shape[2], x + roi_radius + 1)

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
            self.logging(f"No connected component (Intensity : {intensity})")
            return

        # Initialize mask accumulator with zeros (for all slices in ROI)
        accum_mask = np.zeros((min(self.slice_count, z + self.propagete_slides + 1) - max(0, z - self.propagete_slides),
                            y_max - y_min, x_max - x_min), dtype=bool)

        z_start = max(0, z - self.propagete_slides)
        z_end = min(self.slice_count, z + self.propagete_slides + 1)

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

    def remove_annotation_range(self, y, x, drag=False):
        z = self.slice_index
        R = self.brush_size

        # Define bounds of the circular region
        y_min = max(0, y - R)
        y_max = min(self.volume_shape[1], y + R + 1)
        x_min = max(0, x - R)
        x_max = min(self.volume_shape[2], x + R + 1)

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

        if drag and len(self.annotation_history) > 0:
            self.annotation_history[-1].extend(undo_entry)
        else:
            self.annotation_history.append(undo_entry)
        self.update_slice(z, no_volume_update=drag)

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
            self.logging("No connected component found at clicked voxel.")
            return

        # Find slices to remove annotation in
        if direction == '-':
            z_min = z
            z_max = z + 1
        elif direction=='left':
            z_min = z - self.propagete_slides
            z_max = z + 1  
        elif direction=='right':
            z_min = z
            z_max = z + self.propagete_slides + 1
        else:
            z_min = z - self.propagete_slides
            z_max = z + self.propagete_slides + 1

        # Create mask for voxels to remove:
        # Only those in the connected component AND in slices in [z_min, z_max)
        mask_remove = np.zeros_like(ann_map, dtype=bool)
        for slice_idx in range(z_min, z_max):
            mask_slice = (labeled[slice_idx] == comp_label) & (self.annotation_map[slice_idx] == self.annotation_number)
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
            save_path = QFileDialog.getSaveFileName(self, caption="Save Annotation", filter="Nifti file (*.nii)", directory=os.path.join(self.last_basename, self.nifti_default_save_name))[0]
            if save_path:
                nib.save(nib.Nifti1Image(np.transpose(anno, (1,2,0)), np.ones((4,4))), save_path)
        elif self.file_mode == "numpy":
            save_path = QFileDialog.getSaveFileName(self, caption="Save Annotation", filter="Numpy file (*.npy)", directory=os.path.join(self.last_basename, self.numpy_default_save_name))[0]
            if save_path:
                np.save(save_path, np.transpose(anno, (1,2,0)))
        else:
            self.logging("Invalid file mode", self.file_mode)

    def save_annotation(self):
        anno = self.annotation_map
        save_name = None
        if self.flip_idx:
            anno = np.flip(anno, 0)
        if self.file_mode == "dicom":
            save_name = os.path.join(self.last_basename, self.nifti_default_save_name)
            nib.save(nib.Nifti1Image(np.transpose(anno, (1,2,0)), np.ones((4,4))), save_name)
        elif self.file_mode == "numpy":
            save_name = os.path.join(self.last_basename, self.numpy_default_save_name)
            np.save(save_name, np.transpose(anno, (1,2,0)))
        else:
            self.logging("Invalid file mode", self.file_mode)
        if save_name:
            self.logging(f"Save complete : {save_name}")
