from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QDialog, QFormLayout,
    QLineEdit, QPushButton, QVBoxLayout, QLabel, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt
from qtrangeslider import QRangeSlider
class FileSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Settings")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.nii_default_load_input = QLineEdit(self)
        self.numpy_default_load_input = QLineEdit(self)
        self.nii_default_save_input = QLineEdit(self)
        self.numpy_default_save_input = QLineEdit(self)

        self.nii_default_load_input.setText(parent.nifti_default_load_name)
        self.numpy_default_load_input.setText(parent.numpy_default_load_name)
        self.nii_default_save_input.setText(parent.nifti_default_save_name)
        self.numpy_default_save_input.setText(parent.numpy_default_save_name)

        layout = QFormLayout()
        layout.addWidget(QLabel("Default load annotation names"))
        layout.addRow("Nifti:", self.nii_default_load_input)
        layout.addRow("NumPy:", self.numpy_default_load_input)

        layout.addWidget(QLabel("Default save annotation names"))
        layout.addRow("Nifti:", self.nii_default_save_input)
        layout.addRow("NumPy:", self.numpy_default_save_input)

        intensity_layout = QHBoxLayout()
        intensity_min = parent.default_intensity_min
        intensity_max = parent.default_intensity_max
        self.intensity_slider = QRangeSlider()
        self.intensity_slider.setOrientation(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue((intensity_min, intensity_max))  # Set initial lower and upper bounds
        self.intensity_slider.valueChanged.connect(self.update_intensity_range)
        self.intensity_slider.setTracking(True)
        self.intensity_label = QLabel(f"Intensity: {(intensity_min, intensity_max)}")
        intensity_layout.addWidget(self.intensity_label)
        intensity_layout.addWidget(self.intensity_slider)

        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self.accept)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(intensity_layout)
        main_layout.addWidget(apply_button)

        self.setLayout(main_layout)

    def update_intensity_range(self, value):
        low, high = value
        self.intensity_label.setText(f"Intensity: {(low, high)}")

    def get_settings(self):
        return {
            "nifti_default_load": self.nii_default_load_input.text(),
            "numpy_default_load": self.numpy_default_load_input.text(),
            "nifti_default_save": self.nii_default_save_input.text(),
            "numpy_default_save": self.numpy_default_save_input.text(),
            "default_intensity":self.intensity_slider.value()
        }