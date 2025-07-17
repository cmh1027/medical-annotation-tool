from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QDialog, QFormLayout,
    QLineEdit, QPushButton, QVBoxLayout, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt

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

        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self.accept)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(apply_button)

        self.setLayout(main_layout)

    def get_settings(self):
        return {
            "nifti_default_load": self.nii_default_load_input.text(),
            "numpy_default_load": self.numpy_default_load_input.text(),
            "nifti_default_save": self.nii_default_save_input.text(),
            "numpy_default_save": self.numpy_default_save_input.text(),
        }