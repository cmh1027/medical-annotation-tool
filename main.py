import sys
from PyQt5.QtWidgets import QApplication
from modules.mainWindow import MainWindow
def main():
    app = QApplication(sys.argv)
    viewer = MainWindow()
    viewer.resize(1024, 768)
    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
