from paradox.NWENV import *
from qtpy.QtWidgets import QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTextEdit
from qtpy.QtGui import QImage, QPixmap, QFont
from qtpy.QtCore import Signal, QSize, QTimer
import os


# load csv file widgets
class ChooseFileInputWidget(IWB, QPushButton):
    
    path_chosen = Signal(str)

    def __init__(self, params):
        IWB.__init__(self, params)
        QPushButton.__init__(self, "Select")

        self.clicked.connect(self.button_clicked)

    def button_clicked(self):
        file_path = QFileDialog.getOpenFileName(self, 'Select CSV file')[0]
        try:
            file_path = os.path.relpath(file_path)
        except ValueError:
            return
        
        self.path_chosen.emit(file_path)

# widgets for the folder selection
class ChooseFolderInputWidget(IWB, QPushButton):
    
    path_chosen = Signal(str)

    def __init__(self, params):
        IWB.__init__(self, params)
        QPushButton.__init__(self, "Select")

        self.clicked.connect(self.button_clicked)

    def button_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Train folder')[0]
        try:
            folder_path = os.path.relpath(folder_path)
        except ValueError:
            return
        
        self.path_chosen.emit(folder_path)


# Exporting the widgets class
export_widgets(
    ChooseFileInputWidget,
    ChooseFolderInputWidget,
)        