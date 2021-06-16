# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:51:19 2021

@author: vajra
"""

import os
import sys

from PIL import Image, ImageQt
import numpy as np
from functools import partial
from compute_fMap import compute_fMap

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.data_handlers import grab_position_data
from core.Tint_Matlab import speed2D
from core.ProcessingFunctions import (speed_bins)
from core.WorkerSignals import WorkerSignals

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import (QGridLayout, QWidget, QListWidget, QListWidgetItem,
                             QAbstractItemView, QLabel, QApplication, QPushButton,
                             QFileDialog, QHBoxLayout, QVBoxLayout, QSlider, QErrorMessage,
                             QMessageBox, QDesktopWidget, QComboBox, QLineEdit, QGraphicsPixmapItem, 
                             QGraphicsScene, QGraphicsView, QProgressBar)
      
# =========================================================================== #

class frequencyPlotWindow(QWidget):
    
    def __init__(self):
        
        QWidget.__init__(self)
        
        # Setting main window geometry
        self.setGeometry(50, 50, 950, 950)
        self.center()
        self.mainUI()
        
    # ------------------------------------------- # 
    
    def center(self):
        # geometry of the main window
        qr = self.frameGeometry()

        # center point of screen
        cp = QDesktopWidget().availableGeometry().center()

        # move rectangle's center point to screen's center point
        qr.moveCenter(cp)

        # top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())
        
    # ------------------------------------------- #  
    # ppm, chunk_size, window_type, low_speed, high_speed, **kwargs
    def mainUI(self):
        
        # Initialize layout and title
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Power Spectrum Interactive Plot")
        
        # Data
        self.frequencyBand = 'Delta'
        self.files = [None, None]
        self.position_data = [None, None, None]
        self.active_folder = ''
        self.ppm = 600
        self.chunk_size = 10
        self.window_type = 'hamming'
        self.speed_lowerbound = None
        self.speed_upperbound = None
        self.scaling_factor = None
        self.images = None
        self.freq_dict = None
        self.pos_t = None
         
        # Creating widgets
        windowTypeBox = QComboBox()
        frequencyBandBox = QComboBox()
        
        timeSlider_Label = QLabel()
        ppm_Label = QLabel()
        chunkSize_Label = QLabel()
        speed_Label = QLabel()
        window_Label = QLabel()
        frequency_Label = QLabel()
        self.timeInterval_Label = QLabel()
        session_Label = QLabel()
        self.session_Text = QLabel()
        self.progressBar_Label = QLabel()
        
        ppmTextBox = QLineEdit(self)
        chunkSizeTextBox = QLineEdit(self)
        speedTextBox = QLineEdit()
        # self.sessionTextBox = QLineEdit()
        quit_button = QPushButton('Quit', self)
        browse_button = QPushButton('Browse files', self)
        self.render_button = QPushButton('Re-Render', self)
        # self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        slider = QSlider(Qt.Horizontal)
        self.bar = QProgressBar(self)
        
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.imageMapper = QGraphicsPixmapItem()
        self.scene.addItem(self.imageMapper)
        self.view.centerOn(self.imageMapper)
        
        # Instantiating widget properties 
        timeSlider_Label.setText("Time slider")
        ppm_Label.setText("Pixel per meter (ppm)")
        chunkSize_Label.setText("Chunk size (seconds)")
        speed_Label.setText("Speed filter (optional)")
        window_Label.setText("Window type")
        frequency_Label.setText("Frequency band")
        session_Label.setText("Current session")
        # self.canvas.axes.imshow(np.zeros((1001,1001)),cmap='jet')    # Set default map to empty
        self.render_button.setStyleSheet("background-color : light gray")
        frequencyBandBox.addItem("Delta")
        frequencyBandBox.addItem("Theta")
        frequencyBandBox.addItem("Beta")
        windowTypeBox.addItem("hamming")
        windowTypeBox.addItem("hann")
        windowTypeBox.addItem("blackmanharris")
        windowTypeBox.addItem("boxcar")
        speedTextBox.setPlaceholderText("Ex: Type 5,10 for 5cms to 10cms range filter")
        ppmTextBox.setText("600")
        chunkSizeTextBox.setText("10")
        
        resizeWidgets = [windowTypeBox, chunkSizeTextBox, speedTextBox, ppmTextBox, frequencyBandBox]
        for widget in resizeWidgets:
            widget.setFixedWidth(300)
            
        slider.setTickInterval(10)
        slider.setSingleStep(1)
        
        # Placing widgets
        self.layout.addWidget(browse_button, 0,1)
        self.layout.addWidget(session_Label, 1, 0)
        self.layout.addWidget(self.session_Text, 1, 1)
        self.layout.addWidget(self.render_button, 1, 2)
        self.layout.addWidget(frequency_Label, 2,0)
        self.layout.addWidget(frequencyBandBox, 2,1)
        self.layout.addWidget(window_Label, 3,0)
        self.layout.addWidget(windowTypeBox, 3,1)
        self.layout.addWidget(ppm_Label, 4,0)
        self.layout.addWidget(ppmTextBox, 4,1)
        self.layout.addWidget(chunkSize_Label, 5,0)
        self.layout.addWidget(chunkSizeTextBox, 5,1)
        self.layout.addWidget(speed_Label, 6,0)
        self.layout.addWidget(speedTextBox, 6,1)
        self.layout.addWidget(self.view, 7,1)
        self.layout.addWidget(timeSlider_Label,8,0)
        self.layout.addWidget(slider,8,1)
        self.layout.addWidget(self.timeInterval_Label, 8, 2)
        self.layout.addWidget(self.bar, 9, 1)
        self.layout.addWidget(self.progressBar_Label, 9, 2)
        self.layout.addWidget(quit_button,0,2)
        self.layout.setSpacing(10)
        
        # Widget signaling
        ppmTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'ppm'))
        chunkSizeTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'chunk_size'))
        speedTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'speed'))
        quit_button.clicked.connect(self.quitClicked)
        browse_button.clicked.connect(self.runSession)
        self.render_button.clicked.connect(self.runSession)
        slider.valueChanged[int].connect(self.sliderChanged)
        windowTypeBox.activated[str].connect(self.windowChanged)
        frequencyBandBox.activated[str].connect(self.frequencyChanged)
        
    # ------------------------------------------- #  
    
    # CORRECT IN MAG FOR EMPTY BOXES
    def textBoxChanged(self, label):
        
        cbutton = self.sender()
        curr_string = str(cbutton.text()).split(',')
        self.render_button.setStyleSheet("background-color : rgb(0, 180,0)")
        
        if label == 'speed':
            if len(curr_string) == 1 and len(curr_string[0]) == 0:
                self.speed_lowerbound = self.speed_upperbound = None
            elif len(curr_string) == 1:
                if curr_string[0].isnumeric():
                    self.speed_lowerbound = int(curr_string[0])
                    self.speed_upperbound = 100
            elif len(curr_string) == 2:
                if curr_string[0].isnumeric() and curr_string[1].isnumeric():
                    self.speed_lowerbound = int(curr_string[0])
                    self.speed_upperbound = int(curr_string[1])
                   
        elif label == 'ppm': 
            if len(curr_string) == 1 and len(curr_string[0]) == 0:
                self.ppm = None
            if curr_string[0].isnumeric():
                self.ppm = int(curr_string[0])
                
        elif label == 'chunk_size':
            if len(curr_string) == 1 and len(curr_string[0]) == 0:
                self.chunk_size = None
            if curr_string[0].isnumeric():
                self.chunk_size = int(curr_string[0])
                self.scaling_factor = None
        
     # ------------------------------------------- # 
     
    def openFileNamesDialog(self):
    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Choose .pos file and .eeg/.egf file", self.active_folder, options=options)
        
        if len(files) > 0:
            self.active_folder = dir_path = os.path.dirname(os.path.realpath((files[0]))) 
        
        for file in files:
            extension = file.split(sep='.')[1]
            if 'pos' in extension:
                self.files[0] = file
            elif 'eeg' in extension or 'egf' in extension:
                self.files[1] = file
            else: 
                self.error_dialog.showMessage('You must choose one .pos and one .eeg/.egf file.')
                return
        
        self.scaling_factor = None
        self.session_Text.setText(str(self.files[1]))
    # ------------------------------------------- #
    
    def frequencyChanged(self, value): 
        self.frequencyBand = value
        
        if self.freq_dict != None:
            self.images = self.freq_dict[self.frequencyBand]
        
    # ------------------------------------------- #
    
    def quitClicked(self):
       print('quit')
       QApplication.quit()
       self.close() 
       
    # ------------------------------------------- #  
    
    def sliderChanged(self, value): 
        
        if self.images != None:
            # self.canvas.axes.imshow( self.images[value], cmap='jet')
            # self.canvas.draw()
            self.imageMapper.setPixmap(self.images[value])
            self.timeInterval_Label.setText( "{:.3f}".format(self.pos_t[  value * int(len(self.pos_t)/100)  ]) + "s" )
    
    # ------------------------------------------- #  
    
    def progressBar(self, n):
  
        n = int(n)
        # setting geometry to progress bar
        self.bar.setValue(n)
        
    # ------------------------------------------- #  
    
    def updateLabel(self, value): 
        self.progressBar_Label.setText(value)
        
    # ------------------------------------------- #  
    
    def runSession(self):
        
        cbutton = self.sender()
        
        # Prepare error dialog window 
        boolean_check = True
        self.error_dialog = QErrorMessage()
        
        if (self.speed_lowerbound != None and self.speed_upperbound == None):
            self.speed_upperbound = 100
            
        # If speed filter text is left blank, set default to 0cms to 100cms
        if self.speed_lowerbound == None and self.speed_upperbound == None: 
            self.speed_lowerbound = 0
            self.speed_upperbound = 100
            
        if self.speed_lowerbound != None and self.speed_upperbound != None:
            if self.speed_lowerbound > self.speed_upperbound: 
                self.error_dialog.showMessage('Speed filter range must be ascending. Lower speed first, higher speed after. Ex: 1,5')
                boolean_check = False
        
        # Error checking 
        if self.ppm == None or self.chunk_size == None: 
            self.error_dialog.showMessage('PPM field and/or Chunk Size field is blank, or has a non-numeric input. Please enter appropriate numerics.')
            boolean_check = False
            
        if boolean_check: 
            if cbutton.text() != 'Re-Render':
                self.openFileNamesDialog()
                if len(self.files) == 0: 
                    return
                
                self.worker = Worker(compute_fMap, self.files, self.ppm, self.chunk_size, self.window_type, 
                            self.speed_lowerbound, self.speed_upperbound, scaling_factor=self.scaling_factor)
                self.worker.start()
                self.worker.signals.image_data.connect(self.setData)
                self.worker.signals.progress.connect(self.progressBar)
                self.worker.signals.text_progress.connect(self.updateLabel)
            
            else: 
                
                if self.files[0] == self.files[1] == None:
                     self.error_dialog.showMessage("You haven't selected a session yet from browse files")
                     return
                else: 
                    self.render_button.setStyleSheet("background-color : light gray")
                    self.worker = Worker(compute_fMap, self.files, self.ppm, self.chunk_size, self.window_type, 
                                self.speed_lowerbound, self.speed_upperbound, scaling_factor=self.scaling_factor)
                    self.worker.start()
                    self.worker.signals.image_data.connect(self.setData)
                    self.worker.signals.progress.connect(self.progressBar)
                    self.worker.signals.text_progress.connect(self.updateLabel)
                
    # ------------------------------------------- #  
    
    def windowChanged(self, value): 
        self.window_type = value
        self.scaling_factor = None
        self.render_button.setStyleSheet("background-color : rgb(0, 180,0)")
        
    # ------------------------------------------- #  
    
    def setData(self, data):
        self.freq_dict = data[0]
        self.images = self.freq_dict[self.frequencyBand]
        self.scaling_factor = data[1]
        self.pos_t = data[2]
        
        print(self.images[0])
        print("Data loaded")
        
# =========================================================================== #

class Worker(QThread): 
    
    def __init__(self, function, *args, **kwargs):
        
        QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.data = None
        
        self.signals = WorkerSignals()
        
    # ------------------------------------------- #   
    
    def run(self, **kwargs):
        self.data = self.function(self, *self.args, **self.kwargs)
        self.signals.image_data.emit( (self.data[0], self.data[1], self.data[2]) )
# =========================================================================== #         

app = QApplication(sys.argv)
screen = frequencyPlotWindow()
screen.show()
sys.exit(app.exec_())