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
from initialize_fMap import initialize_fMap

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
from PyQt5.QtWidgets import *
      
# =========================================================================== #

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

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
        self.plot_flag = False
        self.plot_data = None
        self.frequencyBand = 'Delta'
        self.files = [None, None]
        self.position_data = [None, None, None]
        self.active_folder = ''
        self.ppm = 600
        self.chunk_size = 10
        self.window_type = 'hamming'
        self.speed_lowerbound = None
        self.speed_upperbound = None
        self.images = None
        self.freq_dict = None
        self.pos_t = None
        self.scaling_factor_crossband = None
        self.chunk_index = None
         
        # Creating widgets
        windowTypeBox = QComboBox()
        frequencyBandBox = QComboBox()
        
        timeSlider_Label = QLabel()
        ppm_Label = QLabel()
        chunkSize_Label = QLabel()
        speed_Label = QLabel()
        window_Label = QLabel()
        frequency_Label = QLabel()
        session_Label = QLabel()
        self.timeInterval_Label = QLabel()
        self.session_Text = QLabel()
        self.progressBar_Label = QLabel()
        self.power_Label = QLabel()
        
        ppmTextBox = QLineEdit(self)
        chunkSizeTextBox = QLineEdit(self)
        speedTextBox = QLineEdit()
        quit_button = QPushButton('Quit', self)
        browse_button = QPushButton('Browse files', self)
        self.graph_mode_button = QPushButton('Graph mode', self)
        self.render_button = QPushButton('Re-Render', self)
        self.slider = QSlider(Qt.Horizontal)
        self.bar = QProgressBar(self)
        
        self.canvas = MplCanvas(self, width=5, height=5, dpi=100)
        
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.imageMapper = QGraphicsPixmapItem()
        self.scene.addItem(self.imageMapper)
        self.view.centerOn(self.imageMapper)
        self.view.scale(5,5)
        
        # Instantiating widget properties 
        timeSlider_Label.setText("Time slider")
        ppm_Label.setText("Pixel per meter (ppm)")
        chunkSize_Label.setText("Chunk size (seconds)")
        speed_Label.setText("Speed filter (optional)")
        window_Label.setText("Window type")
        frequency_Label.setText("Frequency band")
        session_Label.setText("Current session")
        self.bar.setOrientation(Qt.Vertical)
        self.render_button.setStyleSheet("background-color : light gray")
        frequencyBandBox.addItem("Delta")
        frequencyBandBox.addItem("Theta")
        frequencyBandBox.addItem("Beta")
        frequencyBandBox.addItem("Low Gamma")
        frequencyBandBox.addItem("High Gamma")
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
            
        # Placing widgets
        self.layout.addWidget(self.graph_mode_button, 0, 0)
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
        self.layout.addWidget(self.power_Label, 7, 0)
        self.layout.addWidget(self.view, 7,1)
        self.layout.addWidget(self.canvas, 7,1)
        self.layout.addWidget(self.bar, 7,2)
        self.layout.addWidget(timeSlider_Label,8,0)
        self.layout.addWidget(self.slider,8,1)
        self.layout.addWidget(self.timeInterval_Label, 8, 2)
        self.layout.addWidget(self.progressBar_Label, 9, 2)
        self.layout.addWidget(quit_button,0,2)
        self.layout.setSpacing(10)
        
        # Hiding the canvas widget on startup (view and canvas are in the same place)
        self.canvas.close()
        
        # Widget signaling
        ppmTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'ppm'))
        chunkSizeTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'chunk_size'))
        speedTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'speed'))
        quit_button.clicked.connect(self.quitClicked)
        browse_button.clicked.connect(self.runSession)
        self.graph_mode_button.clicked.connect(self.switch_graph)
        
        self.render_button.clicked.connect(self.runSession)
        self.slider.valueChanged[int].connect(self.sliderChanged)
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
        
     # ------------------------------------------- # 
     
    def openFileNamesDialog(self):
    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Choose .pos file and .eeg/.egf file", self.active_folder, options=options)
        
        if len(files) > 0:
            self.active_folder = dir_path = os.path.dirname(os.path.realpath((files[0]))) 
        else: 
            return False
        for file in files:
            extension = file.split(sep='.')[1]
            if 'pos' in extension:
                self.files[0] = file
            elif 'eeg' in extension or 'egf' in extension:
                self.files[1] = file
            else: 
                self.error_dialog.showMessage('You must choose one .pos and one .eeg/.egf file.')
                return False
        
        self.session_Text.setText(str(self.files[1]))
        return True
    # ------------------------------------------- #
    
    def frequencyChanged(self, value): 
        self.frequencyBand = value
        
        if self.scaling_factor_crossband != None:
            self.power_Label.setText( "{:.3f}".format( self.scaling_factor_crossband[self.frequencyBand]  * 100 ) + "% of overall signal" )
        
        if self.freq_dict != None:
            self.images = self.freq_dict[self.frequencyBand]
        
    # ------------------------------------------- #
    
    def quitClicked(self):
       print('quit')
       QApplication.quit()
       self.close() 
    
    # ------------------------------------------- #
    
    def switch_graph(self):
        
        cbutton = self.sender()
        if cbutton.text() == 'Graph mode':
            # self.layout.replaceWidget(self.view, self.canvas)
            self.canvas.show()
            self.view.close()
            self.plot_flag = True
            self.graph_mode_button.setText("Frequency image mode")
            
            if self.chunk_index != None and self.plot_data is not None:
                freq, pdf = self.plot_data[self.chunk_index][0]
                self.canvas.axes.plot( freq, pdf , linewidth=0.5 )
        else:
            # self.layout.replaceWidget(self.canvas, self.view)
            self.view.show()
            self.canvas.close()
            self.plot_flag = False
            self.graph_mode_button.setText("Graph mode")
        
    # ------------------------------------------- #  
    
    def sliderChanged(self, value): 
        
        self.chunk_index = value 
        
        if self.plot_flag:
            if self.plot_data is not None:
                freq, pdf = self.plot_data[value][0]
                self.canvas.axes.cla()
                self.canvas.axes.plot( freq, pdf, linewidth=0.5 )
                self.canvas.draw()
            
                
        elif self.images is not None:
            self.imageMapper.setPixmap(self.images[value])
            
        self.timeInterval_Label.setText( "{:.3f}".format(self.pos_t[value]) + "s" )
    
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
                run_flag = self.openFileNamesDialog()
                
                if not run_flag:
                    return
                
                self.worker = Worker(initialize_fMap, self.files, self.ppm, self.chunk_size, self.window_type, 
                            self.speed_lowerbound, self.speed_upperbound)
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
                    self.worker = Worker(initialize_fMap, self.files, self.ppm, self.chunk_size, self.window_type, 
                                self.speed_lowerbound, self.speed_upperbound)
                    self.worker.start()
                    self.worker.signals.image_data.connect(self.setData)
                    self.worker.signals.progress.connect(self.progressBar)
                    self.worker.signals.text_progress.connect(self.updateLabel)
                
    # ------------------------------------------- #  
    
    def windowChanged(self, value): 
        self.window_type = value
        self.render_button.setStyleSheet("background-color : rgb(0, 180,0)")
        
    # ------------------------------------------- #  
    
    def setData(self, data):
        self.freq_dict = data[0]
        self.plot_data = data[1]
        self.images = self.freq_dict[self.frequencyBand]
        self.pos_t = data[2]
        self.scaling_factor_crossband = data[3]
        self.power_Label.setText( "{:.3f}".format( self.scaling_factor_crossband[self.frequencyBand] * 100) + "% of overall signal" )
        
        self.slider.setMinimum(0)
        self.slider.setMaximum( len(self.images)-1 )
        self.slider.setSingleStep(1)
        
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
        self.signals.image_data.emit( (self.data[0], self.data[1], self.data[2], self.data[3]) )
# =========================================================================== #         

app = QApplication(sys.argv)
screen = frequencyPlotWindow()
screen.show()
sys.exit(app.exec_())