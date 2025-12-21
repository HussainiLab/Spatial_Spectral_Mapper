# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:51:19 2021
@author: vajramsrujan
"""

import os
import sys
import matplotlib
import numpy as np
import csv

from PyQt5.QtWidgets import QMessageBox
from functools import partial
from initialize_fMap import initialize_fMap
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from core.data_loaders import grab_position_data
from core.processors.Tint_Matlab import speed2D
from core.processors.spectral_functions import (speed_bins)
from core.worker_thread import WorkerSignals
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from core.worker_thread.Worker import Worker

matplotlib.use('Qt5Agg')
    
# =========================================================================== #

class MplCanvas(FigureCanvasQTAgg):

    '''
        Canvas class to generate matplotlib plot widgets in the GUI.
        This class takes care of plotting any image data.
    '''

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

# =========================================================================== #
        
class frequencyPlotWindow(QWidget):
    
    '''
        Class which handles frequency map plotting on UI
    '''

    def __init__(self):
        
        QWidget.__init__(self)
        
        # Setting main window geometry
        self.center()
        self.mainUI()
        self.showMaximized()
        self.setWindowFlag(Qt.WindowCloseButtonHint, False) # Grey out 'X' button to prevent improper PyQt termination
        
    # ------------------------------------------- # 
    
    def center(self):

        '''
            Centers the GUI window on screen upon launch
        '''

        # Geometry of the main window
        qr = self.frameGeometry()

        # Center point of screen
        cp = QDesktopWidget().availableGeometry().center()

        # Move rectangle's center point to screen's center point
        qr.moveCenter(cp)

        # Top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())
        
    # ------------------------------------------- #  
    
    def mainUI(self):
        
        # Initialize layout and title
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Power Spectrum Interactive Plot")
        
        # Data initialization
        self.plot_flag = False          # Flag to signal if we are plotting graphs or maps. True is graph, false is maps
        self.plot_data = None           # Holds aray of power spectrum densities and frequency axis from welch method
        self.frequencyBand = 'Delta'    # Current band of frequency (choose between Delta, Theta, Beta, Low Gamma, High Gamma)
        self.files = [None, None]       # Holds .pos and .eeg/.egf file
        self.position_data = [None, None, None] # Hods pos_x, pos_y and pos_t
        self.active_folder = ''                 # Keeps track of last accessed directory
        self.ppm = 600                          # Pixel per meter value
        self.chunk_size = 10                    # Size of each signal chunk in seconds (user defined)
        self.window_type = 'hamming'            # Window type for welch 
        self.speed_lowerbound = None            # Lower limit for speed filter
        self.speed_upperbound = None            # Upper limit for speed filter
        self.images = None                      # Holds all frequency map images
        self.freq_dict = None                   # Holds frequency map bands and associated Hz ranges
        self.pos_t = None                       # Time tracking array
        self.scaling_factor_crossband = None    # Will hold scaling factor to normalize maps across all bands
        self.chunk_powers_data = None           # Array of total powers per chunk per band
        self.chunk_index = None                 # Keeps track of what signal chunk we are on
        self.tracking_data = None               # Keeps track of animal's position data (x and y coordinates) for plotting
         
        # Widget initialization
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
        self.frequencyViewer_Label = QLabel()
        self.graph_Label = QLabel()
        self.tracking_Label = QLabel()
        
        ppmTextBox = QLineEdit(self)
        chunkSizeTextBox = QLineEdit(self)
        speedTextBox = QLineEdit()
        quit_button = QPushButton('Quit', self)
        browse_button = QPushButton('Browse file', self)
        self.graph_mode_button = QPushButton('Graph mode', self)
        self.render_button = QPushButton('Re-Render', self)
        save_button = QPushButton('Save data', self)
        self.slider = QSlider(Qt.Horizontal)
        self.bar = QProgressBar(self)
        
        # Create canvases for embedded plotting
        self.graph_canvas = MplCanvas(self, width=5, height=5, dpi=100)     # For fft plotting
        self.tracking_canvas = MplCanvas(self, width=6, height=6, dpi=100)  # For position tracking plotting

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.imageMapper = QGraphicsPixmapItem()
        self.scene.addItem(self.imageMapper)
        self.view.centerOn(self.imageMapper)
        self.view.scale(3,3)
        
        # Instantiating widget properties 
        timeSlider_Label.setText("Time slider")
        ppm_Label.setText("Pixel per meter (ppm)")
        chunkSize_Label.setText("Chunk size (seconds)")
        speed_Label.setText("Speed filter (optional)")
        window_Label.setText("Window type")
        frequency_Label.setText("Frequency band")
        session_Label.setText("Current session")
        self.frequencyViewer_Label.setText("Frequency map")
        self.graph_Label.setText("Power spectrum graph")
        self.tracking_Label.setText("Animal tracking")
        self.bar.setOrientation(Qt.Vertical)
        self.render_button.setStyleSheet("background-color : light gray")
        frequencyBandBox.addItem("Delta")
        frequencyBandBox.addItem("Theta")
        frequencyBandBox.addItem("Beta")
        frequencyBandBox.addItem("Low Gamma")
        frequencyBandBox.addItem("High Gamma")
        frequencyBandBox.addItem("Ripple") #Abid 4/16/2022
        frequencyBandBox.addItem("Fast Ripple")
        windowTypeBox.addItem("hamming")
        windowTypeBox.addItem("hann")
        windowTypeBox.addItem("blackmanharris")
        windowTypeBox.addItem("boxcar")
        speedTextBox.setPlaceholderText("Ex: Type 5,10 for 5cms to 10cms range filter")
        ppmTextBox.setText("600")
        chunkSizeTextBox.setText("10")
        
        # Set font sizes of label headers
        self.frequencyViewer_Label.setFont(QFont("Times New Roman", 18))
        self.graph_Label.setFont(QFont("Times New Roman", 18))
        self.tracking_Label.setFont(QFont("Times New Roman", 18))

        # Set header alignments
        self.frequencyViewer_Label.setAlignment(Qt.AlignCenter)
        self.graph_Label.setAlignment(Qt.AlignCenter)
        self.tracking_Label.setAlignment(Qt.AlignCenter)
        
        # Resize widgets to fixed width
        resizeWidgets = [windowTypeBox, chunkSizeTextBox, speedTextBox, ppmTextBox, 
                         frequencyBandBox, browse_button]
        for widget in resizeWidgets:
            widget.setFixedWidth(300)
        
        # Set width of righthand buttons
        quit_button.setFixedWidth(150)
        save_button.setFixedWidth(150)
        self.render_button.setFixedWidth(150)

        # Placing widgets
        self.layout.addWidget(browse_button, 0,1)
        self.layout.addWidget(session_Label, 1, 0)
        self.layout.addWidget(self.session_Text, 1, 1)
        self.layout.addWidget(self.render_button, 1, 2, alignment=Qt.AlignRight)
        self.layout.addWidget(frequency_Label, 2,0)
        self.layout.addWidget(frequencyBandBox, 2,1)
        self.layout.addWidget(save_button,2,2, alignment=Qt.AlignRight)
        self.layout.addWidget(window_Label, 3,0)
        self.layout.addWidget(windowTypeBox, 3,1)
        self.layout.addWidget(ppm_Label, 4,0)
        self.layout.addWidget(ppmTextBox, 4,1)
        self.layout.addWidget(chunkSize_Label, 5,0)
        self.layout.addWidget(chunkSizeTextBox, 5,1)
        self.layout.addWidget(speed_Label, 6,0)
        self.layout.addWidget(speedTextBox, 6,1)
        self.layout.addWidget(self.graph_mode_button, 7, 0)
        self.layout.addWidget(self.frequencyViewer_Label, 7,1)
        self.layout.addWidget(self.graph_Label, 7,1)
        self.layout.addWidget(self.tracking_Label, 7,2)
        self.layout.addWidget(self.power_Label, 8, 0)
        self.layout.addWidget(self.view, 8,1)
        self.layout.addWidget(self.graph_canvas, 8,1)
        self.layout.addWidget(self.tracking_canvas, 8,2)
        self.layout.addWidget(self.bar, 8,3)
        self.layout.addWidget(timeSlider_Label,9,0)
        self.layout.addWidget(self.slider,9,1)
        self.layout.addWidget(self.timeInterval_Label, 9, 3)
        self.layout.addWidget(self.progressBar_Label, 10, 3)
        self.layout.addWidget(quit_button,0,2, alignment=Qt.AlignRight)
        self.layout.setSpacing(10)
        
        # Hiding the canvas and graph label widget on startup 
        self.graph_canvas.close()
        self.graph_Label.close()
        
        # Widget signaling
        ppmTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'ppm'))
        chunkSizeTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'chunk_size'))
        speedTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'speed'))
        quit_button.clicked.connect(self.quitClicked)
        browse_button.clicked.connect(self.runSession)
        self.graph_mode_button.clicked.connect(self.switch_graph)
        save_button.clicked.connect(self.saveClicked)
        self.render_button.clicked.connect(self.runSession)
        self.slider.valueChanged[int].connect(self.sliderChanged)
        windowTypeBox.activated[str].connect(self.windowChanged)
        frequencyBandBox.activated[str].connect(self.frequencyChanged)
        
    # ------------------------------------------- #  
    
    def textBoxChanged(self, label):
        
        '''
            Invoked when any one of the textboxes have their text changed.
            Handles the new input, sets variables.

            Params: 
                label (str) : 
                    The textbox label as a string. Chose between 'Speed', 
                    'ppm', and 'chunk_size'

            Returns: No return
        '''

        cbutton = self.sender()
        # If any parameter is changed, will highlight
        # the re-render button to signal a re-rendering is needed
        curr_string = str(cbutton.text()).split(',')
        self.render_button.setStyleSheet("background-color : rgb(0, 180,0)")
        
        # The following fields are only set if field inputs are numeric
        # Sets speed filtering limits 
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
        # Sets ppm           
        elif label == 'ppm': 
            if len(curr_string) == 1 and len(curr_string[0]) == 0:
                self.ppm = None
            if curr_string[0].isnumeric():
                self.ppm = int(curr_string[0])
        # Sets chunk size        
        elif label == 'chunk_size':
            if len(curr_string) == 1 and len(curr_string[0]) == 0:
                self.chunk_size = None
            if curr_string[0].isnumeric():
                self.chunk_size = int(curr_string[0])
        
     # ------------------------------------------- # 
     
    def openFileNamesDialog(self):
    
        '''
            Will query OS to open file dialog box to choose pos and eeg/egf file.
            Additionally remembers the last opened directory. 
            
            Returns: 
                bool: bool value depending on whether appropriate files were chosen.
        '''

        # Set file dialog options
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # Open file dialog
        files, _ = QFileDialog.getOpenFileNames(self, "Choose .eeg/.egf file", self.active_folder, options=options)

        # If a file was chosen, remember directory
        if len(files) > 0:
            self.active_folder = dir_path = os.path.dirname(os.path.realpath((files[0])))
        else:
            return False

        # Reset tracked selection
        self.files = [None, None]

        # Helper to set by extension (case-insensitive)
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ('pos' in ext):
                self.files[0] = file
            elif ('eeg' in ext) or ('egf' in ext):
                self.files[1] = file

        # If only EEG/EGF was selected, try auto-find matching .pos by basename
        if self.files[1] and not self.files[0]:
            eeg_path = self.files[1]
            base_no_ext = os.path.splitext(eeg_path)[0]
            candidate_pos = base_no_ext + '.pos'
            candidate_pos_upper = base_no_ext + '.POS'
            if os.path.exists(candidate_pos):
                self.files[0] = candidate_pos
            elif os.path.exists(candidate_pos_upper):
                self.files[0] = candidate_pos_upper
            else:
                # Try glob for any .pos sharing the same base name segment before last dot
                base_dir = os.path.dirname(eeg_path)
                base_name = os.path.basename(base_no_ext)
                # Find any file that starts with base_name and ends with .pos
                for name in os.listdir(base_dir):
                    if name.lower().endswith('.pos') and name.startswith(base_name):
                        self.files[0] = os.path.join(base_dir, name)
                        break

        # Validate we have at least EEG/EGF; and ensure POS exists (either selected or auto-found)
        if not self.files[1]:
            self.error_dialog.showMessage('Please select an .eeg/.egf file.')
            return False
        if not self.files[0]:
            self.error_dialog.showMessage('Matching .pos file not found. Please select the .pos file as well.')
            return False

        # Reflect session name using the electrophysiology file
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

        '''
            Application exit
        '''

        print('quit')
        QApplication.quit()
        self.close() 
    
    # ------------------------------------------- #
    
    def saveClicked(self):
        
        '''
            Automatically saves CSV and Excel with average frequency powers vs time
            into the selected folder with filename suffix '_SSM'.
        '''

        # If there are no chunk powers, do nothing
        if self.chunk_powers_data is None:
            return

        # Expected band order and labels
        bands = [
            ("Delta", "Avg Delta Power"),
            ("Theta", "Avg Theta Power"),
            ("Beta", "Avg Beta Power"),
            ("Low Gamma", "Avg Low Gamma Power"),
            ("High Gamma", "Avg High Gamma Power"),
            ("Ripple", "Avg Ripple Power"),
            ("Fast Ripple", "Avg Fast Ripple Power"),
        ]

        # Determine output directory and base name
        out_dir = self.active_folder or os.getcwd()
        base_name = os.path.splitext(os.path.basename(self.files[1] or 'output'))[0]
        out_csv = os.path.join(out_dir, f"{base_name}_SSM.csv")

        # Build header and rows and write CSV without external apps
        band_labels = [label for _, label in bands]
        percent_labels = [f"Percent {name}" for name, _ in bands]
        
        # Calculate distances for each chunk if tracking data is available
        distances_per_bin = []
        cumulative_distances = []
        cumulative_sum = 0.0
        
        if self.tracking_data is not None and len(self.tracking_data) == 2:
            pos_x_chunks, pos_y_chunks = self.tracking_data
            
            for i in range(len(pos_x_chunks)):
                distance_cm_in_bin = 0.0
                
                # Get the positions for this bin only (not cumulative from start)
                if i == 0:
                    # First bin: use all positions in the first chunk
                    x_bin = pos_x_chunks[i]
                    y_bin = pos_y_chunks[i]
                else:
                    # Subsequent bins: positions from previous chunk end to current chunk end
                    # Since chunks are cumulative from 0, we need to subtract
                    prev_len = len(pos_x_chunks[i-1])
                    x_bin = pos_x_chunks[i][prev_len:]
                    y_bin = pos_y_chunks[i][prev_len:]
                
                # Calculate distance within this specific bin only
                if len(x_bin) > 1:
                    dx = np.diff(np.array(x_bin))
                    dy = np.diff(np.array(y_bin))
                    distances_in_bin_pixels = np.sqrt(dx**2 + dy**2)
                    total_distance_pixels_in_bin = np.sum(distances_in_bin_pixels)
                    
                    # Convert from pixels to centimeters using PPM
                    if self.ppm is not None and self.ppm > 0:
                        distance_cm_in_bin = (total_distance_pixels_in_bin / self.ppm) * 100
                    else:
                        distance_cm_in_bin = total_distance_pixels_in_bin
                
                # Store distance for this bin only (not cumulative)
                distances_per_bin.append(distance_cm_in_bin)
                
                # Update cumulative distance
                cumulative_sum += distance_cm_in_bin
                cumulative_distances.append(cumulative_sum)
        
        header = ["Time Bin (s)", "Distance Per Bin (cm)", "Cumulative Distance (cm)"] + band_labels + percent_labels
        
        # Determine chunk size for proper time bin labeling
        chunk_size = self.chunk_size if self.chunk_size is not None else 10
        
        # Calculate the actual recording duration and round down to nearest chunk boundary
        # This handles cases where recording is slightly over (e.g., 901s, 1201s, 601s)
        if len(self.pos_t) > 0:
            actual_duration = float(self.pos_t[-1])
            # Round down to nearest chunk boundary
            max_full_chunks = int(actual_duration / chunk_size)
            num_rows = min(len(self.pos_t), max_full_chunks)
        else:
            num_rows = len(self.pos_t)
        
        # Gather per-band arrays, ensure 1D length matches timestamps
        band_arrays = {}
        for key, label in bands:
            arr = np.array(self.chunk_powers_data.get(key, [])).reshape(-1)
            band_arrays[label] = arr
        
        # Write CSV
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(num_rows):
                # Create time bin string based on chunk size
                # Bins start at chunk_size because first chunk (0 to chunk_size) is used for baseline
                time_bin_start = i * chunk_size
                time_bin_end = (i + 1) * chunk_size
                time_bin_str = f"{time_bin_start}-{time_bin_end}"
                
                row = [time_bin_str]
                # Add distance per bin
                if distances_per_bin and i < len(distances_per_bin):
                    row.append(round(distances_per_bin[i], 3))
                else:
                    row.append("")
                # Add cumulative distance
                if cumulative_distances and i < len(cumulative_distances):
                    row.append(round(cumulative_distances[i], 3))
                else:
                    row.append("")
                band_values = []
                for _, label in bands:
                    val = band_arrays[label][i] if i < len(band_arrays[label]) else ""
                    band_values.append(val)
                # Append band values
                row.extend(band_values)
                # Compute total power across bands at this row (ignore blanks)
                numeric_vals = [float(v) for v in band_values if v != ""]
                total_power = sum(numeric_vals)
                # Compute per-band percentages
                for v in band_values:
                    if v == "" or total_power == 0:
                        row.append("")
                    else:
                        row.append(round((float(v) / total_power) * 100.0, 3))
                writer.writerow(row)
        QMessageBox.information(self, "Save Complete", f"Data saved to:\n{out_csv}")
            
    # ------------------------------------------- #
    
    def switch_graph(self):
        
        '''
            Will switch between plotting power spectral density per chunk to frequency map per chunk.
        '''

        cbutton = self.sender()
        # Show the PSD graph is graph mode is chosen
        if cbutton.text() == 'Graph mode':
            self.graph_canvas.show()
            self.view.close()
            self.frequencyViewer_Label.close()
            self.graph_Label.show()
            self.plot_flag = True
            self.graph_mode_button.setText("Frequency image mode")
            # Only show the graph if there is something to plot
            if self.chunk_index != None and self.plot_data is not None:
                freq, pdf = self.plot_data[self.chunk_index][0]
                self.graph_canvas.axes.plot( freq, pdf , linewidth=0.5 )
        # Else show freq map
        else:
            self.view.show()
            self.graph_canvas.close()
            self.graph_Label.close()
            self.frequencyViewer_Label.show()
            self.plot_flag = False
            self.graph_mode_button.setText("Graph mode")
        
    # ------------------------------------------- #  
    
    def sliderChanged(self, value): 
        
        '''
            Create a slider that allows the user to sift through each chunk and
            view how the graph/frequency maps change as a function of time. 
        '''

        # Sliders value acts as chunk index
        self.chunk_index = value 
        
        # If we have data to plot, plot the graph
        if self.plot_flag:
            if self.plot_data is not None:
                freq, pdf = self.plot_data[value][0]
                self.graph_canvas.axes.cla()
                self.graph_canvas.axes.plot( freq, pdf, linewidth=0.5 )
                self.graph_canvas.axes.set_xlabel('Frequency (Hz)')
                self.graph_canvas.axes.set_ylabel('Power Spectral Density (microWatts / Hz)')
                self.graph_canvas.draw()
        # Else if we have maps to plot, plot the frequency maps    
        elif self.images is not None:
            self.imageMapper.setPixmap(self.images[value])
        
        if self.tracking_data is not None:
            self.tracking_canvas.axes.cla()
            self.tracking_canvas.axes.set_xlabel('X - coordinates')
            self.tracking_canvas.axes.set_ylabel('Y - coordinates')
            self.tracking_canvas.axes.plot(self.tracking_data[0][value], self.tracking_data[1][value], linewidth=0.5)
            self.tracking_canvas.draw()
        # Reflect what time interval we are plotting
        if self.pos_t is not None:
            self.timeInterval_Label.setText( "{:.3f}".format(self.pos_t[value]) + "s" )
    
    # ------------------------------------------- #  
    
    def progressBar(self, n):
  
        '''
            Reflects progress of frequency map and scaling factor computations
        '''

        n = int(n)
        # Setting geometry to progress bar
        self.bar.setValue(n)
        
    # ------------------------------------------- #  
    
    def updateLabel(self, value): 

        ''' 
            Updates the value of the progress bar
        '''

        self.progressBar_Label.setText(value)
        
    # ------------------------------------------- #  
    
    def runSession(self):
        
        '''
            Error checks user input, and invokes worker thread function 
            to compute maps, graphs and scaling factors.
        '''

        cbutton = self.sender()
        
        # Prepare error dialog window 
        boolean_check = True
        self.error_dialog = QErrorMessage()
        
        # If speed input only specifies lower bound, set upperbound to default
        if (self.speed_lowerbound != None and self.speed_upperbound == None):
            self.speed_upperbound = 100
            
        # If speed filter text is left blank, set default to 0cms to 100cms
        if self.speed_lowerbound == None and self.speed_upperbound == None: 
            self.speed_lowerbound = 0
            self.speed_upperbound = 100

         # Sheck speed bounds are ascending
        if self.speed_lowerbound != None and self.speed_upperbound != None:
            if self.speed_lowerbound > self.speed_upperbound: 
                self.error_dialog.showMessage('Speed filter range must be ascending. Lower speed first, higher speed after. Ex: 1,5')
                boolean_check = False
        
        # Error checking ppm 
        if self.ppm == None or self.chunk_size == None: 
            self.error_dialog.showMessage('PPM field and/or Chunk Size field is blank, or has a non-numeric input. Please enter appropriate numerics.')
            boolean_check = False
            
        # If all checks pass, and we are not in re-render mode, query user for files.
        if boolean_check: 
            if cbutton.text() != 'Re-Render':
                run_flag = self.openFileNamesDialog()
                # If the user did not choose the correct files, do not execute thread.
                if not run_flag:
                    return
                
                # Set and execute initialize_fMap function in worker thread
                self.worker = Worker(initialize_fMap, self.files, self.ppm, self.chunk_size, self.window_type, 
                            self.speed_lowerbound, self.speed_upperbound)
                self.worker.start()
                # Worker thread signaling
                self.worker.signals.image_data.connect(self.setData)
                self.worker.signals.progress.connect(self.progressBar)
                self.worker.signals.text_progress.connect(self.updateLabel)
             
            # If we are in re-render mode
            else: 
                # If no files chosen
                if self.files[0] == self.files[1] == None:
                     self.error_dialog.showMessage("You haven't selected a session yet from Browse file")
                     return

                else: 
                    # Set re-render button back to default once files have been chosen
                    self.render_button.setStyleSheet("background-color : light gray")
                    # Launch worker thread
                    self.worker = Worker(initialize_fMap, self.files, self.ppm, self.chunk_size, self.window_type, 
                                self.speed_lowerbound, self.speed_upperbound)
                    self.worker.start()
                    self.worker.signals.image_data.connect(self.setData)
                    self.worker.signals.progress.connect(self.progressBar)
                    self.worker.signals.text_progress.connect(self.updateLabel)
                
    # ------------------------------------------- #  
    
    def windowChanged(self, value): 

        '''
            Updates window type if user chooses different window. 
            Also invokes color change on re-render button to signal that a re-render is needed.
        '''

        self.window_type = value
        self.render_button.setStyleSheet("background-color : rgb(0, 180,0)")
        
    # ------------------------------------------- #  
    
    def setData(self, data):

        '''
            Acquire and set references from returned worker thread data.
            For details on the nature of return value, check initialize_fMap.py 
            return description. 
        '''

        self.freq_dict = data[0]    # Grab frequency dictionary
        self.plot_data = data[1]    # Grab graph plot data
        self.images = self.freq_dict[self.frequencyBand]    # Grab frequency maps
        self.pos_t = data[2]                                
        self.scaling_factor_crossband = data[3]             # Scaling factor crossband dictionary
        self.chunk_powers_data = data[4]                    # Array of chunk power data
        self.tracking_data = data[5]
        # Set label on UI to show what percentage of the signal power the chosen band comprises
        self.power_Label.setText( "{:.3f}".format( self.scaling_factor_crossband[self.frequencyBand] * 100) + "% of overall signal" )
        
        # Slider limits
        self.slider.setMinimum(0)
        self.slider.setMaximum( len(self.images)-1 )
        self.slider.setSingleStep(1)
        
        print("Data loaded")
        
# =========================================================================== #         

def main(): 
    
    '''
        Main function invokes application start.
    '''
    
    app = QApplication(sys.argv)
    screen = frequencyPlotWindow()
    screen.show()
    sys.exit(app.exec_())
    
if __name__=="__main__":
    main()
