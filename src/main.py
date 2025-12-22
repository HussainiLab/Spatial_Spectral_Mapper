# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:51:19 2021
Updated on Dec 21 2025
@author: vajramsrujan
@author: Hussainilab
"""

import os
import sys
import matplotlib
import numpy as np
import csv
import glob
from pathlib import Path

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
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import *
from core.worker_thread.Worker import Worker

matplotlib.use('Qt5Agg')
    
# =========================================================================== #

class BatchSignals(QThread):
    '''Wrapper class to provide signals interface compatible with initialize_fMap'''
    def __init__(self):
        self.progress = None
        self.text_progress = None

class BatchProgressSignal:
    '''Signal wrapper for progress updates'''
    def __init__(self, parent_signal):
        self.parent_signal = parent_signal
    
    def emit(self, value):
        if self.parent_signal:
            self.parent_signal.emit(value)

class BatchTextSignal:
    '''Signal wrapper for text progress updates'''
    def __init__(self, parent_signal):
        self.parent_signal = parent_signal
    
    def emit(self, value):
        if self.parent_signal:
            self.parent_signal.emit(value)

class BatchWorkerThread(QThread):
    '''
        Worker thread for batch processing multiple files in a folder.
        Emits signals for progress updates.
    '''
    progress_update = pyqtSignal(str)  # For text progress updates
    progress_value = pyqtSignal(int)   # For progress bar
    finished_signal = pyqtSignal(dict) # For completion with results
    error_signal = pyqtSignal(str)     # For error messages
    
    def __init__(self, folder_path, ppm, chunk_size, window_type, 
                 low_speed, high_speed, output_dir=None):
        super().__init__()
        self.folder_path = folder_path
        self.ppm = ppm
        self.chunk_size = chunk_size
        self.window_type = window_type
        self.low_speed = low_speed
        self.high_speed = high_speed
        self.output_dir = output_dir or folder_path
        self.total_files = 0
        self.processed_files = 0
        self.successful_files = []
        self.failed_files = []
        
        # Create signals wrapper for compatibility with initialize_fMap
        self.signals = BatchSignals()
        self.signals.progress = BatchProgressSignal(self.progress_value)
        self.signals.text_progress = BatchTextSignal(self.progress_update)
    
    def find_recording_files(self):
        '''
            Find electrophysiology recordings in folder by priority:
              1) .egf
              2) .egf2, .egf3, .egf4
              3) .eeg
              4) .eeg2, .eeg3, .eeg4
            Returns: (recordings_list, missing_pos_bases, found_any_ephys)
        '''
        recordings = {}
        
        # Gather all candidate files across prioritized extensions
        patterns = ["*.egf", "*.egf2", "*.egf3", "*.egf4",
                    "*.eeg", "*.eeg2", "*.eeg3", "*.eeg4"]
        all_files = []
        for pat in patterns:
            all_files.extend(glob.glob(os.path.join(self.folder_path, pat)))
        
        # Group by basename (without extension)
        base_to_files = {}
        for fpath in all_files:
            base = Path(fpath).stem
            ext = Path(fpath).suffix.lower().lstrip('.')  # e.g., 'egf', 'egf2', 'eeg3'
            base_to_files.setdefault(base, {})[ext] = fpath
        
        # Selection priority
        priority = ["egf", "egf2", "egf3", "egf4",
                    "eeg", "eeg2", "eeg3", "eeg4"]
        
        missing_pos_bases = []

        # For each base, pick best available by priority and ensure .pos exists
        for base, ext_map in base_to_files.items():
            chosen = None
            for ext in priority:
                if ext in ext_map:
                    chosen = ext_map[ext]
                    break
            if not chosen:
                continue
            pos_file = self._find_pos_file(chosen)
            if pos_file:
                recordings[base] = (chosen, pos_file)
            else:
                missing_pos_bases.append(base)
        
        found_any_ephys = len(base_to_files) > 0
        return list(recordings.values()), missing_pos_bases, found_any_ephys
    
    def _find_pos_file(self, eeg_file):
        '''Auto-detect .pos file based on .eeg/.egf filename'''
        base_name = os.path.splitext(eeg_file)[0]
        pos_file = base_name + '.pos'
        
        if os.path.exists(pos_file):
            return pos_file
        return None
    
    def run(self):
        '''Execute batch processing in worker thread'''
        try:
            recordings, missing_pos_bases, found_any_ephys = self.find_recording_files()
            self.total_files = len(recordings)
            
            # Report missing .pos per base
            if missing_pos_bases:
                bases_preview = ", ".join(missing_pos_bases[:5])
                more_count = len(missing_pos_bases) - 5
                suffix = f" ... and {more_count} more" if more_count > 0 else ""
                self.error_signal.emit(
                    f"Missing .pos files for {len(missing_pos_bases)} base(s): {bases_preview}{suffix}"
                )

            # If no recordings and no ephys found at all, show specific message
            if not recordings and not found_any_ephys:
                self.error_signal.emit(
                    f"No electrophysiology files found in {self.folder_path}. Expected .egf/.egf2-4 or .eeg/.eeg2-4."
                )
                self.finished_signal.emit({'successful': [], 'failed': []})
                return
            
            # If some ephys were found but none had .pos, end gracefully
            if not recordings and found_any_ephys:
                self.error_signal.emit(
                    f"No recordings could be processed because matching .pos files were missing."
                )
                self.finished_signal.emit({'successful': [], 'failed': []})
                return
            
            self.progress_update.emit(f"Found {self.total_files} file(s) to process")
            
            for electrophys_file, pos_file in recordings:
                try:
                    self.processed_files += 1
                    filename = os.path.basename(electrophys_file)
                    
                    self.progress_update.emit(f"[{self.processed_files}/{self.total_files}] Processing: {filename}")
                    self.progress_value.emit(int((self.processed_files / self.total_files) * 100))
                    
                    # Call initialize_fMap to process the file
                    result = initialize_fMap(
                        self,  # Pass self as the worker/signals object
                        files=[pos_file, electrophys_file],
                        ppm=self.ppm,
                        chunk_size=self.chunk_size,
                        window_type=self.window_type,
                        low_speed=self.low_speed,
                        high_speed=self.high_speed
                    )
                    
                    # Export results to CSV
                    self._export_to_csv(result, electrophys_file)
                    self.successful_files.append(filename)
                    self.progress_update.emit(f"  ✓ Complete: {filename}")
                    
                except Exception as e:
                    self.failed_files.append((os.path.basename(electrophys_file), str(e)))
                    self.progress_update.emit(f"  ✗ Failed: {os.path.basename(electrophys_file)} - {str(e)}")
            
            # Emit completion signal
            results = {
                'successful': self.successful_files,
                'failed': self.failed_files,
                'total': self.total_files
            }
            self.finished_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(f"Batch processing error: {str(e)}")
            self.finished_signal.emit({'successful': [], 'failed': []})
    
    def _export_to_csv(self, result, electrophys_file):
        '''Export processing results to CSV'''
        freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_powers_data, tracking_data = result
        
        base_name = os.path.splitext(os.path.basename(electrophys_file))[0]
        output_csv = os.path.join(self.output_dir, f"{base_name}_SSM.csv")
        
        bands = [
            ("Delta", "Avg Delta Power"),
            ("Theta", "Avg Theta Power"),
            ("Beta", "Avg Beta Power"),
            ("Low Gamma", "Avg Low Gamma Power"),
            ("High Gamma", "Avg High Gamma Power"),
            ("Ripple", "Avg Ripple Power"),
            ("Fast Ripple", "Avg Fast Ripple Power"),
        ]
        
        band_labels = [label for _, label in bands]
        percent_labels = [f"Percent {name}" for name, _ in bands]
        
        # Calculate distances for each chunk
        distances_per_bin = []
        cumulative_distances = []
        cumulative_sum = 0.0
        
        if tracking_data:
            pos_x_chunks, pos_y_chunks = tracking_data
            
            for i in range(len(pos_x_chunks)):
                distance_cm_in_bin = 0.0
                
                if i == 0:
                    x_bin = pos_x_chunks[i]
                    y_bin = pos_y_chunks[i]
                else:
                    prev_len = len(pos_x_chunks[i-1])
                    x_bin = pos_x_chunks[i][prev_len:]
                    y_bin = pos_y_chunks[i][prev_len:]
                
                if len(x_bin) > 1:
                    dx = np.diff(np.array(x_bin))
                    dy = np.diff(np.array(y_bin))
                    distances_in_bin_pixels = np.sqrt(dx**2 + dy**2)
                    total_distance_pixels_in_bin = np.sum(distances_in_bin_pixels)
                    
                    if self.ppm is not None and self.ppm > 0:
                        distance_cm_in_bin = (total_distance_pixels_in_bin / self.ppm) * 100
                    else:
                        distance_cm_in_bin = total_distance_pixels_in_bin
                
                distances_per_bin.append(distance_cm_in_bin)
                cumulative_sum += distance_cm_in_bin
                cumulative_distances.append(cumulative_sum)
        
        # Write CSV
        header = ["Time Bin (s)", "Distance Per Bin (cm)", "Cumulative Distance (cm)"] + band_labels + percent_labels
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            actual_duration = float(pos_t[-1])
            max_full_chunks = int(actual_duration / self.chunk_size)
            num_rows = min(len(pos_t), max_full_chunks)
            
            # Gather per-band arrays
            band_arrays = {}
            for key, label in bands:
                arr = np.array(chunk_powers_data.get(key, [])).reshape(-1)
                band_arrays[label] = arr
            
            for i in range(num_rows):
                time_bin_start = i * self.chunk_size
                time_bin_end = (i + 1) * self.chunk_size
                time_bin_str = f"{time_bin_start}-{time_bin_end}"
                
                row = [time_bin_str]
                
                if distances_per_bin and i < len(distances_per_bin):
                    row.append(round(distances_per_bin[i], 3))
                else:
                    row.append("")
                
                if cumulative_distances and i < len(cumulative_distances):
                    row.append(round(cumulative_distances[i], 3))
                else:
                    row.append("")
                
                band_values = []
                for _, label in bands:
                    val = band_arrays[label][i] if i < len(band_arrays[label]) else ""
                    band_values.append(val)
                
                row.extend(band_values)
                
                numeric_vals = [float(v) for v in band_values if v != ""]
                total_power = sum(numeric_vals)
                
                for v in band_values:
                    if v == "" or total_power == 0:
                        row.append("")
                    else:
                        row.append(round((float(v) / total_power) * 100.0, 3))
                
                writer.writerow(row)

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
        
        timeSlider_Label = QLabel()
        ppm_Label = QLabel()
        chunkSize_Label = QLabel()
        speed_Label = QLabel()
        window_Label = QLabel()
        session_Label = QLabel()
        self.timeInterval_Label = QLabel()
        self.session_Text = QLabel()
        self.progressBar_Label = QLabel()
        self.power_Label = QTextEdit()
        self.power_Label.setReadOnly(True)
        self.power_Label.setWordWrapMode(True)
        self.power_Label.setMaximumHeight(150)
        self.frequencyViewer_Label = QLabel()
        self.graph_Label = QLabel()
        self.tracking_Label = QLabel()
        
        ppmTextBox = QLineEdit(self)
        chunkSizeTextBox = QLineEdit(self)
        speedTextBox = QLineEdit()
        quit_button = QPushButton('Quit', self)
        browse_button = QPushButton('Browse file', self)
        browse_folder_button = QPushButton('Browse folder', self)
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
        session_Label.setText("Current session")
        self.frequencyViewer_Label.setText("Frequency map")
        self.graph_Label.setText("Power spectrum graph")
        self.tracking_Label.setText("Animal tracking")
        self.bar.setOrientation(Qt.Vertical)
        self.render_button.setStyleSheet("background-color : light gray")
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
        
        # Set fixed width and height for power display to fit all bands on single lines
        self.power_Label.setFixedWidth(380)
        self.power_Label.setFixedHeight(220)
        
        # Resize widgets to fixed width
        resizeWidgets = [windowTypeBox, chunkSizeTextBox, speedTextBox, ppmTextBox, 
                         browse_button, browse_folder_button]
        for widget in resizeWidgets:
            widget.setFixedWidth(300)
        
        # Set width of righthand buttons
        quit_button.setFixedWidth(150)
        save_button.setFixedWidth(150)
        self.render_button.setFixedWidth(150)

        # Placing widgets
        # Swap positions: Browse file (left), Browse folder (right)
        self.layout.addWidget(browse_button, 0, 0)
        self.layout.addWidget(browse_folder_button, 0, 1)
        self.layout.addWidget(quit_button, 0, 2, alignment=Qt.AlignRight)
        self.layout.addWidget(session_Label, 1, 0)
        self.layout.addWidget(self.session_Text, 1, 1)
        self.layout.addWidget(self.render_button, 1, 2, alignment=Qt.AlignRight)
        self.layout.addWidget(save_button, 2, 2, alignment=Qt.AlignRight)
        self.layout.addWidget(window_Label, 2, 0)
        self.layout.addWidget(windowTypeBox, 2, 1)
        self.layout.addWidget(ppm_Label, 3, 0)
        self.layout.addWidget(ppmTextBox, 3, 1)
        self.layout.addWidget(chunkSize_Label, 4, 0)
        self.layout.addWidget(chunkSizeTextBox, 4, 1)
        self.layout.addWidget(speed_Label, 5, 0)
        self.layout.addWidget(speedTextBox, 5, 1)
        self.layout.addWidget(self.graph_mode_button, 6, 0)
        self.layout.addWidget(self.frequencyViewer_Label, 6, 1)
        self.layout.addWidget(self.graph_Label, 6, 1)
        self.layout.addWidget(self.tracking_Label, 6, 2)
        self.layout.addWidget(self.power_Label, 7, 0)
        self.layout.addWidget(self.view, 7, 1)
        self.layout.addWidget(self.graph_canvas, 7, 1)
        self.layout.addWidget(self.tracking_canvas, 7, 2)
        self.layout.addWidget(self.bar, 7, 3)
        self.layout.addWidget(timeSlider_Label, 8, 0)
        self.layout.addWidget(self.slider, 8, 1)
        self.layout.addWidget(self.timeInterval_Label, 8, 3)
        self.layout.addWidget(self.progressBar_Label, 9, 3)
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
        browse_folder_button.clicked.connect(self.browseFolderClicked)
        self.graph_mode_button.clicked.connect(self.switch_graph)
        save_button.clicked.connect(self.saveClicked)
        self.render_button.clicked.connect(self.runSession)
        self.slider.valueChanged[int].connect(self.sliderChanged)
        windowTypeBox.activated[str].connect(self.windowChanged)
        
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
        # Open file dialog with file filter (include numbered variants)
        file_filter = (
            "EEG/EGF Files (*.eeg *.eeg2 *.eeg3 *.eeg4 *.egf *.egf2 *.egf3 *.egf4 *.pos);;"
            "EEG Files (*.eeg *.eeg2 *.eeg3 *.eeg4);;"
            "EGF Files (*.egf *.egf2 *.egf3 *.egf4);;"
            "Position Files (*.pos);;"
            "All Files (*)"
        )
        files, _ = QFileDialog.getOpenFileNames(self, "Choose .eeg/.egf file", self.active_folder, file_filter, options=options)
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
    
    def updatePowerDisplay(self, chunk_index):
        '''
            Update the power label to show all frequency bands and their 
            percentages for the current time chunk.
        '''
        if self.chunk_powers_data is None:
            return
        
        # Define frequency bands in order
        freq_bands = ['Delta', 'Theta', 'Beta', 'Low Gamma', 'High Gamma', 'Ripple', 'Fast Ripple']
        
        # Calculate total power across all bands for this chunk
        total_power = 0
        band_powers = {}
        for band in freq_bands:
            if band in self.chunk_powers_data and chunk_index < len(self.chunk_powers_data[band]):
                band_powers[band] = self.chunk_powers_data[band][chunk_index][0]
                total_power += band_powers[band]
        
        # Build the display text with all bands and their percentages
        if total_power > 0:
            display_text = "<b>Power Distribution:</b><br>"
            for band in freq_bands:
                if band in band_powers:
                    percentage = (band_powers[band] / total_power) * 100
                    # Highlight the currently selected band
                    if band == self.frequencyBand:
                        display_text += f"<b>\u2192 {band}: {percentage:.2f}%</b><br>"
                    else:
                        display_text += f"&nbsp;&nbsp;&nbsp;{band}: {percentage:.2f}%<br>"
            self.power_Label.setHtml(display_text)
        else:
            self.power_Label.setPlainText("No data available")
    
    # ------------------------------------------- #
    
    def quitClicked(self):

        '''
            Application exit
        '''

        print('quit')
        QApplication.quit()
        self.close() 
    
    # ------------------------------------------- #
    
    def browseFolderClicked(self):
        
        '''
            Opens folder selection dialog and initiates batch processing
            of all EGF/EEG files in the selected folder.
        '''
        
        # Prepare error dialog window 
        self.error_dialog = QErrorMessage()
        
        # Error checking ppm 
        if self.ppm == None or self.chunk_size == None: 
            self.error_dialog.showMessage('PPM field and/or Chunk Size field is blank, or has a non-numeric input. Please enter appropriate numerics.')
            return
        
        # If speed input only specifies lower bound, set upperbound to default
        if (self.speed_lowerbound != None and self.speed_upperbound == None):
            self.speed_upperbound = 100
            
        # If speed filter text is left blank, set default to 0cms to 100cms
        if self.speed_lowerbound == None and self.speed_upperbound == None: 
            self.speed_lowerbound = 0
            self.speed_upperbound = 100
        
        # Check speed bounds are ascending
        if self.speed_lowerbound != None and self.speed_upperbound != None:
            if self.speed_lowerbound > self.speed_upperbound: 
                self.error_dialog.showMessage('Speed filter range must be ascending. Lower speed first, higher speed after. Ex: 1,5')
                return
        
        # Open folder dialog
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Select folder containing EGF/EEG files", 
            self.active_folder,
            options=options
        )
        
        if not folder_path:
            return
        
        # Remember the selected folder
        self.active_folder = folder_path
        
        # Update session label
        self.session_Text.setText(f"Batch: {folder_path}")
        
        # Disable buttons during processing
        self.setButtonsEnabled(False)
        
        # Create and start batch worker thread
        self.batch_worker = BatchWorkerThread(
            folder_path,
            self.ppm,
            self.chunk_size,
            self.window_type,
            self.speed_lowerbound,
            self.speed_upperbound,
            output_dir=folder_path
        )
        
        # Connect signals
        self.batch_worker.progress_update.connect(self.updateBatchLabel)
        self.batch_worker.progress_value.connect(self.progressBar)
        self.batch_worker.finished_signal.connect(self.batchProcessingFinished)
        self.batch_worker.error_signal.connect(self.batchProcessingError)
        
        # Start processing
        self.batch_worker.start()
        self.progressBar_Label.setText("Batch processing in progress...")
    
    # ------------------------------------------- #
    
    def updateBatchLabel(self, text):
        '''Update progress label during batch processing'''
        self.progressBar_Label.setText(text)
        print(text)
    
    # ------------------------------------------- #
    
    def batchProcessingFinished(self, results):
        '''Handle completion of batch processing'''
        
        successful = results.get('successful', [])
        failed = results.get('failed', [])
        total = results.get('total', 0)
        
        # Re-enable buttons
        self.setButtonsEnabled(True)
        
        # Build summary message
        summary = f"Batch processing complete!\n\n"
        summary += f"Total files: {total}\n"
        summary += f"✓ Successful: {len(successful)}\n"
        
        if successful:
            summary += "\nProcessed files:\n"
            for fname in successful[:5]:  # Show first 5
                summary += f"  • {fname}\n"
            if len(successful) > 5:
                summary += f"  ... and {len(successful)-5} more"
        
        if failed:
            summary += f"\n✗ Failed: {len(failed)}\n"
            for fname, error in failed[:3]:  # Show first 3 errors
                summary += f"  • {fname}: {error[:50]}...\n"
            if len(failed) > 3:
                summary += f"  ... and {len(failed)-3} more"
        
        summary += f"\nCSV files saved to:\n{self.active_folder}"
        
        # Show summary dialog
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Batch Processing Complete")
        msg_box.setText(summary)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
        
        self.progressBar_Label.setText("Batch processing complete!")
        self.bar.setValue(100)
    
    # ------------------------------------------- #
    
    def batchProcessingError(self, error_msg):
        '''Handle errors during batch processing'''
        self.setButtonsEnabled(True)
        self.error_dialog = QErrorMessage()
        self.error_dialog.showMessage(f"Batch processing error: {error_msg}")
        self.progressBar_Label.setText("Batch processing error!")
    
    # ------------------------------------------- #
    
    def setButtonsEnabled(self, enabled):
        '''Enable/disable buttons during processing'''
        # Find all buttons and disable them
        for widget in self.findChildren(QPushButton):
            if widget.text() not in ['Quit']:  # Keep Quit button enabled
                widget.setEnabled(enabled)
    
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
                
                # Plot with modern styling
                self.graph_canvas.axes.plot(freq, pdf, linewidth=2, color='#2E86AB', alpha=0.9)
                self.graph_canvas.axes.fill_between(freq, pdf, alpha=0.3, color='#2E86AB')
                
                # Add frequency band shading with ranges in labels
                freq_bands = {
                    'Delta (1-3 Hz)': (1, 3, '#FF6B6B'),
                    'Theta (4-12 Hz)': (4, 12, '#4ECDC4'),
                    'Beta (13-20 Hz)': (13, 20, '#95E1D3'),
                    'Low Gamma (35-55 Hz)': (35, 55, '#F38181'),
                    'High Gamma (65-120 Hz)': (65, 120, '#AA96DA'),
                    'Ripple (80-250 Hz)': (80, 250, '#FCBAD3'),
                    'Fast Ripple (250-500 Hz)': (250, 500, '#A8D8EA')
                }
                for band_label, (low, high, color) in freq_bands.items():
                    self.graph_canvas.axes.axvspan(low, high, alpha=0.1, color=color, label=band_label)
                
                # Styling
                self.graph_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
                self.graph_canvas.axes.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=11, fontweight='bold')
                self.graph_canvas.axes.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                self.graph_canvas.axes.set_facecolor('#F8F9FA')
                self.graph_canvas.axes.spines['top'].set_visible(False)
                self.graph_canvas.axes.spines['right'].set_visible(False)
                self.graph_canvas.axes.legend(loc='upper right', fontsize=7, framealpha=0.9)
        # Else show freq map
        else:
            self.view.show()
            self.graph_canvas.close()
            self.graph_Label.close()
            self.frequencyViewer_Label.show()
            self.plot_flag = False
            self.graph_mode_button.setText("Graph mode")
        
    # ------------------------------------------- #  
    
    def updatePowerDisplay(self, chunk_index):
        '''
            Update the power label to show all frequency bands with ranges and their 
            percentages for the current time chunk.
        '''
        if self.chunk_powers_data is None:
            return
        
        # Define frequency bands with ranges
        freq_band_ranges = {
            'Delta': '1-3 Hz',
            'Theta': '4-12 Hz',
            'Beta': '13-20 Hz',
            'Low Gamma': '35-55 Hz',
            'High Gamma': '65-120 Hz',
            'Ripple': '80-250 Hz',
            'Fast Ripple': '250-500 Hz'
        }
        freq_bands = list(freq_band_ranges.keys())
        
        # Calculate total power across all bands for this chunk
        total_power = 0
        band_powers = {}
        for band in freq_bands:
            if band in self.chunk_powers_data and chunk_index < len(self.chunk_powers_data[band]):
                band_powers[band] = self.chunk_powers_data[band][chunk_index][0]
                total_power += band_powers[band]
        
        # Build the display text with all bands, ranges, and their percentages
        if total_power > 0:
            display_text = "<b>Power Distribution:</b><br>"
            for band in freq_bands:
                if band in band_powers:
                    percentage = (band_powers[band] / total_power) * 100
                    freq_range = freq_band_ranges[band]
                    display_text += f"{band} ({freq_range}): {percentage:.2f}%<br>"
            self.power_Label.setHtml(display_text)
        else:
            self.power_Label.setPlainText("No data available")
    
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
                
                # Plot with modern styling
                self.graph_canvas.axes.plot(freq, pdf, linewidth=2, color='#2E86AB', alpha=0.9)
                self.graph_canvas.axes.fill_between(freq, pdf, alpha=0.3, color='#2E86AB')
                
                # Add frequency band shading
                freq_bands = {
                    'Delta': (1, 3, '#FF6B6B'),
                    'Theta': (4, 12, '#4ECDC4'),
                    'Beta': (13, 20, '#95E1D3'),
                    'Low Gamma': (35, 55, '#F38181'),
                    'High Gamma': (65, 120, '#AA96DA'),
                    'Ripple': (80, 250, '#FCBAD3'),
                    'Fast Ripple': (250, 500, '#A8D8EA')
                }
                y_max = pdf.max() * 1.1
                for band_name, (low, high, color) in freq_bands.items():
                    self.graph_canvas.axes.axvspan(low, high, alpha=0.1, color=color, label=band_name)
                
                # Styling
                self.graph_canvas.axes.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
                self.graph_canvas.axes.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=11, fontweight='bold')
                self.graph_canvas.axes.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                self.graph_canvas.axes.set_facecolor('#F8F9FA')
                self.graph_canvas.axes.spines['top'].set_visible(False)
                self.graph_canvas.axes.spines['right'].set_visible(False)
                
                # Add legend
                self.graph_canvas.axes.legend(loc='upper right', fontsize=7, framealpha=0.9)
                
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
        
        # Update power display with current chunk's frequency band percentages
        self.updatePowerDisplay(value)
        
        # Update power display with current chunk's frequency band percentages
        self.updatePowerDisplay(value)
    
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
        
        # Slider limits
        self.slider.setMinimum(0)
        self.slider.setMaximum( len(self.images)-1 )
        self.slider.setSingleStep(1)
        
        # Initialize power display with first chunk
        self.chunk_index = 0
        self.updatePowerDisplay(0)
        
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
