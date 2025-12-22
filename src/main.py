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
import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
from pathlib import Path
import subprocess

from PyQt5.QtWidgets import QMessageBox
from functools import partial
from initialize_fMap import initialize_fMap
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from core.data_loaders import grab_position_data
from core.processors.Tint_Matlab import speed2D
from core.processors.spectral_functions import (
    speed_bins,
    export_binned_analysis_to_csv,
    visualize_binned_analysis,
    visualize_binned_analysis_by_chunk,
    visualize_binned_occupancy_and_dominant
)
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
        '''Export processing results to CSV and binned analysis'''
        # Support legacy (6-tuple) and new (7-tuple with binned_data)
        if len(result) == 6:
            freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_powers_data, tracking_data = result
            binned_data = None
        else:
            freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_powers_data, tracking_data, binned_data = result
        
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

        # Export binned analysis (Excel + JPG visualization) if available
        try:
            if binned_data is not None:
                output_prefix = os.path.join(self.output_dir, f"{base_name}_binned")
                export_result = export_binned_analysis_to_csv(binned_data, output_prefix)
                if export_result and export_result.get('format') == 'csv' and export_result.get('reason') == 'openpyxl_not_installed' and hasattr(self, 'progress_update'):
                    self.progress_update.emit("  ⚠ Excel export unavailable (openpyxl missing). CSV fallback used.")
                # Export binned heatmap as JPG instead of PNG
                viz_path = f"{output_prefix}_heatmap.jpg"
                visualize_binned_analysis(binned_data, save_path=viz_path)
        except Exception as e:
            # Non-fatal error: report in progress area
            if hasattr(self, 'progress_update'):
                self.progress_update.emit(f"  ⚠ Binned export failed: {str(e)}")

# =========================================================================== #

class BinnedAnalysisWindow(QDialog):
    '''
        Separate window for binned analysis functionality with dynamic visualization.
        Displays frequency band power, occupancy, and dominant band heatmaps.
    '''
    
    def __init__(self, parent=None, binned_data=None, files=None, active_folder=None):
        super().__init__(parent)
        self.binned_data = binned_data
        self.files = files or [None, None]
        self.active_folder = active_folder or os.getcwd()
        self.current_chunk = 0
        self.show_percent_power = False
        self.setWindowTitle("Binned Analysis Studio")
        self.setGeometry(100, 100, 1400, 900)
        self.initUI()
    
    def initUI(self):
        '''Initialize the binned analysis window UI'''
        main_layout = QVBoxLayout(self)
        
        # Top section: Title
        title_label = QLabel("4×4 Binned Frequency Analysis Studio")
        title_label.setFont(QFont("Times New Roman", 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Time chunk slider
        slider_label = QLabel("Time Chunk:")
        control_layout.addWidget(slider_label)
        
        self.chunk_slider = QSlider(Qt.Horizontal)
        self.chunk_slider.setMinimum(0)
        if self.binned_data:
            self.chunk_slider.setMaximum(max(0, self.binned_data['time_chunks'] - 1))
        self.chunk_slider.setSingleStep(1)
        self.chunk_slider.setValue(0)
        self.chunk_slider.setMaximumWidth(300)
        self.chunk_slider.valueChanged.connect(self.onChunkChanged)
        control_layout.addWidget(self.chunk_slider)
        
        self.chunk_display_label = QLabel("0 / 1")
        self.chunk_display_label.setMinimumWidth(60)
        control_layout.addWidget(self.chunk_display_label)
        
        # Power mode toggle
        control_layout.addSpacing(20)
        self.power_mode_btn = QPushButton("Switch to % Power", self)
        self.power_mode_btn.setCheckable(True)
        self.power_mode_btn.setChecked(False)
        self.power_mode_btn.toggled.connect(self.onPowerModeToggled)
        self.power_mode_btn.setMaximumWidth(150)
        control_layout.addWidget(self.power_mode_btn)
        
        # Buttons
        control_layout.addSpacing(20)
        save_pngs_btn = QPushButton("Export All JPGs", self)
        save_pngs_btn.clicked.connect(self.exportAllPngs)
        control_layout.addWidget(save_pngs_btn)
        
        export_data_btn = QPushButton("Export Data (Excel)", self)
        export_data_btn.clicked.connect(self.exportData)
        control_layout.addWidget(export_data_btn)
        
        control_layout.addStretch()
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.close)
        control_layout.addWidget(close_btn)
        
        main_layout.addLayout(control_layout)
        
        # Display areas with scroll
        display_layout = QHBoxLayout()
        
        # Left column: Mean power
        left_layout = QVBoxLayout()
        left_label = QLabel("Frequency Band Power")
        left_label.setFont(QFont("", 11, QFont.Bold))
        left_layout.addWidget(left_label)
        
        self.power_scroll = QScrollArea(self)
        self.power_scroll.setWidgetResizable(True)
        self.power_label = QLabel()
        self.power_label.setAlignment(Qt.AlignCenter)
        self.power_scroll.setWidget(self.power_label)
        left_layout.addWidget(self.power_scroll)
        display_layout.addLayout(left_layout, 2)
        
        # Right column: Occupancy and Dominant Band (stacked)
        right_layout = QVBoxLayout()
        right_label = QLabel("Occupancy & Dominant Band")
        right_label.setFont(QFont("", 11, QFont.Bold))
        right_layout.addWidget(right_label)
        
        self.occ_scroll = QScrollArea(self)
        self.occ_scroll.setWidgetResizable(True)
        self.occ_label = QLabel()
        self.occ_label.setAlignment(Qt.AlignCenter)
        self.occ_scroll.setWidget(self.occ_label)
        right_layout.addWidget(self.occ_scroll)
        display_layout.addLayout(right_layout, 1)
        
        main_layout.addLayout(display_layout, 1)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
        # Initial render
        if self.binned_data:
            self.renderChunkViews()
    
    def getOutputFolder(self):
        '''Get or create the binned analysis output folder'''
        out_dir = self.active_folder
        binned_folder = os.path.join(out_dir, "binned_analysis_output")
        if not os.path.exists(binned_folder):
            os.makedirs(binned_folder)
        return binned_folder
    
    def onChunkChanged(self, value):
        '''Update chunk display and re-render on slider change'''
        self.current_chunk = value
        max_chunks = self.binned_data['time_chunks'] if self.binned_data else 1
        self.chunk_display_label.setText(f"{value} / {max_chunks - 1}")
        self.renderChunkViews()
    
    def onPowerModeToggled(self, checked):
        '''Toggle between absolute and percent power'''
        self.show_percent_power = checked
        mode = "% Power" if checked else "Absolute Power"
        self.power_mode_btn.setText(f"Switch to {'Absolute' if checked else '%'} Power")
        self.status_label.setText(f"Switched to {mode}")
        self.renderChunkViews()
    
    def renderChunkViews(self):
        '''Render all visualizations for current chunk (in-memory, no temp files)'''
        if self.binned_data is None:
            return
        
        try:
            # Render frequency band power heatmap in memory
            fig_power, axes_power = self._create_power_heatmap(self.current_chunk, self.show_percent_power)
            power_pixmap = self._fig_to_pixmap(fig_power)
            if not power_pixmap.isNull():
                if power_pixmap.width() > 1000:
                    power_pixmap = power_pixmap.scaledToWidth(1000, Qt.SmoothTransformation)
                self.power_label.setPixmap(power_pixmap)
            
            # Render occupancy and dominant band in memory
            fig_occ, _ = self._create_occupancy_heatmap(self.current_chunk)
            occ_pixmap = self._fig_to_pixmap(fig_occ)
            if not occ_pixmap.isNull():
                if occ_pixmap.width() > 600:
                    occ_pixmap = occ_pixmap.scaledToWidth(600, Qt.SmoothTransformation)
                self.occ_label.setPixmap(occ_pixmap)
            
            self.status_label.setText(f"Chunk {self.current_chunk} rendered")
        except Exception as e:
            self.status_label.setText(f"Error rendering views: {str(e)}")
    
    def _fig_to_pixmap(self, fig):
        '''Convert matplotlib figure to QPixmap without saving to disk'''
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), 'PNG')
        plt.close(fig)
        return pixmap
    
    def _create_power_heatmap(self, chunk_idx, show_percent=False):
        '''Create power heatmap figure (does not save)'''
        n_chunks = self.binned_data['time_chunks']
        if chunk_idx < 0 or chunk_idx >= n_chunks:
            chunk_idx = max(0, min(chunk_idx, n_chunks - 1))
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        power_type = "Percent Power" if show_percent else "Power"
        fig.suptitle(f'4x4 Spatial Bins - Chunk {chunk_idx} (Frequency Band {power_type})', 
                     fontsize=14, fontweight='bold')
        
        bands = self.binned_data['bands']
        
        # Get min/max across all chunks for consistent color scale
        vmin_all = {}
        vmax_all = {}
        for band in bands:
            timeseries_data = self.binned_data['bin_power_timeseries'][band]
            
            if show_percent:
                percent_data = np.zeros_like(timeseries_data)
                for t in range(timeseries_data.shape[2]):
                    for x in range(4):
                        for y in range(4):
                            total_power = sum(self.binned_data['bin_power_timeseries'][b][x, y, t] 
                                            for b in bands)
                            if total_power > 0:
                                percent_data[x, y, t] = (timeseries_data[x, y, t] / total_power) * 100
                vmin_all[band] = np.nanmin(percent_data)
                vmax_all[band] = np.nanmax(percent_data)
            else:
                vmin_all[band] = np.nanmin(timeseries_data)
                vmax_all[band] = np.nanmax(timeseries_data)
        
        # First row: First 4 bands
        for idx, band in enumerate(bands[:4]):
            chunk_power = self.binned_data['bin_power_timeseries'][band][:, :, chunk_idx]
            
            if show_percent:
                total_power = sum(self.binned_data['bin_power_timeseries'][b][:, :, chunk_idx] 
                                for b in bands)
                chunk_power = np.divide(chunk_power, total_power, where=total_power>0, 
                                       out=np.zeros_like(chunk_power)) * 100
            
            im = axes[0, idx].imshow(chunk_power, cmap='hot', aspect='auto',
                                     vmin=vmin_all[band], vmax=vmax_all[band])
            axes[0, idx].set_title(f'{band}')
            axes[0, idx].set_xticks([0, 1, 2, 3])
            axes[0, idx].set_yticks([0, 1, 2, 3])
            axes[0, idx].grid(True, alpha=0.3)
            cbar = plt.colorbar(im, ax=axes[0, idx])
            cbar.set_label('%' if show_percent else 'Power', fontsize=9)
        
        # Second row: Remaining bands
        remaining_bands = bands[4:]
        for idx, band in enumerate(remaining_bands):
            chunk_power = self.binned_data['bin_power_timeseries'][band][:, :, chunk_idx]
            
            if show_percent:
                total_power = sum(self.binned_data['bin_power_timeseries'][b][:, :, chunk_idx] 
                                for b in bands)
                chunk_power = np.divide(chunk_power, total_power, where=total_power>0, 
                                       out=np.zeros_like(chunk_power)) * 100
            
            im = axes[1, idx].imshow(chunk_power, cmap='hot', aspect='auto',
                                     vmin=vmin_all[band], vmax=vmax_all[band])
            axes[1, idx].set_title(f'{band}')
            axes[1, idx].set_xticks([0, 1, 2, 3])
            axes[1, idx].set_yticks([0, 1, 2, 3])
            axes[1, idx].grid(True, alpha=0.3)
            cbar = plt.colorbar(im, ax=axes[1, idx])
            cbar.set_label('%' if show_percent else 'Power', fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(remaining_bands), 4):
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def _create_occupancy_heatmap(self, chunk_idx):
        '''Create occupancy and dominant band heatmap figure (does not save)'''
        n_chunks = self.binned_data['time_chunks']
        if chunk_idx < 0 or chunk_idx >= n_chunks:
            chunk_idx = max(0, min(chunk_idx, n_chunks - 1))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left panel: Occupancy
        occupancy = self.binned_data['bin_occupancy']
        im1 = axes[0].imshow(occupancy, cmap='viridis', aspect='auto')
        axes[0].set_title('Bin Occupancy (Total Time Spent)')
        axes[0].set_xticks([0, 1, 2, 3])
        axes[0].set_yticks([0, 1, 2, 3])
        axes[0].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Occupancy (samples)', fontsize=9)
        
        # Right panel: Dominant band for specific chunk
        dominant_chunk = self.binned_data['bin_dominant_band'][chunk_idx]
        bands = self.binned_data['bands']
        band_map = {band: idx for idx, band in enumerate(bands)}
        numeric_dominant = np.zeros((4, 4))
        for x in range(4):
            for y in range(4):
                band = dominant_chunk[x, y]
                numeric_dominant[x, y] = band_map.get(band, 0)
        
        im2 = axes[1].imshow(numeric_dominant, cmap='tab10', aspect='auto', vmin=0, vmax=len(bands)-1)
        axes[1].set_title(f'Dominant Band - Chunk {chunk_idx}')
        axes[1].set_xticks([0, 1, 2, 3])
        axes[1].set_yticks([0, 1, 2, 3])
        axes[1].grid(True, alpha=0.3)
        
        # Add colorbar with band labels
        cbar2 = plt.colorbar(im2, ax=axes[1], ticks=range(len(bands)))
        cbar2.set_ticklabels(bands, fontsize=8)
        cbar2.set_label('Band', fontsize=9)
        
        plt.tight_layout()
        return fig, axes
    
    def updateBinnedData(self, binned_data, files=None, active_folder=None):
        '''Update binned data when new session is loaded'''
        self.binned_data = binned_data
        if files:
            self.files = files
        if active_folder:
            self.active_folder = active_folder
        
        # Update slider range
        if binned_data:
            max_chunks = binned_data['time_chunks']
            self.chunk_slider.setMaximum(max(0, max_chunks - 1))
            self.chunk_slider.setValue(0)
            self.current_chunk = 0
            self.chunk_display_label.setText(f"0 / {max_chunks - 1}")
            self.renderChunkViews()
        
        self.status_label.setText("Data updated. Ready for analysis.")
    
    def exportAllPngs(self):
        '''Export all JPG visualizations (mean, percent for all chunks; occupancy once; dominant band per chunk)'''
        if self.binned_data is None:
            QMessageBox.information(self, 'Export PNGs', 'No binned data available.')
            return
        
        try:
            self.status_label.setText("Exporting all JPGs...")
            QApplication.processEvents()
            
            output_folder = self.getOutputFolder()
            base_name = os.path.splitext(os.path.basename(self.files[1] or 'output'))[0]
            
            n_chunks = self.binned_data['time_chunks']
            export_count = 0
            
            # Export mean power for all chunks (JPG, quality 85)
            for chunk_idx in range(n_chunks):
                fig, _ = self._create_power_heatmap(chunk_idx, show_percent=False)
                jpg_path = os.path.join(output_folder, f"{base_name}_chunk{chunk_idx}_mean_power.jpg")
                fig.savefig(jpg_path, format='jpg', pil_kwargs={'quality': 85}, bbox_inches='tight')
                plt.close(fig)
                export_count += 1
            
            # Export percent power for all chunks (JPG, quality 85)
            for chunk_idx in range(n_chunks):
                fig, _ = self._create_power_heatmap(chunk_idx, show_percent=True)
                jpg_path = os.path.join(output_folder, f"{base_name}_chunk{chunk_idx}_percent_power.jpg")
                fig.savefig(jpg_path, format='jpg', pil_kwargs={'quality': 85}, bbox_inches='tight')
                plt.close(fig)
                export_count += 1
            
            # Export occupancy only once (JPG, quality 85) - use same colormap as GUI
            fig_occ = plt.figure(figsize=(6, 5))
            ax = fig_occ.add_subplot(111)
            occ = self.binned_data['bin_occupancy']
            im = ax.imshow(occ, cmap='viridis', aspect='auto')
            ax.set_title('Bin Occupancy (Total Time Spent)', fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2, 3])
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Occupancy (samples)', fontsize=9)
            jpg_path = os.path.join(output_folder, f"{base_name}_occupancy.jpg")
            fig_occ.savefig(jpg_path, format='jpg', pil_kwargs={'quality': 85}, bbox_inches='tight')
            plt.close(fig_occ)
            export_count += 1
            
            # Export dominant band per chunk (JPG, quality 85)
            for chunk_idx in range(n_chunks):
                fig, _ = self._create_occupancy_heatmap(chunk_idx)
                jpg_path = os.path.join(output_folder, f"{base_name}_chunk{chunk_idx}_dominant_band.jpg")
                fig.savefig(jpg_path, format='jpg', pil_kwargs={'quality': 85}, bbox_inches='tight')
                plt.close(fig)
                export_count += 1
            
            self.status_label.setText(f"✓ Exported {export_count} JPGs")
            QMessageBox.information(
                self,
                'Export Complete',
                f'Exported {export_count} JPG files to:\n{output_folder}\n\n'
                f'  • {n_chunks} mean power heatmaps\n'
                f'  • {n_chunks} percent power heatmaps\n'
                f'  • 1 occupancy heatmap (time-invariant)\n'
                f'  • {n_chunks} dominant band heatmaps\n\n'
                f'(JPG format for faster export & smaller file size)'
            )
        except Exception as e:
            self.status_label.setText(f"Export failed: {str(e)}")
            QMessageBox.warning(self, 'Export Error', f'Failed to export JPGs: {str(e)}')
    
    def exportData(self):
        '''Export binned analysis data to Excel files (or CSV fallback)'''
        if self.binned_data is None:
            QMessageBox.information(self, 'Export Data', 'No binned data available. Please run a session first.')
            return
        
        try:
            self.status_label.setText("Exporting data...")
            QApplication.processEvents()
            
            output_folder = self.getOutputFolder()
            base_name = os.path.splitext(os.path.basename(self.files[1] or 'output'))[0]
            result = export_binned_analysis_to_csv(self.binned_data, os.path.join(output_folder, f"{base_name}_binned"))
            self.status_label.setText(f"✓ Data exported ({result['format']})")
            file_list = "\n".join([f"  • {os.path.basename(p)}" for p in result['files']])
            if result.get('format') == 'csv' and result.get('reason') == 'openpyxl_not_installed':
                warn = QMessageBox(self)
                warn.setWindowTitle('Excel Export Unavailable')
                warn.setText('openpyxl not installed — exported CSV instead.\n\nInstall openpyxl to enable Excel export with multiple sheets.')
                install_btn = warn.addButton('Install openpyxl', QMessageBox.AcceptRole)
                warn.addButton('Skip', QMessageBox.RejectRole)
                warn.exec_()
                if warn.clickedButton() == install_btn:
                    try:
                        self.status_label.setText('Installing openpyxl...')
                        QApplication.processEvents()
                        proc = subprocess.run([sys.executable, '-m', 'pip', 'install', 'openpyxl'], capture_output=True, text=True)
                        if proc.returncode == 0:
                            QMessageBox.information(self, 'Installation Complete', 'openpyxl installed successfully. Re-exporting to Excel...')
                            # Re-run export to produce Excel files
                            result = export_binned_analysis_to_csv(self.binned_data, os.path.join(output_folder, f"{base_name}_binned"))
                            self.status_label.setText(f"✓ Data exported ({result['format']})")
                            file_list = "\n".join([f"  • {os.path.basename(p)}" for p in result['files']])
                        else:
                            QMessageBox.warning(self, 'Installation Failed', f'Could not install openpyxl.\n\n{proc.stderr[:500]}')
                    except Exception as ie:
                        QMessageBox.warning(self, 'Installation Error', f'Error installing openpyxl: {str(ie)}')
            QMessageBox.information(self, 'Export Complete',
                f'Binned analysis data exported to:\n{output_folder}\n\n'
                f'Format: {result["format"].upper()}\n\nFiles:\n{file_list}')
        except Exception as e:
            self.status_label.setText(f"Export failed: {str(e)}")
            QMessageBox.warning(self, 'Export Error', f'Failed to export data: {str(e)}')

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
        self.ppm = 511                          # Pixel per meter value
        self.chunk_size = 10                    # Size of each signal chunk in seconds (user defined)
        self.window_type = 'hann'               # Window type for welch 
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
        self.binned_analysis_window = None  # Reference to separate binned analysis window
        
        ppmTextBox = QLineEdit(self)
        chunkSizeTextBox = QLineEdit(self)
        speedTextBox = QLineEdit()
        quit_button = QPushButton('Quit', self)
        browse_button = QPushButton('Browse file', self)
        browse_folder_button = QPushButton('Browse folder', self)
        self.binned_analysis_btn = QPushButton('Binned Analysis Studio', self)
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
        windowTypeBox.addItem("hann")
        windowTypeBox.addItem("hamming")
        windowTypeBox.addItem("blackmanharris")
        windowTypeBox.addItem("boxcar")
        speedTextBox.setPlaceholderText("Ex: Type 5,10 for 5cms to 10cms range filter")
        ppmTextBox.setText("511")
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
        # Place binned analysis button near the bottom-right
        self.layout.addWidget(self.binned_analysis_btn, 9, 2, alignment=Qt.AlignRight)
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
        self.binned_analysis_btn.clicked.connect(self.openBinnedAnalysisWindow)
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
            Automatically saves Excel and CSV with average frequency powers vs time
            into the output folder with filename suffix '_SSM'.
        '''

        # If there are no chunk powers, do nothing
        if self.chunk_powers_data is None:
            return

        # Try to use Excel; fallback to CSV
        try:
            import openpyxl
            use_excel = True
        except ImportError:
            use_excel = False
            import csv

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
        
        if use_excel:
            out_file = os.path.join(out_dir, f"{base_name}_SSM.xlsx")
        else:
            out_file = os.path.join(out_dir, f"{base_name}_SSM.csv")

        # Build header and rows and write CSV or Excel
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
        
        # Prepare data rows
        data_rows = []
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
            data_rows.append(row)
        
        # Write Excel or CSV
        if use_excel:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = 'SSM Data'
            ws.append(header)
            for row in data_rows:
                ws.append(row)
            wb.save(out_file)
            QMessageBox.information(self, "Save Complete", f"Data saved to:\n{out_file}\n\nFormat: Excel")
        else:
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in data_rows:
                    writer.writerow(row)
            QMessageBox.information(self, "Save Complete", f"Data saved to:\n{out_file}\n\nFormat: CSV (openpyxl not installed)")
            
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

        # Support both legacy (6-tuple) and new (7-tuple with binned_data)
        self.freq_dict = data[0]
        self.plot_data = data[1]
        self.images = self.freq_dict[self.frequencyBand]
        self.pos_t = data[2]
        self.scaling_factor_crossband = data[3]
        self.chunk_powers_data = data[4]
        self.tracking_data = data[5]
        self.binned_data = data[6] if len(data) > 6 else None
        
        # Update the binned analysis window if it's open
        if self.binned_analysis_window is not None and self.binned_analysis_window.isVisible():
            self.binned_analysis_window.updateBinnedData(self.binned_data, self.files, self.active_folder)
        
        # Slider limits
        self.slider.setMinimum(0)
        self.slider.setMaximum( len(self.images)-1 )
        self.slider.setSingleStep(1)
        
        # Initialize power display with first chunk
        self.chunk_index = 0
        self.updatePowerDisplay(0)
        
        print("Data loaded")

    # ------------------------------------------- #
    
    def openBinnedAnalysisWindow(self):
        '''Open the separate binned analysis studio window'''
        if not hasattr(self, 'binned_data') or self.binned_data is None:
            QMessageBox.information(
                self, 
                'Binned Analysis Studio', 
                'No binned data available yet. Please run a session first.'
            )
            return
        
        # Create or show existing window
        if self.binned_analysis_window is None or not self.binned_analysis_window.isVisible():
            self.binned_analysis_window = BinnedAnalysisWindow(
                parent=self,
                binned_data=self.binned_data,
                files=self.files,
                active_folder=self.active_folder
            )
        
        self.binned_analysis_window.show()
        self.binned_analysis_window.raise_()
        self.binned_analysis_window.activateWindow()

    # ------------------------------------------- #
    
    def showBinnedAnalysis(self):
        '''Generate and display 4x4 binned analysis heatmap in a modal panel.'''
        if not hasattr(self, 'binned_data') or self.binned_data is None:
            QMessageBox.information(self, 'Binned Analysis', 'No binned analysis available yet. Please run a session first.')
            return
        try:
            # Save visualization to a temporary path in the active folder
            out_dir = self.active_folder or os.getcwd()
            base_name = os.path.splitext(os.path.basename(self.files[1] or 'output'))[0]
            viz_path = os.path.join(out_dir, f"{base_name}_binned_heatmap.png")
            visualize_binned_analysis(self.binned_data, save_path=viz_path)
            # Show in a modal dialog with responsive sizing
            dlg = QDialog(self)
            dlg.setWindowTitle('4x4 Binned Analysis Heatmap')
            vbox = QVBoxLayout(dlg)
            
            # Create a scroll area for responsive sizing
            scroll = QScrollArea(dlg)
            scroll.setWidgetResizable(True)
            img_label = QLabel()
            pix = QPixmap(viz_path)
            img_label.setPixmap(pix)
            img_label.setScaledContents(False)
            scroll.setWidget(img_label)
            vbox.addWidget(scroll)
            
            # Get available screen geometry and size dialog to fit within 70% of screen
            screen_geometry = QDesktopWidget().availableGeometry()
            max_width = int(screen_geometry.width() * 0.7)
            max_height = int(screen_geometry.height() * 0.7)
            # Ensure minimum reasonable size
            dlg_width = min(max_width, max(600, pix.width() + 50))
            dlg_height = min(max_height, max(400, pix.height() + 50))
            dlg.resize(dlg_width, dlg_height)
            dlg.exec_()
        except Exception as e:
            QMessageBox.warning(self, 'Binned Analysis', f'Failed to render binned analysis: {str(e)}')

    # ------------------------------------------- #
    
    def exportBinnedCsvs(self):
        '''Re-export binned CSVs on demand via the Options menu.'''
        if not hasattr(self, 'binned_data') or self.binned_data is None:
            QMessageBox.information(self, 'Export Binned CSVs', 'No binned analysis available yet. Please run a session first.')
            return
        try:
            out_dir = self.active_folder or os.getcwd()
            base_name = os.path.splitext(os.path.basename(self.files[1] or 'output'))[0]
            output_prefix = os.path.join(out_dir, f"{base_name}_binned")
            export_binned_analysis_to_csv(self.binned_data, output_prefix)
            QMessageBox.information(self, 'Export Binned CSVs', f'Exported binned CSVs to:\n{output_prefix}_*.csv')
        except Exception as e:
            QMessageBox.warning(self, 'Export Binned CSVs', f'Failed to export binned CSVs: {str(e)}')
        
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
