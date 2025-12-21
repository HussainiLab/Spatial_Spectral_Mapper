# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:47:42 2021
@author: vajramsrujan
"""

import numpy as np

from math import floor
from matplotlib import cm
from PIL import Image, ImageQt
from scipy.signal import welch
from matplotlib import pyplot as plt
from core.data_loaders import grab_position_data, grab_chunks
from core.processors.Tint_Matlab import get_active_eeg, remEEGShift, ReadEEG, bits2uV, speed2D
from core.processors.spectral_functions import *

# =========================================================================== #

def initialize_fMap(self, files: list, ppm: int, chunk_size: int, window_type: str, 
                    low_speed: float, high_speed: float) -> tuple:
    
    '''
        Master function for data acquisition, validation, and frequency map compute. 
        
        Params:
            files (list):
                List of filepaths for eeg/egf and position file
            ppm (int):
                Pixel per meter value
            chunk_size (int):
                Size of chunks in seconds
            window_type 9str):
                Type of window for welch method. Choose between 'hamming','hann','backmanharris','boxcar'
            low_speed (flaot): 
                Lower bound for speed filtering 
            high_speed (float):
                Higher bound for speed filtering

        Returns:
            Tuple: freq_maps, plot_data, chosen_times, scaling_factor_crossband, chunk_pows_perBand
            --------
            freq_maps (dict):
                Dictionary containing a list of maps per freq band
            plot_data (np.ndarray):
                Array of power spectrum densities and frequencies per chunk
            chosen_times (np.ndarray):
                Array of times which match closest to the array of position times equally partitioned 
                using chunk_size. 
            scaling_factor_crossband (dict):
                Dictionary containing the percentage contribution of each band to the total 
                signal power.
            chunk_pows_perBand (dict):
                Dictionary containing array of chunk powers per band
    '''

    # Validate correct files are chosen
    for file in files:
        extension = file.split(sep='.')[1]
        if 'pos' in extension:
            pos_file = file
        else:
            if 'eeg' in extension:
                fs = 250
            elif 'egf' in extension:
                fs = 1200
            electrophys_file = file
    
    # Extract position data and filter with speed filter settings
    pos_x, pos_y, pos_t, arena_size =  grab_position_data(pos_file , ppm)
    pos_v = speed2D(pos_x, pos_y, pos_t)
    new_pos_x, new_pos_y, new_pos_t = speed_bins(low_speed, high_speed, pos_v, pos_x, pos_y, pos_t)
    
    # Chunk eeg data 
    chunks = grab_chunks(electrophys_file, notch=60, chunk_size=chunk_size, chunk_overlap=0)
    
    # Grab scaling factors. We compute this outside the loop since we only need it once. 
    scaling_factor_perBand, scaling_factor_crossband, chunk_pows_perBand, plot_data = compute_scaling_and_tPowers(self, window_type, pos_x, pos_y, pos_t, chunks, fs)
    
    # Choose indices from time vector that most closely match the chunk size time steps
    spaced_t = np.linspace(0, floor(new_pos_t[-1] / chunk_size), floor(new_pos_t[-1] / chunk_size)+1) * chunk_size
    common_indices = finder(new_pos_t, spaced_t)
    chosen_times = new_pos_t[common_indices]
    
    # Create a dictionary storing freq bands and corresponding freq ranges in Hz
    freq_ranges = {'Delta': np.array([1, 3]), 
                   'Theta': np.array([4, 12]),
                   'Beta': np.array([13, 20]),
                   'Low Gamma': np.array([35, 55]),
                   'High Gamma': np.array([65, 120]),
                   'Ripple': np.array([80, 250]), 
                   'Fast Ripple': np.array([250, 500])}
    
    # Progress indicator communicated to the main UI for progress bar
    progress_indicator = 0

    # Compute freq maps
    freq_maps = dict()
    for key, value in freq_ranges.items():
        self.signals.text_progress.emit("Computing mapping for " + key )
        maps = compute_freq_map(self, key, new_pos_x, new_pos_y, new_pos_t, fs, chunks, chunk_size, 
                                scaling_factor_perBand=scaling_factor_perBand, 
                                chunk_pows_perBand=chunk_pows_perBand)
        # Store maps using freq band name as keys
        freq_maps[key] = maps
        progress_indicator += 1
        self.signals.progress.emit(progress_indicator*20)
    
    pos_x_chunks, pos_y_chunks = compute_tracking_chunks(new_pos_x, new_pos_y, new_pos_t, chunk_size)
    self.signals.text_progress.emit("Data loaded!")
    
    return freq_maps, plot_data, chosen_times, scaling_factor_crossband, chunk_pows_perBand, (pos_x_chunks, pos_y_chunks)
