# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:47:42 2021
@author: vajra
"""


import numpy as np
from math import floor
import core.filtering as filt

from core.ProcessingFunctions import (grab_chunks, tot_power_in_fband, 
                                      compute_freq_map, speed_bins, compute_scaling_and_tPowers, finder)
from scipy.signal import welch
from matplotlib import pyplot as plt
from PIL import Image, ImageQt
from matplotlib import cm
from core.Tint_Matlab import get_active_eeg, remEEGShift, ReadEEG, bits2uV, speed2D
from core.data_handlers import grab_terode_cut_position_files, grab_position_data

def initialize_fMap(self, files, ppm, chunk_size, window_type, low_speed, high_speed, **kwargs):
    
    # Validate corect files are chosen
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
    
    
    freq_ranges = {'Delta': np.array([1, 3]), 
                   'Theta': np.array([4, 12]),
                   'Beta': np.array([13, 20]),
                   'Low Gamma': np.array([35, 55]),
                   'High Gamma': np.array([65, 120])}
    
    freq_maps = dict()
    progress_indicator = 0
    for key, value in freq_ranges.items():
        self.signals.text_progress.emit("Computing mapping for " + key )
        maps = compute_freq_map(self, key, window_type, new_pos_x, new_pos_y, new_pos_t, fs, chunks, chunk_size, 
                                scaling_factor_perBand=scaling_factor_perBand, 
                                chunk_pows_perBand=chunk_pows_perBand)
        freq_maps[key] = maps
        progress_indicator += 1
        self.signals.progress.emit(progress_indicator*20)
        
    self.signals.text_progress.emit("Data loaded!")
    
    return freq_maps, plot_data, chosen_times, scaling_factor_crossband, chunk_pows_perBand