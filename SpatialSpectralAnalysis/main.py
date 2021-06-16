# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:47:42 2021
@author: vajra
"""

import numpy as np
import core.filtering as filt

from core.ProcessingFunctions import (grab_chunks, avg_power_in_fband, 
                                      compute_freq_map, compute_scaling_factor, speed_bins, position_map)
from scipy.signal import welch
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import cm
from core.Tint_Matlab import get_active_eeg, remEEGShift, ReadEEG, bits2uV, speed2D
from core.data_handlers import grab_terode_cut_position_files, grab_position_data

def compute_fMaps_for_eeg(files, ppm, chunk_size, window_type, low_speed, high_speed, **kwargs):
    
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
    
    scaling_factor = kwargs.get('scaling_factor', None)
    
    # Compute the occupancy map
    occ_map_LR, occ_map_HR = position_map(pos_x, pos_y, pos_t, 100, 16, 4)
        
    # Chunk eeg data 
    chunks = grab_chunks(electrophys_file, notch=60, chunk_size=chunk_size, chunk_overlap=0)
    
    freq_ranges = {'Delta': np.array([1, 3]), 
                   'Theta': np.array([4, 12]),
                   'Beta': np.array([13, 20]),
                   'Low Gamma': np.array([35, 55]),
                   'High Gamma': np.array([65, 120])}
    
    # Compute occupancy maps
    if scaling_factor is None: 
        scaling_factor = compute_scaling_factor(pos_x, pos_y, pos_t, chunks, fs, 100, 16, 4)
    
    freq_maps = []
    for key, value in freq_ranges.items():
        maps = compute_freq_map(key, pos_x, pos_y, pos_t, fs, chunks, chunk_size, 100, 16, 4, scaling_factor=scaling_factor)
        # colored_occupancy_map = Image.fromarray(np.uint8(cm.jet(freq_map_HR)*255))
        # colored_occupancy_map.save( r'C:\Users\vajra\Desktop\freq_maps' + '/' + key + '.png')
        
    return maps
# files = [r'C:\Users\vajra\Documents\data\20140815-behavior2-90/20140815-behavior2-90.pos', r'C:\Users\vajra\Documents\data\20140815-behavior2-90/20140815-behavior2-90.eeg']
# maps = compute_fMaps_for_eeg(files, 585, 30, 'hamming', 0, 100, scaling_factor=163156718.83084807)