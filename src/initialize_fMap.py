# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:47:42 2021
@author: vajramsrujan
"""

import numpy as np

from math import floor, ceil
from matplotlib import cm
from scipy.signal import welch
from matplotlib import pyplot as plt
from core.data_loaders import grab_position_data, grab_chunks
from core.processors.Tint_Matlab import get_active_eeg, remEEGShift, ReadEEG, bits2uV, speed2D
from core.processors.spectral_functions import *

# =========================================================================== #

def detect_arena_shape(pos_x, pos_y):
    """
    Detects if the arena is circular or rectangular based on trajectory bounds.
    Returns a string describing the shape.
    """
    # Ensure numpy arrays
    x = np.array(pos_x)
    y = np.array(pos_y)
    
    if len(x) < 100:
        return "Unknown (insufficient data)"
        
    # Calculate bounds
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 or height == 0:
        return "Unknown (degenerate)"
        
    # Normalize coordinates to [-1, 1]
    # Center at 0
    norm_x = 2 * (x - min_x) / width - 1
    norm_y = 2 * (y - min_y) / height - 1
    
    # Calculate radius from center
    r = np.sqrt(norm_x**2 + norm_y**2)
    
    # Check 99th percentile of radius
    # For a circle, this should be close to 1.0
    # For a square/rectangle, this should be closer to sqrt(2) ≈ 1.41 (if corners are visited)
    r_99 = np.percentile(r, 99)
    
    # Aspect ratio
    aspect_ratio = width / height
    
    shape_type = ""
    if 0.85 < aspect_ratio < 1.15:
        if r_99 > 1.15:
            shape_type = "Square"
        else:
            shape_type = "Circle"
    else:
        if r_99 > 1.15:
            shape_type = "Rectangle"
        else:
            shape_type = "Ellipse"
            
    return f"{shape_type} (AR: {aspect_ratio:.2f}, R99: {r_99:.2f})"

def compute_polar_binned_analysis(pos_x, pos_y, pos_t, fs, chunks, chunk_size, chunk_pows_perBand, scaling_factor_perBand):
    """
    Computes binned analysis using polar coordinates (2 rings, 8 sectors = 16 bins).
    """
    # 1. Preprocess positions
    x = np.array(pos_x)
    y = np.array(pos_y)
    t = np.array(pos_t)
    
    # Center and normalize to [-1, 1]
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0: width = 1
    if height == 0: height = 1
    
    nx = 2 * (x - min_x) / width - 1
    ny = 2 * (y - min_y) / height - 1
    
    # Convert to polar
    r = np.sqrt(nx**2 + ny**2)
    theta = np.arctan2(ny, nx) # [-pi, pi]
    
    # 2. Define Bins
    # Rings: 0-0.5 (Inner), 0.5-inf (Outer)
    n_rings = 2
    r_bins = [0, 0.5, np.inf]
    r_indices = np.digitize(r, r_bins) - 1 
    r_indices = np.clip(r_indices, 0, n_rings - 1)
    
    # Sectors: 8 bins from -pi to pi
    n_sectors = 8
    theta_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    theta_indices = np.digitize(theta, theta_edges) - 1
    theta_indices = np.clip(theta_indices, 0, n_sectors - 1)
    
    # 3. Process Chunks
    n_chunks = len(chunks)
    bands = list(chunk_pows_perBand.keys())
    
    # Initialize outputs: Shape (Rings, Sectors, Time)
    bin_power_timeseries = {band: np.zeros((n_rings, n_sectors, n_chunks)) for band in bands}
    bin_occupancy = np.zeros((n_rings, n_sectors))
    bin_dominant_band = np.empty((n_chunks, n_rings, n_sectors), dtype=object)
    
    for i in range(n_chunks):
        t_start = i * chunk_size
        t_end = (i + 1) * chunk_size
        
        mask = (t >= t_start) & (t < t_end)
        if not np.any(mask):
            continue
            
        chunk_r_idx = r_indices[mask]
        chunk_th_idx = theta_indices[mask]
        
        # Count occupancy
        flat_indices = chunk_r_idx * n_sectors + chunk_th_idx
        counts = np.bincount(flat_indices, minlength=n_rings*n_sectors)
        chunk_occupancy = counts.reshape((n_rings, n_sectors))
        bin_occupancy += chunk_occupancy
        
        # Mask of visited bins in this chunk
        visited_mask = chunk_occupancy > 0
        
        # Distribute power
        for band in bands:
            if i < len(chunk_pows_perBand[band]):
                power_val = chunk_pows_perBand[band][i]
                if isinstance(power_val, (list, np.ndarray)):
                    power_val = np.mean(power_val)
                bin_power_timeseries[band][:, :, i] = visited_mask * power_val

    # Determine dominant band
    # (Simplified: just string label of max band)
    for i in range(n_chunks):
        for r_idx in range(n_rings):
            for s_idx in range(n_sectors):
                powers = {band: bin_power_timeseries[band][r_idx, s_idx, i] for band in bands}
                bin_dominant_band[i, r_idx, s_idx] = max(powers, key=powers.get) if sum(powers.values()) > 0 else "None"

    return {
        'type': 'polar',
        'time_chunks': n_chunks,
        'bands': bands,
        'bin_power_timeseries': bin_power_timeseries,
        'bin_occupancy': bin_occupancy,
        'bin_dominant_band': bin_dominant_band
    }

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
    
    print("  → Loading position data...")
    # Extract position data and filter with speed filter settings
    pos_x, pos_y, pos_t, arena_size =  grab_position_data(pos_file , ppm)
    
    # Detect arena shape
    arena_shape = detect_arena_shape(pos_x, pos_y)
    print(f"  → Detected Arena Shape: {arena_shape}")
    if hasattr(self, 'signals') and hasattr(self.signals, 'text_progress'):
        self.signals.text_progress.emit(f"Arena Shape: {arena_shape}")
    
    pos_v = speed2D(pos_x, pos_y, pos_t)
    new_pos_x, new_pos_y, new_pos_t = speed_bins(low_speed, high_speed, pos_v, pos_x, pos_y, pos_t)
    
    print("  → Chunking EEG data...")
    # Chunk eeg data 
    chunks = grab_chunks(electrophys_file, notch=60, chunk_size=chunk_size, chunk_overlap=0)
    
    # Pad chunks if position data extends beyond EEG data
    if len(new_pos_t) > 0 and len(chunks) > 0:
        expected_chunks = ceil(new_pos_t[-1] / chunk_size)
        if len(chunks) < expected_chunks:
            print(f"  → Padding EEG: {len(chunks)} chunks found, {expected_chunks} expected. Padding with zeros.")
            zero_chunk = np.zeros_like(chunks[0])
            for _ in range(expected_chunks - len(chunks)):
                chunks.append(zero_chunk)
    
    print("  → Computing scaling factors and powers...")
    # Grab scaling factors. We compute this outside the loop since we only need it once. 
    scaling_factor_perBand, scaling_factor_crossband, chunk_pows_perBand, plot_data = compute_scaling_and_tPowers(self, window_type, pos_x, pos_y, pos_t, chunks, fs)
    
    print("  → Aligning time bins...")
    # Choose indices from time vector that most closely match the chunk size time steps
    num_chunks = len(chunks)
    spaced_t = np.arange(0, num_chunks + 1, dtype=float) * float(chunk_size)
    if spaced_t[-1] < float(new_pos_t[-1]):
        spaced_t[-1] = float(new_pos_t[-1])

    boundaries = np.searchsorted(new_pos_t, spaced_t, side='left')
    boundaries[-1] = len(new_pos_t)
    boundaries = np.clip(boundaries, 0, len(new_pos_t))
    boundaries = np.maximum.accumulate(boundaries)
    chosen_times = new_pos_t[boundaries[:-1]]  # One timestamp per chunk boundary start
    
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
        print(f"  → Computing {key} frequency maps...")
        self.signals.text_progress.emit("Computing mapping for " + key )
        maps = compute_freq_map(self, key, new_pos_x, new_pos_y, new_pos_t, fs, chunks, chunk_size, 
                                scaling_factor_perBand=scaling_factor_perBand, 
                                chunk_pows_perBand=chunk_pows_perBand)
        # Store maps using freq band name as keys
        freq_maps[key] = maps
        progress_indicator += 1
        self.signals.progress.emit(progress_indicator*20)
    
    print("  → Computing tracking chunks...")
    pos_x_chunks, pos_y_chunks = compute_tracking_chunks(new_pos_x, new_pos_y, new_pos_t, chunk_size, n_chunks=len(chunks))
    # Compute binned analysis (multi-band, time-tracked)
    try:
        if "Circle" in arena_shape or "Ellipse" in arena_shape:
            self.signals.text_progress.emit("Computing polar binned analysis (16 bins)...")
            binned_data = compute_polar_binned_analysis(
                new_pos_x, new_pos_y, new_pos_t,
                fs=fs,
                chunks=chunks,
                chunk_size=chunk_size,
                chunk_pows_perBand=chunk_pows_perBand,
                scaling_factor_perBand=scaling_factor_perBand
            )
        else:
            self.signals.text_progress.emit("Computing 4x4 binned analysis...")
            binned_data = compute_binned_freq_analysis(
                new_pos_x, new_pos_y, new_pos_t,
                fs=fs,
                chunks=chunks,
                chunk_size=chunk_size,
                chunk_pows_perBand=chunk_pows_perBand,
                scaling_factor_perBand=scaling_factor_perBand
            )
        self.signals.text_progress.emit("Binned analysis complete!")
    except Exception as e:
        binned_data = None
        self.signals.text_progress.emit(f"Binned analysis skipped: {str(e)}")

    self.signals.text_progress.emit("Data loaded!")
    
    return freq_maps, plot_data, chosen_times, scaling_factor_crossband, chunk_pows_perBand, (pos_x_chunks, pos_y_chunks), binned_data, arena_shape
