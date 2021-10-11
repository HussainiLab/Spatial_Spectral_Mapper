
import os
import cv2
import numpy as np
import mmap
import contextlib

from PIL import Image, ImageQt
from PyQt5.QtGui import QPixmap
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy import signal
from scipy.signal import welch
from math import floor
from .Tint_Matlab import bits2uV
from ..data_loaders import get_output_filename

# =========================================================================== # 

def tot_power_in_fband(fft_xaxis: np.ndarray, fft_yaxis: np.ndarray, 
                       low_freq: float, high_freq: float) -> float: 
    
    '''
        Computes the total power within a specified frequency band of a signal

        Params:
            fft_axis (np.ndarray):
                fast fourier transform axis array of frequencies 
            fft_yaxis (np.ndarray):
                fast fourier transform amplitude array of signal
            low_freq, high_freq (float):
                Low and high end of frequency band respectively
        
        Returns:
            tot_power (np.float): Value of total power in specific band
    '''

    # Grab indices of values between freq band of interest in the fft
    indices = np.where(  ((fft_xaxis >= low_freq) & (fft_xaxis <= high_freq)) )[0]
    fft_xaxis_band = fft_xaxis[indices]
    fft_yaxis_band = fft_yaxis[indices]
    
    # Computes the total power in band via integration
    tot_power = simps(fft_yaxis_band, x=fft_xaxis_band, axis=-1, even='avg')
    
    return tot_power

# =========================================================================== # 

def tPowers_Fband_chunked(chunks: np.ndarray, fs: int, window: str, 
                          freq_low: float, freq_high: float) -> tuple:
    
    '''
        Computes the total power per chunk of a chunked signal within
        a specific frequency band

        Params: 
            chunks (np.ndarray): 
                List of numpy arrays representing the chunked signal
            fs (int): 
                Sampling frequency
            window (str): 
                Window type for pwelch (eg. hann, blackmann...)
            freq_low , freq_high (float): 
                Lower and upper bounds of specified frequency band of interest respectively
        
        Returns: 
            tuple: tot_chunk_pows, plot_data
            --------
            tot_chunk_pows (np.ndarray): 
                array of total powers per chunk for specific freq band
            plot_data (np.ndarray):
                array of power spectrum densities and frequencies per chunk
    '''
    
    # Instantiate empty arrays in preparation for plotting and 
    # consolidating total chunk powers
    plot_data = np.empty((len(chunks), 1), dtype=tuple)
    tot_chunk_pows = np.empty((len(chunks), 1), dtype=tuple)
    for i, data_chunk in enumerate(chunks): 
            # Compute the spectral density per chunk using welch method
            # Compute no. samples per segment for welch fft
            overlap = 0.5
            nPerSeg = len(data_chunk)
            nOverlap = np.round(overlap * nPerSeg)
            freq, psd = welch(data_chunk, fs=fs, window=window, 
                    nperseg=nPerSeg, noverlap=nOverlap, detrend="constant", 
                    return_onesided=True, scaling="density")
            
            # Compute average power within specified spectrum band per chunk
            plot_data[i][0] = (freq, psd) 
            tot_chunk_pows[i] = tot_power_in_fband(freq, psd, freq_low, freq_high)
            
    return tot_chunk_pows, plot_data
    
# =========================================================================== #

def compute_tracking_chunks(pos_x, pos_y, pos_t, chunk_size):
    
    '''
        Chunks the position data into 'n' second intervals based on chunk size.
        Used for plotting subject tracking

        Params:
            pos_x, pos_y, pos_t (np.ndarray) :
                x,y position data coordinates and timestamp arrays respectively
            chunk_size (int):
                Length of each chunk of position data in seconds

        Returns:
            Tuple: pos_x_chunks, pos_y_chunks
            --------
            pos_x_chunks (list):
                List of x coordinate data in the following format:
                    [ [x coordinates between 0s and n seconds], [x coordinates between 0 seconds and 2n seconds...] ...etc]
            pos_y_chunks (list):
                List of y coordinate data in the following format:
                    [ [y coordinates between 0s and n seconds], [y coordinates between 0 seconds and 2n seconds...] ...etc]
    '''

    # Create a new 'spaced out' time vector that equally partitions the time vector based on chunk size.
    spaced_t = np.linspace(0, floor(pos_t[-1] / chunk_size), floor(pos_t[-1] / chunk_size)+1) * chunk_size
    # Find all the common indices between the spaced time vector and the timestamps (i.e experiment time) vector. 
    common_indices = finder(pos_t, spaced_t)
    # Grab corresponding timestamps from the pos_t vector with the common indices
    chosen_times = pos_t[common_indices]

    # We will append elements to this list
    pos_x_chunks = []
    pos_y_chunks = []

    # Iterate through each 'time chunk', and append the tracking data per chunk
    for i in range(len(common_indices[1:])):
        pos_x_chunks.append(pos_x[:common_indices[i]])
        pos_y_chunks.append(pos_y[:common_indices[i]])

    return pos_x_chunks, pos_y_chunks

# =========================================================================== #

def compute_freq_map(self, freq_range: str, pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray, 
                     fs: int, chunks: list, chunk_size: int, **kwargs) -> list:
    
    '''
        Maps the frequency power in a specific band as a function of subject position

        Params:
            freq_range (str):
                Choose frequency band ('Delta', 'Theta', 'Beta', 'Low_Gamma', or 'High_Gamma')
            pos_x, pos_y, pos_t (np.ndarray): 
                x,y coordinates, and timestamp arrays respectively
            fs (int):
                Sampling frequency
            chunks (list):
                List containing 'n' second signal chunks
            chunk_size (int):
                Length of signal chunks in an integer number of seconds

        Kwargs*:
            scaling_factor_perBand (dict):
                Dictionary containing the normalization factor for each frequency band
            chunk_pows_perBand (dict):
                Dictionary containing the total power across the whole signal in each band

        Returns:
            List: maps
            ------
            A list of frequency maps, one map per signal chunk. This is for one frequency band. 
    '''

    # Grab kwargs
    chunk_pows_perBand = kwargs.get('chunk_pows_perBand', None)
    scaling_factor_perBand = kwargs.get('scaling_factor_perBand', None)
    
    # Set kernel and std block sizes to about 20% of the map sizes in preparation for convolutional smoothing.
    kernlen = int(256*0.2)
    std = int(kernlen*0.2)

    # Smooth via kernel convolution
    kernel = gkern(kernlen,std)
    
    # Vector that will help associate each chunk with the current time in pos_t
    time_vec = np.linspace(0,pos_t[-1], len(chunks))
    
    ########### Compute the spectral density per chunk ###########
    # Min and max dimensions of arena for scaling 
    min_x = min(pos_x)
    max_x = max(pos_x)
    min_y = min(pos_y)
    max_y = max(pos_y)
    
    # Calculate the new dimensions of the frequncy map
    resize_ratio = np.abs(max_x - min_x) / np.abs(max_y - min_y)
    # Ensure the largest dimension of the map is 256
    base_resolution_scale = 256
    
    # Adjust the map dimensions based on which side is longer or shorter
    if resize_ratio > 1:
      x_resize = int(np.ceil (base_resolution_scale*resize_ratio)) 
      y_resize = base_resolution_scale
      
    elif resize_ratio < 1: 
        x_resize = base_resolution_scale
        y_resize = int(np.ceil (base_resolution_scale*(1/resize_ratio)))
     
    # If the map is perfectly symmetrical, set both sides to 256
    else: 
        x_resize = base_resolution_scale
        y_resize = base_resolution_scale
    
    # Equally part the x and y dimensions of the map into 256 spatial bins
    row_values = np.linspace(min_x,max_x,256)
    column_values = np.linspace(min_y,max_y,256)
    
    # Create a new 'spaced out' time vector that equally partitions the time vector based on chunk size.
    spaced_t = np.linspace(0, floor(pos_t[-1] / chunk_size), floor(pos_t[-1] / chunk_size)+1) * chunk_size
    # Find all the common indices between the spaced time vector and the timestamps (i.e experiment time) vector. 
    common_indices = finder(pos_t, spaced_t)
    # Grab corresponding timestamps from the pos_t vector with the common indices
    chosen_times = pos_t[common_indices]
    
    # Compute a scaling factor for the occupancy map
    occ_fs = pos_t[1] - pos_t[0]
    occ_scaling_factor = len(pos_t) * ( pos_t[1] - pos_t[0] )
    maximum_value = index = 0
    # Initialize the number of maps based on the number of chunks
    maps = [None] * len(common_indices[1:])
    
    # Iterate through each 'time chunk', and generate the maps per chunk
    for i in range(len(common_indices[1:])):
        
        # Initialize empty occupancy and eeg maps
        occ_map_raw = np.zeros((256,256))
        eeg_map_raw = np.zeros((256,256))
        
        # Build the occupancy and eeg maps for the current time chunk
        for j in range(common_indices[i-1], common_indices[i]):         
            # Grab the row and column where the mice is based on timestamp
            row_index = np.abs(row_values - pos_x[j]).argmin()
            column_index = np.abs(column_values - pos_y[j]).argmin()
            
            # Encode the power of the band into whatever bins have been visited
            eeg_map_raw[row_index][column_index] += chunk_pows_perBand[freq_range][np.abs(time_vec - pos_t[j]).argmin()]
            # Encode the occupancy based on whatever bins have been visited
            occ_map_raw[row_index][column_index] += occ_fs
            
        # Normalize the maps using the scaling factors
        eeg_map_normalized = eeg_map_raw / (scaling_factor_perBand[freq_range])
        occ_map_normalized = occ_map_raw / occ_scaling_factor

        # Compute the frequency map by dividing the eeg map with the occupancy map.
        # This follows the same compute principle as a neural ratemap. Regions of high 
        # eeg activity and low occupancy will encode as significant activity.
        fMap = eeg_map_normalized / (occ_map_normalized)
        # Replace any Nans with zero in case zero divide is encountered
        fMap[np.isnan(fMap)] = 0
        # Rotate and smooth the map
        fMap_smoothed = np.rot90(cv2.filter2D(fMap,-1,kernel))
        
        # Keep track of the largest value encountered in the sequence of smoothed maps
        if max(fMap_smoothed.flatten()) > maximum_value:
            maximum_value = max(fMap_smoothed.flatten())
        
        maps[index] = fMap_smoothed

        index+=1
    
    # Finally, divide all the maps by the largest value encountered. This ensures that 
    # all the maps scale between 0 and 1, which is needed for plt.imshow function later on.
    # This scaling preserves the relative relationships within each map, while
    # also normalizing the maps relative to eachother. (i.e we can deduce differences in
    # frequency strengh between the maps by comparing their brightness's to eachother.) 
    for i in range(len(maps)): 
        smoothed_and_normalized_map = (maps[i] / maximum_value)
        smoothed_and_normalized_map = cv2.resize(smoothed_and_normalized_map, dsize=(x_resize, y_resize), interpolation=cv2.INTER_CUBIC)
        PIL_Image = Image.fromarray(np.uint8(cm.jet(smoothed_and_normalized_map)*255))
        qimage = ImageQt.ImageQt(PIL_Image)
        
        maps[i] = QPixmap.fromImage(qimage)
        
    return maps
    
# =========================================================================== #

def compute_scaling_and_tPowers(self, window: str, pos_x: np.ndarray, pos_y: np.ndarray, 
                                pos_t: np.ndarray, chunks: list, fs: int) -> tuple:
    
    '''
        Compute 2 sets of scaling factors. The first set helps normalize maps belonging to the same band. 
        The second helps compute what percentage each bands total power contributes to the entire signal. 

        Params:
            window (str):
                Type of window for fourier transform. Choose between 'hamming','hann','backmanharris','boxcar'
            pos_x, pos_y, pos_t (np.ndarray): 
                x,y coordinates, and timestamp arrays respectively
            chunks (list):
                List of chunked signals, each 'n' seconds long
            fs (int):
                Sampling frequency for Welch method. 

        Return:
            Tuple: scaling_factor_perBand, scaling_factor_crossband, chunk_pows_perBand, plot_data
            --------
            scaling_factor_perBand (dict):
                Dictionary containing the total power of each band, indexed using band name.
            scaling_factor_crossband (dict):
                Dictionary containing the percentage contribution of each band to the total 
                signal power.
            chunk_pows_perBand (dict):
                Dictionary containing array of chunk powers per band
            plot_data (np.ndarray):
                Array of power spectrum densities and frequencies per chunk
    '''

    # Smooth via kernel convolution
    # Set kernel and stdev block sizes to 20% map size
    kernlen = int(256*0.2)
    std = int(kernlen*0.2)
    kernel = gkern(kernlen,std)
    
    # Create dictionary defining frequency bands and corresponding ranges in Hz
    freq_ranges = {'Delta': np.array([1, 3]), 
                   'Theta': np.array([4, 12]),
                   'Beta': np.array([13, 20]),
                   'Low Gamma': np.array([35, 55]),
                   'High Gamma': np.array([65, 120])}
    
    # Dict to store total power per band
    scaling_factor_perBand = dict()
    # Vector that will help associate each chunk with the current time in pos_t
    time_vec = np.linspace(0,pos_t[-1], len(chunks))
    
    # Will store the total chunk powers per freq band
    chunk_pows_perBand = dict()
    
    ########### Compute the spectral density per chunk ###########
    
    # Grab chunks
    # Calculate a scaling factor by adding up all the power PER band. This will help us 
    # normalize the eeg graphs per band. 
    for key, value in freq_ranges.items():
        self.signals.text_progress.emit("Computing " + key + " scaling")
        # Compute chunk powers for each band
        chunk_pows, plot_data = tPowers_Fband_chunked(chunks, fs, window, value[0], value[1])
        # Using the number of samples in each chunk, compute the total power in each band. 
        nSamples_perChunk = (time_vec[1] - time_vec[0])/(1/fs)
        total_powInBand = sum(chunk_pows) * nSamples_perChunk
        
        chunk_pows_perBand[key] = chunk_pows
        scaling_factor_perBand[key] = total_powInBand[0]
        
        print(key + ' scaling is done')
    self.signals.text_progress.emit("Scaling complete!")    

    # Calculate a scaling factor as a proportion of the total power of all the bands in the signal
    # By this logic, Delta will make up x% of signal power, Beta = y% ...etc such that
    # the total ratio will add up to 100%. This will help us get a sense of relation between the signal bands
    sum_values = sum(scaling_factor_perBand.values())
    scaling_factor_crossband = dict()
    for key, value in freq_ranges.items():
        scaling_factor_crossband[key] = scaling_factor_perBand[key] / sum_values
    
    return scaling_factor_perBand, scaling_factor_crossband, chunk_pows_perBand, plot_data

# =========================================================================== #

def speed_bins(lower_speed: float, higher_speed: float, pos_v: np.ndarray, 
               pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray) -> tuple: 
    
    '''
        Selectively filters position values of subject travelling between
        specific speed limits. 
        
        Params: 
            lower_speed (float): 
                Lower speed bound (cm/s) 
            higher_speed (float): 
                Higher speed bound (cm/s) 
            pos_v (np.ndarray): 
                Array holding speed values of subject for entire experiment
            pos_x, pos_y, pos_t (np.ndarray): X, Y coordinates and timestamp array of subject respectively
            
            Returns: 
                Tuple: new_pos_x, new_pos_y, new_pos_t  
                --------
                New x,y coordinates and time array 
    '''
        
    chosen_indices = np.where((pos_v >= lower_speed) & (pos_v <= higher_speed))
    new_pos_x = pos_x[chosen_indices].flatten()
    new_pos_y = pos_y[chosen_indices].flatten()
    new_pos_t = pos_t[chosen_indices].flatten()
            
    return new_pos_x, new_pos_y, new_pos_t  

# =========================================================================== #  

# Credit:
# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

def gkern(kernlen, std):
    
    '''
        Returns a 2D Gaussian kernel array
    '''

    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# =========================================================================== #

# Credit:
# https://stackoverflow.com/questions/44526121/finding-closest-values-in-two-numpy-arrays

def finder(a, b):

    '''
         Finds the common indices between two arrays
    '''
    dup = np.searchsorted(a, b)
    uni = np.unique(dup)
    uni = uni[uni < a.shape[0]]
    ret_b = np.zeros(uni.shape[0])
    for idx, val in enumerate(uni):
        bw = np.argmin(np.abs(a[val]-b[dup == val]))
        tt = dup == val
        ret_b[idx] = np.where(tt == True)[0][bw]
    
    common_indices = np.column_stack((uni, ret_b))
    return common_indices[:,0].astype(int)