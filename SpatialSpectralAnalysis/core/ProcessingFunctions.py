from .Tint_Matlab import bits2uV
import os
import numpy as np
import mmap
import contextlib
from . import filtering as filt
import datetime
from .data_handlers import get_output_filename
import cv2
from PIL import Image, ImageQt
from PyQt5.QtGui import QPixmap
from matplotlib import cm
from scipy import signal
from scipy.signal import welch

# =========================================================================== # 

def grab_chunks(filename, notch=60, chunk_size=10, chunk_overlap=0):
    """In some cases we will have files that are too long to at once, this function will break up
    the data into chunks and then downsample the frequency data into it's appropriate frequency bands
    as this will significantly decrease the memory usage (and is ultimately what we want anyways)

    -------------------------------inputs---------------------------------------------------------

    filename: full filepath to the .eeg or .egf file that you want to analyze

    notch: the frequency to notch filter (60 is generally the US value, 50 most likely in Europe)

    chunk_size: this is a size value the represents how many seconds of data you want to have analyzed per chunk

    chunk_overlap: the amount of overlap (half from the front, and half from the end)

    """
    
    # Initialize empty list for chunks
    chunks = []

    if chunk_overlap >= chunk_size:
        print(
            '[%s %s]: Chunk Overlap is too large, must be less than chunk_size!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8]))
        return

    if os.path.exists(get_output_filename(filename)):
        print(
            '[%s %s]: The output for this file already exists, skipping: %s!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], filename))
        return

    with open(filename, 'rb') as f:

        is_eeg = False
        if 'eeg' in filename:
            is_eeg = True

        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            # find the data_start
            start_index = int(m.find(b'data_start') + len('data_start'))  # start of the data
            stop_index = int(m.find(b'\r\ndata_end'))  # end of the data

            m = m[start_index:stop_index]
            
            ########## test ##########
            if is_eeg:
                Fs = 250
                f_max = 125

                # reading in the data
                m = np.fromstring(m, dtype='>b')
                m, scalar = bits2uV(m, filename)
            else:
                recorded_Fs = 4.8e3
                Fs = 1200  # this will be the downsampled sampling rate
                downsamp_factor = recorded_Fs / Fs
                f_max = f_max
                if f_max > Fs / 2:
                    f_max = Fs / 2

                # reading in the data
                m = np.fromstring(m, dtype='<h')
                m, scalar = bits2uV(m, filename)

                # filter before downsampling to avoid anti-aliasing
                m = filt.iirfilt(bandtype='low', data=m, Fs=recorded_Fs, Wp=Fs, order=6,
                                           automatic=0, filttype='butter', showresponse=0)

                # downsample the data so it only is 1.2 kHz instead of 4.8kHz
                m = m[0::int(downsamp_factor)]

            m = filt.dcblock(m, 0.1, Fs)  # removes DC Offset

            # removes 60 (or 50 Hz)
            m = filt.notch_filt(m, Fs, freq=notch, band=10, order=3)
            
            n_samples = int(len(m))

            chunk_overlap = int(chunk_overlap * Fs)  # converting from seconds to samples
            chunk_size = int(chunk_size * Fs)  # converting from seconds to samples

            concurrent_samples = chunk_size  # samples that we will analyze per chunk

            ######################### calculate ? per chunk ######################
            
            index_start = 0
            index_stop = index_start + concurrent_samples

            percentages = np.linspace(0.1, 1, 10)
            max_index = int(n_samples - 1)

            # Grab chunks
            while index_start < max_index:

                if index_stop > max_index:
                    index_stop = max_index
                    index_start = max_index - concurrent_samples

                percent_bool = np.where(index_stop / max_index >= percentages)[0]
                if len(percent_bool) >= 1:
                    print(
                        '[%s %s]: %d percent complete for the following file: %s!' %
                        (str(datetime.datetime.now().date()),
                         str(datetime.datetime.now().time())[:8], (int(100 * percentages[percent_bool[-1]])), filename))

                    try:
                        percentages = percentages[percent_bool[-1] + 1:]
                    except IndexError:
                        percentages = np.array([])

                current_indices = [index_start, index_stop]
                data = m[current_indices[0]:current_indices[1]]
                chunks.append(data)
                data = None
                
                if index_stop == max_index:
                    break
                
                index_start += concurrent_samples - chunk_overlap  # need the overlap since we took off the beginning
                index_stop = index_start + concurrent_samples

    return chunks

# =========================================================================== # 

def avg_power_in_fband(fft_xaxis, fft_yaxis, low_freq, high_freq): 
    
    # Grab indices of values between freqw band of interest in the fft
    indices = np.where(  ((fft_xaxis >= low_freq) & (fft_xaxis <= high_freq)) )[0]
    # Average them
    avgpow_in_band = np.mean(fft_yaxis[indices])
    
    return avgpow_in_band
 
# =========================================================================== #

def compute_freq_map(self, freq_range, window, pos_x, pos_y, pos_t, fs, chunks, chunk_size, bins, kernlen, std, **kwargs):
    
    scaling_factor = kwargs.get('scaling_factor', None)
    if scaling_factor == None: 
        scaling_factor = compute_scaling_factor( self, window, pos_x, pos_y, pos_t, chunks, fs, bins, kernlen, std)
    
    freq_ranges = {'Delta': np.array([1, 3]), 
                   'Theta': np.array([4, 12]),
                   'Beta': np.array([13, 20]),
                   'Low Gamma': np.array([35, 55]),
                   'High Gamma': np.array([65, 120])}
    
    freq_low = freq_ranges[freq_range][0]
    freq_high = freq_ranges[freq_range][1]
    
    # Smooth via kernel convolution
    kernel = gkern(kernlen,std)
    
    # Vector that will help associate each chunk with the current time in pos_t
    time_vec = np.linspace(0,pos_t[-1], len(chunks))
    
    ########### Compute the spectral density per chunk ###########
    
    # Grab each chunk
    avg_chunk_pows = []     # Stores the average spectrum power per chunk
    
    for data_chunk in chunks: 
        # Compute the spectral density per chunk using welch method
        # Compute no. samples per segment for welch fft
        overlap = 0.5
        nPerSeg = len(data_chunk)
        nOverlap = np.round(overlap * nPerSeg)
        freq, psd = welch(data_chunk, fs=fs, window=window, 
                nperseg=nPerSeg, noverlap=nOverlap, detrend="constant", 
                return_onesided=True, scaling="density")
        
        # Compute average power within specified spectrum band per chunk
        avg_chunk_pows.append(avg_power_in_fband(freq, psd, freq_low, freq_high))

    # Min and max dimensions of arena for scaling 
    min_x = min(pos_x)
    max_x = max(pos_x)
    min_y = min(pos_y)
    max_y = max(pos_y)
    
    # Calcualte the new dimensions of the frequncy map
    resize_ratio = np.abs(max_x - min_x) / np.abs(max_y - min_y)
    base_resolution_scale = 1000
    
    if resize_ratio > 1:
      x_resize = int(np.ceil (base_resolution_scale*resize_ratio)) 
      y_resize = base_resolution_scale
      
    elif resize_ratio < 1: 
        x_resize = base_resolution_scale
        y_resize = int(np.ceil (base_resolution_scale*(1/resize_ratio)))
        
    else: 
        x_resize = base_resolution_scale
        y_resize = base_resolution_scale
        
    row_values = np.linspace(min_x,max_x,bins)
    column_values = np.linspace(min_y,max_y,bins)
    
    maps = []
    time_step = int(len(pos_t)/100)
    
    for i in range(time_step, len(pos_t), time_step):
        
        ind_start = i - time_step
        ind_end = i
        
        pos_x_segment = pos_x[ ind_start:ind_end ]
        pos_y_segment = pos_y[ ind_start:ind_end ]
        pos_t_segment = pos_t[ ind_start:ind_end ]
        
        occ_map_LR, occ_map_HR = position_map(pos_x_segment, pos_y_segment, pos_t_segment, 100, 16, 5)
        eeg_map = np.zeros((bins,bins))
        
        for j in range(i - time_step, i):         
            # Grab the row and column where the mice is based on timestamp
            row_index = np.abs(row_values - pos_x[j]).argmin()
            column_index = np.abs(column_values - pos_y[j]).argmin()
            eeg_map[row_index][column_index] += avg_chunk_pows[np.abs(time_vec - pos_t[j]).argmin()]
        
        smoothed_eeg_map = np.rot90(cv2.filter2D(eeg_map,-1,kernel))
        
        freq_map = smoothed_eeg_map / (occ_map_LR + np.spacing(1))
        freq_map = cv2.filter2D(freq_map,-1,kernel)
        freq_map = freq_map / scaling_factor[freq_range]
        
        freq_map_highres = np.copy(freq_map)
        freq_map_highres = Image.fromarray(freq_map_highres)
        freq_map_highres = freq_map_highres.resize((x_resize,y_resize))
        freq_map_highres = np.array(freq_map_highres)
        freq_map[freq_map == np.nan] = 0
        freq_map_highres[freq_map_highres == np.nan] = 0
        
        PIL_Image = Image.fromarray(np.uint8(cm.jet(freq_map_highres)*255))
        qimage = ImageQt.ImageQt(PIL_Image)
        image = QPixmap.fromImage(qimage)
                
        maps.append(image)
        
    return maps, scaling_factor

def compute_scaling_factor(self, window, pos_x, pos_y, pos_t, chunks, fs, bins, kernlen, std):
    
    # Smooth via kernel convolution
    kernel = gkern(kernlen,std)
    
    freq_ranges = {'Delta': np.array([1, 3]), 
                   'Theta': np.array([4, 12]),
                   'Beta': np.array([13, 20]),
                   'Low Gamma': np.array([35, 55]),
                   'High Gamma': np.array([65, 120])}
    
    scaling_factor_dict = dict()
    # Vector that will help associate each chunk with the current time in pos_t
    time_vec = np.linspace(0,pos_t[-1], len(chunks))
    
    ########### Compute the spectral density per chunk ###########
    # Grab chunks
    
    for key, value in freq_ranges.items():
        self.signals.text_progress.emit("Computing " + key + " scaling")
        avg_chunk_pows = []     # Stores the average spectrum power per chunk
       
        for data_chunk in chunks: 
            # Compute the spectral density per chunk using welch method
            # Compute no. samples per segment for welch fft
            nSegments = 2 
            overlap = 0.5
            nPerSeg = np.round(len(data_chunk)//nSegments / overlap)
            nOverlap = np.round(overlap * nPerSeg)
            freq, psd = welch(data_chunk, fs=fs, window=window, 
                    nperseg=nPerSeg, noverlap=nOverlap, nfft=None, detrend="constant", 
                    return_onesided=True, scaling="density")
            
            # Compute average power within specified spectrum band oper chunk
            avg_chunk_pows.append(avg_power_in_fband(freq, psd, value[0], value[1]))
    
        # Min and max dimensions of arena for scaling 
        min_x = min(pos_x)
        max_x = max(pos_x)
        min_y = min(pos_y)
        max_y = max(pos_y)
        
        row_values = np.linspace(min_x,max_x,bins)
        column_values = np.linspace(min_y,max_y,bins)
    
        time_step = int(len(pos_t)/100)
        aboslute_maximum = 0
        for i in range(time_step, len(pos_t), time_step):
            
            ind_start = i - time_step
            ind_end = i
            
            pos_x_segment = pos_x[ ind_start:ind_end ]
            pos_y_segment = pos_y[ ind_start:ind_end ]
            pos_t_segment = pos_t[ ind_start:ind_end ]
            
            eeg_map = np.zeros((100,100))
            occ_map_LR, occ_map_HR = position_map(pos_x_segment, pos_y_segment, pos_t_segment, 100, 16, 5)
            
            for j in range(i - time_step, i): 
                # Grab the row and column where the mice is based on timestamp
                row_index = np.abs(row_values - pos_x[j]).argmin()
                column_index = np.abs(column_values - pos_y[j]).argmin()
                eeg_map[row_index][column_index] += avg_chunk_pows[np.abs(time_vec - pos_t[j]).argmin()]
            
            # Keep track of the largest value obtained amongst all the maps. We will use this for scaling.
            smoothed_eeg_map = np.rot90(cv2.filter2D(eeg_map,-1,kernel))
            freq_map = smoothed_eeg_map / (occ_map_LR + np.spacing(1))
            freq_map = cv2.filter2D(freq_map,-1,kernel)
            
            if max(freq_map.flatten()) > aboslute_maximum:
                aboslute_maximum = max(freq_map.flatten())
                
        
        scaling_factor_dict[key] = aboslute_maximum
        print(key + ' scaling is done')
        
    self.signals.text_progress.emit("Scaling complete!")        
    return scaling_factor_dict

# =========================================================================== #
def speed_bins(lower_speed, higher_speed, pos_v, pos_x, pos_y, pos_t): 
    
    """
        Selectively filters position values of subject travelling between
        specific speed limits. 
        
        Parameters: 
            lower_speed: Lower speed bound (cm/s) 
            higher_speed: Higher speed bound (cm/s) 
            pos_v: Array holding speed values of subject for entire experiment
            pos_x: X coordinate tracking of subject
            pos_y: Y coordinate tracking of subject
            pos_t: Timestamp tracking of subject
            
            Returns: 
                Tuple of new x,y coordinates and time array 
    """
        
    chosen_indices = np.where((pos_v >= lower_speed) & (pos_v <= higher_speed))
    new_pos_x = pos_x[chosen_indices]
    new_pos_y = pos_y[chosen_indices]
    new_pos_t = pos_t[chosen_indices]
            
    return new_pos_x, new_pos_y, new_pos_t  

# =========================================================================== # 
def position_map(pos_x, pos_y, pos_t, bins, kernlen, std):
    
    """
        Computes the position, or occupancy map. 
        
        Parameters: 
            pos_x, pos_y and pos_t: Arrays of the subjects x and y coordinates, as
            well as the time array respectively. 
            
            bins: The number of bins for the map (i.e resolution.)
            Setting the bin to 50 will produce a map witha  resolution of 50 X 50
            
            kernlen: The size of the kernel for smoothing the map. This can be adjusted.
            Typically the kernel is 10% the size of the bin. So for a bin of 50, we can set it to 5. 
            
            std: The standard deviation or 'spread' the kernel will use to smooth the map.
            The larger the std, the more 'strong' the spread of smoothing will be. 
            
            
            Returns: 
                Position map as a 2D array. 
                Plots the positon map as well. 
                
    """
    
    time_step = pos_t[1] - pos_t[0]
    min_x = min(pos_x)
    max_x = max(pos_x)
    min_y = min(pos_y)
    max_y = max(pos_y)
    
    # Calcualte the new dimensions of the frequncy map
    resize_ratio = np.abs(max_x - min_x) / np.abs(max_y - min_y)
    base_resolution_scale = 1000
    
    if resize_ratio > 1:
      x_resize = int(np.ceil (base_resolution_scale*resize_ratio)) 
      y_resize = base_resolution_scale
      
    elif resize_ratio < 1: 
        x_resize = base_resolution_scale
        y_resize = int(np.ceil (base_resolution_scale*(1/resize_ratio)))
        
    else: 
        x_resize = base_resolution_scale
        y_resize = base_resolution_scale
    
    pos_map = np.zeros((bins,bins))
    row_values = np.linspace(min_x,max_x,bins)
    column_values = np.linspace(min_y,max_y,bins)

    for i in range(len(pos_t)): 
        row_index = np.abs(row_values - pos_x[i]).argmin()
        column_index = np.abs(column_values - pos_y[i]).argmin()
        pos_map[row_index][column_index] += time_step
      
    kernel = gkern(kernlen,std)
    smoothed_pos_map = np.rot90(cv2.filter2D(pos_map,-1,kernel))
    smoothed_pos_map = smoothed_pos_map / max(smoothed_pos_map.flatten())
    occupancy_map   = smoothed_pos_map / max(smoothed_pos_map.flatten())
    occupancy_map = Image.fromarray(occupancy_map)
    occupancy_map = occupancy_map.resize((x_resize,y_resize))
    occupancy_map = np.array(occupancy_map)
    
    return smoothed_pos_map, occupancy_map   
 
# =========================================================================== #  

# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# =========================================================================== #