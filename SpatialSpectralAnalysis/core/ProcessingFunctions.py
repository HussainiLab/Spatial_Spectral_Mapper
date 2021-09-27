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
from scipy.integrate import simps
from scipy import signal
from scipy.signal import welch
from math import floor

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

def tot_power_in_fband(fft_xaxis, fft_yaxis, low_freq, high_freq): 
    
    # # Grab indices of values between freqw band of interest in the fft
    indices = np.where(  ((fft_xaxis >= low_freq) & (fft_xaxis <= high_freq)) )[0]
    fft_xaxis_band = fft_xaxis[indices]
    fft_yaxis_band = fft_yaxis[indices]
    
    # Computes the total power in band via integration
    tot_power = simps(fft_yaxis_band, x=fft_xaxis_band, axis=-1, even='avg')
    
    return tot_power

# =========================================================================== # 

def tPowers_Fband_chunked(chunks, fs, window, freq_low, freq_high):
    
    """
    Computes the total power per chunk of a chunked signal within
    a specific frequency band

    params: 
        chunks: List of numpy arrays representing the chunked signal
        fs: sampling frequency
        window: window type for pwelch (eg hann, blackmann...)
        freq_low / freq_high: Lower and upper bounds of specified frequency band of interest
        
    returns: 
        tot_chunk_pows: 
            array of total powers per chunk for specific freq band
        plot_data:
            array of psd's and frequencies per chunk'
    """
    
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

def compute_freq_map(self, freq_range, window, pos_x, pos_y, pos_t, fs, chunks, chunk_size, **kwargs):
    
    chunk_pows_perBand = kwargs.get('chunk_pows_perBand', None)
    scaling_factor_perBand = kwargs.get('scaling_factor_perBand', None)
    
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
    
    # Calcualte the new dimensions of the frequncy map
    resize_ratio = np.abs(max_x - min_x) / np.abs(max_y - min_y)
    base_resolution_scale = 256
    
    if resize_ratio > 1:
      x_resize = int(np.ceil (base_resolution_scale*resize_ratio)) 
      y_resize = base_resolution_scale
      
    elif resize_ratio < 1: 
        x_resize = base_resolution_scale
        y_resize = int(np.ceil (base_resolution_scale*(1/resize_ratio)))
        
    else: 
        x_resize = base_resolution_scale
        y_resize = base_resolution_scale
        
    row_values = np.linspace(min_x,max_x,256)
    column_values = np.linspace(min_y,max_y,256)
    
    # Choose indices from time vector that most closely match the chunk size time steps
    spaced_t = np.linspace(0, floor(pos_t[-1] / chunk_size), floor(pos_t[-1] / chunk_size)+1) * chunk_size
    common_indices = finder(pos_t, spaced_t)
    chosen_times = pos_t[common_indices]
    
    occ_fs = pos_t[1] - pos_t[0]
    occ_scaling_factor = len(pos_t) * ( pos_t[1] - pos_t[0] )
    maximum_value = index = 0
    maps = [None] * len(common_indices[1:])
    
    # Split up time vector into chunk sized intervals
    for i in range(len(common_indices[1:])):
        
        occ_map_raw = np.zeros((256,256))
        eeg_map_raw = np.zeros((256,256))
        
        for j in range(common_indices[i-1], common_indices[i]):         
            # Grab the row and column where the mice is based on timestamp
            row_index = np.abs(row_values - pos_x[j]).argmin()
            column_index = np.abs(column_values - pos_y[j]).argmin()
            eeg_map_raw[row_index][column_index] += chunk_pows_perBand[freq_range][np.abs(time_vec - pos_t[j]).argmin()]
            occ_map_raw[row_index][column_index] += occ_fs
            
        eeg_map_normalized = eeg_map_raw / (scaling_factor_perBand[freq_range])
        occ_map_normalized = occ_map_raw / occ_scaling_factor
        fMap = eeg_map_normalized / (occ_map_normalized)
        fMap[np.isnan(fMap)] = 0
        fMap_smoothed = np.rot90(cv2.filter2D(fMap,-1,kernel))
         
        if max(fMap_smoothed.flatten()) > maximum_value:
            maximum_value = max(fMap_smoothed.flatten())
        
        maps[index] = fMap_smoothed
        index+=1
      
    for i in range(len(maps)): 
        smoothed_and_normalized_map = (maps[i] / maximum_value)
        smoothed_and_normalized_map = cv2.resize(smoothed_and_normalized_map, dsize=(x_resize, y_resize), interpolation=cv2.INTER_CUBIC)
        PIL_Image = Image.fromarray(np.uint8(cm.jet(smoothed_and_normalized_map)*255))
        qimage = ImageQt.ImageQt(PIL_Image)
        
        maps[i] = QPixmap.fromImage(qimage)
        
    return maps
    
# =========================================================================== #

def compute_scaling_and_tPowers(self, window, pos_x, pos_y, pos_t, chunks, fs):
    
    # Smooth via kernel convolution
    kernlen = int(256*0.2)
    std = int(kernlen*0.2)
    kernel = gkern(kernlen,std)
    
    freq_ranges = {'Delta': np.array([1, 3]), 
                   'Theta': np.array([4, 12]),
                   'Beta': np.array([13, 20]),
                   'Low Gamma': np.array([35, 55]),
                   'High Gamma': np.array([65, 120])}
    
    scaling_factor_perBand = dict()
    # Vector that will help associate each chunk with the current time in pos_t
    time_vec = np.linspace(0,pos_t[-1], len(chunks))
    
    # Will store the total chunk powers per freq band
    chunk_pows_perBand = dict()
    
    ########### Compute the spectral density per chunk ###########
    # Grab chunks
    
    # Calculate a scaling ration by adding up all the power PER band. This will help us 
    # first normalize the eeg graphs per band. 
    for key, value in freq_ranges.items():
        self.signals.text_progress.emit("Computing " + key + " scaling")
        chunk_pows, plot_data = tPowers_Fband_chunked(chunks, fs, window, value[0], value[1])
        time_step = int(len(pos_t)/100)
        nSamples_perChunk = (time_vec[1] - time_vec[0])/(1/fs)
        total_powInBand = sum(chunk_pows) * nSamples_perChunk
        
        chunk_pows_perBand[key] = chunk_pows
        scaling_factor_perBand[key] = total_powInBand[0]
        
        print(key + ' scaling is done')
    self.signals.text_progress.emit("Scaling complete!")    

    # Calculate a scaling ratio as a proportion of the total power of all the bands in the signal
    # By this logic, Delta will make up x% of signal power, Beta = y% ...etc such that
    # the total ratio will add up to 100%. This will help us get a sense of relation between the signal bands
    sum_values = sum(scaling_factor_perBand.values())
    scaling_factor_crossband = dict()
    for key, value in freq_ranges.items():
        scaling_factor_crossband[key] = scaling_factor_perBand[key] / sum_values
    
        
    return scaling_factor_perBand, scaling_factor_crossband, chunk_pows_perBand, plot_data

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
    new_pos_x = pos_x[chosen_indices].flatten()
    new_pos_y = pos_y[chosen_indices].flatten()
    new_pos_t = pos_t[chosen_indices].flatten()
            
    return new_pos_x, new_pos_y, new_pos_t  

# =========================================================================== #  

# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# =========================================================================== #

# https://stackoverflow.com/questions/44526121/finding-closest-values-in-two-numpy-arrays

def finder(a, b):
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
