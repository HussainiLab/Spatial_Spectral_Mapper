# -*- coding: utf-8 -*-

"""
Created on Mon May 24 15:11:53 2021
@author: vajramsrujan
"""

import os
import mmap
import scipy
import numpy as np
import contextlib
import datetime

from .processors import filtering as filt
from .processors.Tint_Matlab import getpos, centerBox, remBadTrack, find_tetrodes, bits2uV

# ============================================================================== # 

def grab_position_data(pos_path: str, ppm: int) -> tuple: 
    
    '''
        Extracts position data and corrects bad tracking from .pos file

        Params: 
            pos_path (str): 
                The absolute path of the .pos file
            ppm (int): 
                Pixel per meter value

        Returns: 
            Tuple: pos_x,pos_y,pos_t,(pos_x_width,pos_y_width)
            --------
            pos_x, posy_pos_t: x,y coordinate, and time arrays respectively
            pos_x_width (float): max(pos_x) - min(pos_x)
            pos_y_width (float): max(pos_y) - min(pos_y)
    '''

    # Acquire position data
    pos_data = getpos(pos_path, ppm)
    
    # Correcting pos_t data in case of bad position file
    new_pos_t = np.copy(pos_data[2])
    if len(new_pos_t) != len(pos_data[0]): 
        while len(new_pos_t) != len(pos_data[0]):
            new_pos_t = np.append(new_pos_t, float(new_pos_t[-1] + 0.02))
    
    Fs_pos = pos_data[3]
    
    # Extract x,y coordinate, and t(ime) arrays
    pos_x = pos_data[0]
    pos_y = pos_data[1]
    pos_t = new_pos_t
    
    # Center pos_x and pos_y arrays to the middle of the arena 
    center = centerBox(pos_x, pos_y)
    pos_x = pos_x - center[0]
    pos_y = pos_y - center[1]
    
    # Remove any bad tracking
    pos_data_corrected = remBadTrack(pos_x, pos_y, pos_t, 2)
    pos_x = pos_data_corrected[0]
    pos_y = pos_data_corrected[1]
    pos_t = pos_data_corrected[2]  
    
    nonNanValues = np.where(np.isnan(pos_x) == False)[0]
    pos_t = pos_t[nonNanValues]
    pos_x = pos_x[nonNanValues]
    pos_y = pos_y[nonNanValues]
    
    # Boxcar smooth positioin data using convolution
    B = np.ones((int(np.ceil(0.4 * Fs_pos)), 1)) / np.ceil(0.4 * Fs_pos)
    pos_x = scipy.ndimage.convolve(pos_x, B, mode='nearest')
    pos_y = scipy.ndimage.convolve(pos_y, B, mode='nearest')
    
    # Grab width and height of 2D arena
    pos_x_width = max(pos_x) - min(pos_x)
    pos_y_width = max(pos_y) - min(pos_y)
    
    return pos_x,pos_y,pos_t,(pos_x_width,pos_y_width)
    
# =========================================================================== #

def get_output_filename(filename: str) -> str:
    
    '''
        Returns the output filename for the input eeg file,
        this is the name of the file that the data will be saved to.

        Params:
            filename (str):
                Name of the eeg file

        Returns:
            str: File name prefix with which the output files are labelled for saving. 
    '''

    # figure out what the extension is
    extension = os.path.splitext(filename)[-1]

    tint_basename = os.path.basename(os.path.splitext(filename)[0])

    return os.path.join(os.path.dirname(filename), 'spatialSpectral', tint_basename,
                        '%s_%s.h5' % (tint_basename, extension[1:]))

# =========================================================================== #

def grab_chunks(filename, notch=60, chunk_size=10, chunk_overlap=0):

    """
        In some cases we will have files that are too long to at once, this function will break up
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
                f_max = 600 #Abid: 4/16/2022
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

            ##################### calculate per chunk ######################
            
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
