# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:11:53 2021

@author: vajra
"""
import numpy as np
from .Tint_Matlab import getpos, centerBox, remBadTrack, find_tetrodes, bits2uV
import scipy
import os
import contextlib
import mmap
from . import filtering as filt

def grab_position_data(pos_path, ppm): 
    
    pos_data = getpos(pos_path, ppm)
    
    # Correcting pos_t data in case of bad position file
    new_pos_t = np.copy(pos_data[2])
    if len(new_pos_t) != len(pos_data[0]): 
        while len(new_pos_t) != len(pos_data[0]):
            new_pos_t = np.append(new_pos_t, float(new_pos_t[-1] + 0.02))
    
    Fs_pos = pos_data[3]
    
    pos_x = pos_data[0]
    pos_y = pos_data[1]
    pos_t = new_pos_t
    
    center = centerBox(pos_x, pos_y)
    pos_x = pos_x - center[0]
    pos_y = pos_y - center[1]
    
    pos_data_corrected = remBadTrack(pos_x, pos_y, pos_t, 2)
    pos_x = pos_data_corrected[0]
    pos_y = pos_data_corrected[1]
    pos_t = pos_data_corrected[2]  
    
    nonNanValues = np.where(np.isnan(pos_x) == False)[0]
    pos_t = pos_t[nonNanValues]
    pos_x = pos_x[nonNanValues]
    pos_y = pos_y[nonNanValues]
    
    B = np.ones((int(np.ceil(0.4 * Fs_pos)), 1)) / np.ceil(0.4 * Fs_pos)
    pos_x = scipy.ndimage.convolve(pos_x, B, mode='nearest')
    pos_y = scipy.ndimage.convolve(pos_y, B, mode='nearest')
    
    pos_x_width = max(pos_x) - min(pos_x)
    pos_y_width = max(pos_y) - min(pos_y)
    
    return pos_x,pos_y,pos_t,(pos_x_width,pos_y_width)
    
# =========================================================================== #
def grab_terode_cut_position_files(files):
    
    pos_file = None
    cut_files = []
    tetrode_files = []
    
    if len(files) == 1:
        if files[0][-3:] == 'set':
            files = files[0]
            tetrode_files = find_tetrodes(files)
            cut_path, session = os.path.split(files)
            file_list = os.listdir(cut_path)
            cut_files = [os.path.join(cut_path, file) for file in file_list
                            if file[-3:] == 'cut' if session[:-4] in file]
            
            pos_file = [os.path.join(cut_path, file) for file in file_list
                            if file[-3:] == 'pos' if session[:-4] in file]
            
            if isinstance(pos_file, list): 
                pos_file = pos_file[0]
        else:
            raise NameError("You did not select a .set file")
            
    else: 
        for file in files: 
            if file[-3:] == 'pos': 
                pos_file = file
            elif file[-1:].isdigit():
                tetrode_files.append(file)
            elif file[-3:] == 'cut': 
                cut_files.append(file)
            else:
                raise NameError("One of the chosen files was not a tetrode, position, or cut file")
       
    if pos_file == None: 
        raise NameError("Position file was not chosen")
    
    if len(cut_files) == 0 or len(tetrode_files) == 0: 
        raise NameError("A cut file or tetrode file was not selected")
                
    return tetrode_files, cut_files, pos_file
    
def get_eeg_files(set_filename):

    """Get the EEG and EGF files"""

    directory = os.path.dirname(set_filename)

    try:
        dir_files = os.listdir(directory)
    except FileNotFoundError:
        return

    return [os.path.join(directory, file) for file in dir_files if 'eeg' in os.path.splitext(file)[-1] or
            'egf' in os.path.splitext(file)[-1] if os.path.splitext(os.path.basename(set_filename))[0] in file]

def get_output_filename(filename):
    """Returns the output filename for the input eeg file,
    this is the name of the file that the data will be saved to."""

    # figure out what the extension is
    extension = os.path.splitext(filename)[-1]

    tint_basename = os.path.basename(os.path.splitext(filename)[0])

    return os.path.join(os.path.dirname(filename), 'spatialSpectral', tint_basename,
                        '%s_%s.h5' % (tint_basename, extension[1:]))


def get_eeg_extensions(set_filename):
    """This will return the eeg extensions"""
    eeg_filenames = get_eeg_files(set_filename)
    try:
        return [os.path.splitext(file)[-1] for file in eeg_filenames
                if not os.path.exists(get_output_filename(file)) if file is not None]
    except TypeError:
        """If there is a file does not exist it will return a none"""
        return

def get_average(filename, Fs, notch):
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
                # reading in the data
                m = np.fromstring(m, dtype='>b')
                m, scalar = bits2uV(m, filename)
            else:
                recorded_Fs = 4.8e3
                Fs = 1200  # this will be the downsampled sampling rate

                # reading in the data
                m = np.fromstring(m, dtype='<h')
                m, scalar = bits2uV(m, filename)

                # filter before downsampling to avoid anti-aliasing
                m = filt.iirfilt(bandtype='low', data=m, Fs=recorded_Fs, Wp=Fs, order=6,
                                            automatic=0, filttype='butter', showresponse=0)

                # downsample the data so it only is 1.2 kHz instead of 4.8kHz
                m = m[0::int(recorded_Fs / Fs)]

            m = filt.dcblock(m, 0.1, Fs)  # removes DC Offset

            # removes 60 (or 50 Hz)
            m = filt.notch_filt(m, Fs, freq=notch, band=10, order=3)

    return np.mean(m)