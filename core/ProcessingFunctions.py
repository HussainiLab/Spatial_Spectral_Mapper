import datetime
from core.Tint_Matlab import *
import os, datetime
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from scipy import signal, interpolate, fftpack
import scipy
import numpy as np
from pyfftw.interfaces import scipy_fftpack as fftw
import mmap
import contextlib
import core.SignalProcessing as sp
import webcolors
import datetime


def find_sub(string, sub):
    '''finds all instances of a substring within a string and outputs a list of indices'''
    result = []
    k = 0
    while k < len(string):
        k = string.find(sub, k)
        if k == -1:
            return result
        else:
            result.append(k)
            k += 1  # change to k += len(sub) to not search overlapping results
    return result


def find_consec(data):
    '''finds the consecutive numbers and outputs as a list'''
    consecutive_values = []  # a list for the output
    current_consecutive = [data[0]]

    if len(data) == 1:
        return [[data[0]]]

    for index in range(1, len(data)):

        if data[index] == data[index - 1] + 1:
            current_consecutive.append(data[index])

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)

        else:
            consecutive_values.append(current_consecutive)
            current_consecutive = [data[index]]

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)
    return consecutive_values


def get_file_parameter(parameter, filename):
    if not os.path.exists(filename):
        return

    with open(filename, 'r+') as f:
        for line in f:
            if parameter in line:
                if line.split(' ')[0] == parameter:
                    # prevents part of the parameter being in another parameter name
                    new_line = line.strip().split(' ')
                    if len(new_line) == 2:
                        return new_line[-1]
                    else:
                        return ' '.join(new_line[1:])


def process_basename(self, set_filename):
    """This function will convert the .bin file to the TINT format"""

    frequency_boundaries = {
        'Delta': np.array([1, 3]),
        'Theta': np.array([4, 12]),
        # 'Alpha': np.array([8, 12]),
        'Beta': np.array([13, 20]),
        'Low Gamma': np.array([35, 55]),
        'High Gamma': np.array([65, 120]),
        'Ripple': np.array([80, 250]),
        'Fast Ripple': np.array([250, 500]),
    }

    directory = os.path.dirname(set_filename)

    tint_basename = os.path.basename(os.path.splitext(set_filename)[0])
    pos_filename = '%s.pos' % os.path.splitext(tint_basename)[0]

    # get eeg files
    eeg_files = get_eeg_files(set_filename)

    for filename in eeg_files:

        if not os.path.exists(filename):
            continue

        is_eeg = False
        if 'eeg' in filename:
            is_eeg = True

        # the first created file of the session will be the basename for tint

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Processing the the following file: %s!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], filename))

        # get the sample number from the file

        if is_eeg:
            n_samples = get_file_parameter('num_EEG_samples', filename)
        else:
            n_samples = get_file_parameter('num_EGF_samples', filename)

        pass


def get_eeg_files(set_filename):

    """Get the EEG and EGF files"""

    directory = os.path.dirname(set_filename)

    dir_files = os.listdir(directory)

    return [os.path.join(directory, file) for file in dir_files if 'eeg' in os.path.splitext(file)[-1] or
            'egf' in os.path.splitext(file)[-1]]


def matching_ind(haystack, needle):
    idx = np.searchsorted(haystack, needle)
    mask = idx < haystack.size
    mask[mask] = haystack[idx[mask]] == needle[mask]
    idx = idx[mask]
    return idx


def is_processed(set_filename):
    """This method will check if the file has been processed already"""
    eeg_files = get_eeg_files(set_filename)

    if len(eeg_files) == 0:
        # there are no files to process, so they are all processed
        return True

    files_processed = 0
    # check if all the files have been processed
    for file in eeg_files:
        if os.path.exists(get_output_filename(file)):
            files_processed += 1  # the output file exists, so it has been processed

    if files_processed == len(eeg_files):
        return True
    return False


def get_active_tetrode(set_filename):
    """in the .set files it will say collectMask_X Y for each tetrode number to tell you if
    it is active or not. T1 = ch1-ch4, T2 = ch5-ch8, etc."""
    active_tetrode = []
    active_tetrode_str = 'collectMask_'

    with open(set_filename) as f:
        for line in f:

            # collectMask_X Y, where x is the tetrode number, and Y is eitehr on or off (1 or 0)
            if active_tetrode_str in line:
                tetrode_str, tetrode_status = line.split(' ')
                if int(tetrode_status) == 1:
                    # then the tetrode is saved
                    tetrode_str.find('_')
                    tet_number = int(tetrode_str[tetrode_str.find('_') + 1:])
                    active_tetrode.append(tet_number)

    return active_tetrode


def get_active_eeg(set_filename):
    """This will return a dictionary (cative_eeg_dict) where the keys
    will be eeg channels from 1->64 which will represent the eeg suffixes (2 = .eeg2, 3 = 2.eeg3, etc)
    and the key will be the channel that the EEG maps to (a channel from 0->63)"""
    active_eeg = []
    active_eeg_str = 'saveEEG_ch'

    eeg_map = []
    eeg_map_str = 'EEG_ch_'

    active_eeg_dict = {}

    with open(set_filename) as f:
        for line in f:

            if active_eeg_str in line:
                # saveEEG_ch_X Y, where x is the eeg number, and Y is eitehr on or off (1 or 0)
                _, status = line.split(' ')
                active_eeg.append(int(status))
            elif eeg_map_str in line:
                # EEG_ch_X Y
                _, chan = line.split(' ')
                eeg_map.append(int(chan))

                # active_eeg = np.asarray(active_eeg)
                # eeg_map = np.asarray(eeg_map)

    for i, status in enumerate(active_eeg):
        if status == 1:
            active_eeg_dict[i + 1] = eeg_map[i] - 1

    return active_eeg_dict


def is_egf_active(set_filename):
    active_egf_str = 'saveEGF'

    with open(set_filename) as f:
        for line in f:

            if active_egf_str in line:
                _, egf_status = line.split(' ')

                if int(egf_status) == 1:
                    return True

        return False


def has_files(set_filename):
    """This method will check if all the necessary files exist"""

    # it will need the set file and any .eeg or .egf files

    if len(get_eeg_files(set_filename)) > 0:
        return True
    return False


def get_output_filename(filename):

    """Returns the output filename for the input eeg file,
    this is the name of the file that the data will be saved to."""

    # figure out what the extension is
    extension = os.path.splitext(filename)[-1]

    return '%s_%s.h5' % (os.path.splitext(filename)[0], extension[1:])


def get_eeg_extensions(set_filename):
    """This will return the eeg extensions"""
    eeg_filenames = get_eeg_files(set_filename)
    return [os.path.splitext(file)[-1] for file in eeg_filenames
            if not os.path.exists(get_output_filename(file))]