from core.Tint_Matlab import *
import os
import numpy as np
from scipy import signal, interpolate, fftpack
import scipy
import numpy as np
from pyfftw.interfaces import scipy_fftpack as fftw
import mmap
import contextlib
import core.SignalProcessing as sp
import datetime
import h5py


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


def process_basename(self, set_filename, chunk_size, chunk_overlap):
    """This function will convert the .bin file to the TINT format"""

    self.analyzed_files.append(os.path.basename(set_filename))
    self.current_session = os.path.basename(set_filename)

    frequency_boundaries = {
        'Delta': np.array([1, 3]),
        'Theta': np.array([4, 12]),
        'Alpha': np.array([8, 12]),  # don't include in spatial spectral
        'Beta': np.array([13, 20]),
        'Low Gamma': np.array([35, 55]),
        'High Gamma': np.array([65, 120]),
        'Ripple': np.array([80, 250]),
        'Fast Ripple': np.array([250, 500]),
        'Spindle': np.array([7, 16])  # don't include in spatial spectral
    }

    directory = os.path.dirname(set_filename)
    self.current_directory = os.path.basename(directory)

    # tint_basename = os.path.basename(os.path.splitext(set_filename)[0])
    # pos_filename = '%s.pos' % os.path.splitext(tint_basename)[0]

    # get eeg files
    eeg_files = get_eeg_files(set_filename)

    selected_eeg_types = [item.data(0, 0) for item in self.eeg_types.selectedItems()]

    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: Analyzing files in the following directory: %s!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8], directory))

    if eeg_files is None:
        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: This filename does not have any selected EEG/EGF files: %s!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], set_filename))
        return

    for filename in eeg_files:

        if os.path.splitext(filename)[-1] not in selected_eeg_types:
            continue

        if not os.path.exists(filename):
            continue

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Processing the the following file: %s!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], filename))

        try:
            spectro_ds, total_power, f_peak, t_axis, freqs, band_order, Fs = get_st(self, filename, frequency_boundaries,
                                                                                      notch=60, f_min=1, f_max=500,
                                                                                      add_zero_freq=True,
                                                                                      output='downsample',
                                                                                      chunk_size=chunk_size,
                                                                                      chunk_overlap=chunk_overlap)
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: Saving output as the following filename: %s!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], get_output_filename(filename)))

            save_spatialSpectro(get_output_filename(filename), spectro_ds, total_power, f_peak, band_order,
                                frequency_boundaries, Fs)
        except TypeError:
            # this will happen when get_st returns nothing (quit early)
            continue


def get_eeg_files(set_filename):

    """Get the EEG and EGF files"""

    directory = os.path.dirname(set_filename)

    try:
        dir_files = os.listdir(directory)
    except FileNotFoundError:
        return

    return [os.path.join(directory, file) for file in dir_files if 'eeg' in os.path.splitext(file)[-1] or
            'egf' in os.path.splitext(file)[-1] if os.path.splitext(os.path.basename(set_filename))[0] in file]


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


def save_spatialSpectro(filename, spectro_ds, total_power, f_peak, band_order, frequency_boundaries, Fs):

    # check if the directory exists

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(filename, 'w') as hf:
        hf.attrs['Fs'] = Fs

        datahf = hf.create_group('data')
        datahf.create_dataset("spectro_ds", data=spectro_ds)
        datahf.create_dataset("f_peak", data=f_peak)
        datahf.create_dataset("total_power", data=total_power)

        bandhf = hf.create_group('bands')
        for k, v in frequency_boundaries.items():
            bandhf.create_dataset(k, data=v)

        # get maximum string length
        max_string_length = np.amax(np.asarray([len(value) for value in band_order]))
        # need to convert the string list to a format h5py can understand
        asciiList = [n.encode("ascii", "ignore") for n in band_order]
        bandhf.create_dataset('band_order', (len(asciiList), 1), 'S%d' % (int(max_string_length)), asciiList)


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


def conj_nonzeros(X):
    ind = np.where(X.imag != 0)
    X[ind] = np.conj(X[ind])

    return X


def stransform(h, Fs):
    '''
    Compute S-Transform without for loops

    Converted from MATLAB code written by Kalyan S. Dash

    Converted by Geoffrey Barrett, CUMC

    h - an 1xN vector representing timeseries data, units will most likely by uV

    returns the stockwell transform, representing the values of all frequencies from 0-> Fs/2 (nyquist) for each time
    '''

    h = np.asarray(h, dtype=float)

    # scipy.io.savemat('stransform_numpy.mat', {'h': h})

    h = h.reshape((1, len(h)))  # uV

    n = h.shape[1]

    num_voices = int(Fs / 2)
    '''
    if n is None:
        n = h.shape[1]

    print(n)
    '''

    # n_half = num_voices


    n_half = np.fix(n / 2)

    n_half = int(n_half)

    odd_n = 1

    if n_half * 2 == n:
        odd_n = 0

    f = np.concatenate((np.arange(n_half + 1),
                        np.arange(-n_half + 1 - odd_n, 0)
                        )) / n  # array that goes 0-> 0.5 and then -0.5 -> 0 [2*n_half,]

    Hft = fftw.fft(h, axis=1)  # uV, [1xn]

    Hft = conj_nonzeros(Hft)

    # compute all frequency domain Guassians as one matrix

    invfk = np.divide(1, f[1:n_half + 1])  # matrix of inverse frequencies in Hz, [n_half]

    invfk = invfk.reshape((len(invfk), 1))

    W = np.multiply(
        2 * np.pi * np.tile(f, (n_half, 1)),  # [n_half, f]
        np.tile(invfk.reshape((len(invfk), 1)), (1, n)),  # [n_half(invfk) x n]
    )  # n_half x len(f)

    G = np.exp((-W ** 2) / 2)  # Gaussian in freq domain
    G = np.asarray(G, dtype=np.complex)  # n_half x len(f)

    # Compute Toeplitz matrix with the shifted fft(h)

    HW = scipy.linalg.toeplitz(Hft[0, :n_half + 1].T, np.conj(Hft))  # n_half + 1 x len(h)
    # HW = scipy.linalg.toeplitz(Hft[0,:n_half+1].T, Hft)

    # exclude the first row, corresponding to zero frequency

    HW = HW[1:n_half + 1, :]  # n_half x len(h)

    # compute the stockwell transform

    cwt = np.multiply(HW, G)

    ST = fftw.ifft(cwt, axis=-1)  # compute voices

    # add the zero freq row

    # print(np.mean(h, axis=1))

    st0 = np.multiply(np.mean(h, axis=1),
                      np.ones((1, n)))

    ST = np.vstack((st0, ST))

    return ST


def find_f_peak(spectrogram, freqs):
    """This function will take a spectrogram and find the frequency that contributes the peak power to the signal

    i.e. if 20% of the power spectrum is 40 Hz and 80% is 70 Hz,
    Fpeak = 70 Hz

    spectrogram: mxn matrix (m, frequencies, n time points)"""

    freqs = np.asarray(freqs).reshape((1, -1))  # ensuring that this array is an np array

    freqs_i, time_i = np.where(spectrogram == np.amax(spectrogram, axis=0))

    F_peak = np.zeros((1, spectrogram.shape[1]))

    unique_times, time_counts = np.unique(time_i, return_counts=True)

    if sum(time_counts) == len(time_i):
        # then all the time points have unique max frequency values (most likely will be this way in all cases)
        F_peak[0, time_i] = freqs[0, freqs_i]

    else:
        # can take care of the ones with only one max frequency value
        unique_bools = np.where(time_counts) == 1
        F_peak[0, time_i[unique_bools]] = freqs[0, freqs_i[unique_bools]]

        mult_max_bool = np.where(time_counts > 1)

        for col in time_i[mult_max_bool]:
            freq_peak = freqs[0, freqs_i[np.where(col == time_i)]]
            F_peak[0, col] = np.nanmean(freq_peak)

    return F_peak.flatten()


def get_st(self, filename, frequency_boundaries, notch=60, f_min=0, f_max=500, output_Fs=1, SquaredMagnitude=True,
            add_zero_freq=False, chunk_size=10, chunk_overlap=25, output='downsample'):
    """In some cases we will have files that are too long to at once, this function will break up
    the data into chunks and then downsample the frequency data into it's appropriate frequency bands
    as this will significantly decrease the memory usage (and is ultimately what we want anyways)

    -------------------------------inputs---------------------------------------------------------

    filename: full filepath to the .eeg or .egf file that you want to analyze

    frequency boundaries: dict of frequencies to analyze key is the name, and the value is the range,
    example {'Theta': [4, 12]}

    notch: the frequency to notch filter (60 is generally the US value, 50 most likely in Europe)

    f_min: the minimum frequency to have in the results

    f_max: the maximum frequency to have in the results

    output_Fs: the sampling frequency of the frequency values (a value of 1 means 1 value per frequency)

    SquaredMagnitude: generally we want a power measure, so we need the squared magnitude of the matrix,
    this is a bool, either set it to True or False

    add_zero_freq: This will determine if you want to add the zero frequency or not (if the f_min is zero it will do this
    automatically)

    chunk_size: this is a size value the represents how many seconds of data you want to have analyzed per chunk

    chunk_overlap: the amount of overlap (half from the front, and half from the end)

    output: this will determine which output we want, if it is equal to 'downsample' we will not have the entire
    frequency range of values we will only have one value per frequency boundary. if output='power', we will have
    one value per frequency between f_min and f_max
    """

    if chunk_overlap >= chunk_size:
        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Chunk Overlap is too large, must be less than chunk_size!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8]))
        return

    if os.path.exists(get_output_filename(filename)):
        self.LogAppend.myGUI_signal.emit(
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
                m = sp.Filtering().iirfilt(bandtype='low', data=m, Fs=recorded_Fs, Wp=Fs, order=6,
                                           automatic=0, filttype='butter', showresponse=0)

                # downsample the data so it only is 1.2 kHz instead of 4.8kHz
                m = m[0::int(downsamp_factor)]

            m = sp.Filtering().dcblock(m, 0.1, Fs)  # removes DC Offset

            # removes 60 (or 50 Hz)
            m = sp.Filtering().notch_filt(m, Fs, freq=notch, band=10, order=3)

            n_samples = int(len(m))

            chunk_overlap = int(chunk_overlap * Fs)  # converting from seconds to samples
            chunk_size = int(chunk_size * Fs)  # converting from seconds to samples

            concurrent_samples = chunk_size  # samples that we will analyze per chunk

            if output == 'downsample':
                spectro_ds = np.zeros(
                    (len(frequency_boundaries), n_samples))  # initializing the downsample spectrogram output
                total_power = np.zeros((1, n_samples))  # initializing the downsample spectrogram output
                f_peak = np.zeros((1, n_samples))  # initializing the peak frequencies value
            else:
                power_all = None

            if f_min == 0 or add_zero_freq:
                data_mean = get_average(filename, Fs, notch)

            ######################### calculate power per chunk ######################
            index_start = 0
            index_stop = index_start + concurrent_samples
            # iteration_overlap = 25  # number of overlapping samples
            half_overlap = int(chunk_overlap / 2)

            percentages = np.linspace(0.1, 1, 10)

            max_index = int(n_samples - 1)

            while index_start < max_index:

                if index_stop > max_index:
                    index_stop = max_index
                    index_start = max_index - concurrent_samples

                percent_bool = np.where(index_stop / max_index >= percentages)[0]
                if len(percent_bool) >= 1:
                    self.LogAppend.myGUI_signal.emit(
                        '[%s %s]: %d percent complete for the following file: %s!' %
                        (str(datetime.datetime.now().date()),
                         str(datetime.datetime.now().time())[:8], (int(100 * percentages[percent_bool[-1]])), filename))

                    try:
                        percentages = percentages[percent_bool[-1] + 1:]
                    except IndexError:
                        percentages = np.array([])

                current_indices = [index_start, index_stop]

                data = m[current_indices[0]:current_indices[1]]

                power, t_axis, freqs = stran_psd(data, Fs, minfreq=f_min, maxfreq=f_max, output_Fs=output_Fs)

                if f_min == 0:
                    # replacing with the proper zeroth order
                    power[freqs == 0, :] = data_mean
                elif add_zero_freq:
                    # append the zeroth order

                    freqs = np.hstack((0, freqs))

                    st0 = np.multiply(data_mean,
                                      np.ones((1, power.shape[1])))
                    power = np.vstack((st0, power))

                if SquaredMagnitude:
                    # pass
                    power = power ** 2

                data = None

                if output == 'power':
                    # uncomment this to view the power values instead of the
                    if power_all is None:
                        power_all = np.zeros((len(freqs), n_samples))

                    if index_stop == max_index:

                        current_cols = np.where(np.sum(power_all, axis=0) == 0)[-1]

                        power_all[:, current_cols] = power[:, current_cols -
                                                              (max_index -
                                                               power.shape[1]
                                                               + 1)]
                        break
                    elif index_start != 0 and half_overlap != 0:

                        power_all[:, index_start + half_overlap:index_stop - half_overlap] = power[:,
                                                                                             half_overlap:-half_overlap]

                    elif index_start == 0 and half_overlap != 0:

                        power_all[:, index_start:index_stop - half_overlap] = power[:, :-half_overlap]

                    elif half_overlap == 0:
                        power_all[:, index_start:index_stop] = power

                    index_start += concurrent_samples - chunk_overlap  # need the overlap since we took off the beginning
                    index_stop = index_start + concurrent_samples
                    continue
                else:
                    # getting the peak frequencies
                    f_peak[0, index_start:index_stop] = find_f_peak(power, freqs)

                    # getting the downsampled (on the freq axis) matrix
                    spectro_ds_current, band_order, total_power_current = SpectroDownSamp(power, freqs,
                                                                                          frequency_boundaries)

                    if index_stop == max_index:

                        current_cols = np.where(np.sum(spectro_ds, axis=0) == 0)[-1]

                        spectro_ds[:, current_cols] = spectro_ds_current[:, current_cols -
                                                                            (max_index -
                                                                             spectro_ds_current.shape[1]
                                                                             + 1)]

                        total_power[:, current_cols] = total_power_current[:, current_cols -
                                                                              (max_index -
                                                                               total_power_current.shape[1]
                                                                               + 1)]
                        break
                    elif index_start != 0 and half_overlap != 0:
                        spectro_ds[:, index_start + half_overlap:index_stop - half_overlap] = spectro_ds_current[:,
                                                                                              half_overlap:-half_overlap]
                        total_power[:, index_start + half_overlap:index_stop - half_overlap] = total_power_current[:,
                                                                                               half_overlap:-half_overlap]

                    elif index_start == 0 and half_overlap != 0:

                        spectro_ds[:, index_start:index_stop - half_overlap] = spectro_ds_current[:, :-half_overlap]
                        total_power[:, index_start:index_stop - half_overlap] = total_power_current[:, :-half_overlap]

                    elif half_overlap == 0:
                        spectro_ds[:, index_start:index_stop] = spectro_ds_current
                        total_power[:, index_start:index_stop] = total_power_current

                    index_start += concurrent_samples - chunk_overlap  # need the overlap since we took off the beginning
                    index_stop = index_start + concurrent_samples

                    spectro_ds_current = None
                    total_power_current = None

    if output == 'power':
        t_axis = np.arange(power_all.shape[1]) / Fs
        return power_all, t_axis, freqs
    else:
        t_axis = np.arange(spectro_ds.shape[1]) / Fs
        return spectro_ds, total_power, f_peak, t_axis, freqs, band_order, Fs


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
                downsamp_factor = recorded_Fs / Fs

                # reading in the data
                m = np.fromstring(m, dtype='<h')
                m, scalar = bits2uV(m, filename)

                # filter before downsampling to avoid anti-aliasing
                m = sp.Filtering().iirfilt(bandtype='low', data=m, Fs=recorded_Fs, Wp=Fs, order=6,
                                           automatic=0, filttype='butter', showresponse=0)

                # downsample the data so it only is 1.2 kHz instead of 4.8kHz
                m = m[0::int(recorded_Fs / Fs)]

            m = sp.Filtering().dcblock(m, 0.1, Fs)  # removes DC Offset

            # removes 60 (or 50 Hz)
            m = sp.Filtering().notch_filt(m, Fs, freq=notch, band=10, order=3)

    return np.mean(m)


def MatlabNumSeq(start, stop, step):
    """In Matlab you can type:

    start:step:stop and easily create a numerical sequence
    """

    '''np.arange(start, stop, step) works good most of the time

    However, if the step (stop-start)/step is an integer, then the sequence
    will stop early'''

    seq = np.arange(start, stop + step, step)

    if seq[-1] > stop:
        seq = seq[:-1]

    return seq


def stran_psd(h, Fs, minfreq=0, maxfreq=600, output_Fs=1):
    '''The s-transform, ST, returns an NxM, N being number of frequencies, M being number of time points'''

    nyquist = Fs / 2

    ST = stransform(h, Fs)  # returns all frequencies between 0 and the nyquist frequency

    # f = stransform(h)
    # return f

    if minfreq < 0 or minfreq > nyquist:
        # maximum frequency you can obtain is Nyquist frequency ()
        minfreq = 0
        print('Minfreq < 0 or > Nyquist, setting value to 0')

    if maxfreq > nyquist:
        print('Maxfreq > Nyquist setting value to Nyquist')
        maxfreq = nyquist

    if minfreq > maxfreq:
        print('Minfreq > Maxfreq, swapping values')
        temp_value = minfreq
        minfreq = maxfreq
        maxfreq = temp_value
        temp_value = None

    # downsample the frequencies so that it matches our output frequencies
    f = np.arange(0, ST.shape[0]) / (ST.shape[0] - 1) * nyquist

    freq_out = MatlabNumSeq(np.floor(minfreq), np.ceil(maxfreq), 1 / output_Fs)
    freq_out = freq_out[np.where((freq_out >= minfreq) * (freq_out <= maxfreq))]
    # print(freq_out)

    if f != freq_out:
        desired_frequency_indices = []
        for freq in freq_out:
            freq_bool = np.where(f == freq)[0]
            if len(freq_bool) > 0:
                desired_frequency_indices.append(freq_bool[0])
            else:
                # find the closest frequency
                new_freq = np.abs(f - freq)
                freq_bool = np.where(new_freq == np.amin(new_freq))[0]
                desired_frequency_indices.append(freq_bool[0])

        desired_frequency_indices = np.asarray(desired_frequency_indices)

    # desired_frequency_indices = np.where((f >= minfreq) & (f <= maxfreq))
    # print(f.shape, f)
    f = f[desired_frequency_indices]

    # maxfreq_index = np.round(nyquist_index * (maxfreq/nyquist_index))
    # minfreq_index = np.round(nyquist_index * (minfreq / nyquist_index))

    # ST = ST[desired_frequency_indices[0], :]
    ST = ST[desired_frequency_indices, :]

    power = np.abs(ST)

    # normalize phase estimates to one length
    # nST = np.divide(ST, power)
    # phase = np.angle(nST)

    t_axis = np.arange(power.shape[1]) / Fs

    return power, t_axis, f


def SpectroDownSamp(spectro, freqs, bands):
    '''This will take in a spectrogram and downsample it to only include one bin representing each frequency band

    spectro: the spectrogram matrix

    freqs: each row should correspond to values of a certain frequency, this is the array that each row corresponds to
    bands: a dictionary of the frequency bands

    bands: consists of a dictionary of frequency
    '''
    freqs = freqs.reshape((1, -1))
    spectro_ds = np.zeros((len(bands), spectro.shape[1]))  # initializes the downsampled matrix

    ordered_bands = []
    for key, val in sorted(bands.items(), key=lambda x: x[1][0]):
        ordered_bands.append(key)

    for i, band in enumerate(ordered_bands):

        freq_boundaries = bands[band]  # [low_freq_boundary, high_freq_boundary]

        # provides the row indices that match the
        boundary_bool = np.where((freqs >= freq_boundaries[0]) * (freqs <= freq_boundaries[1]))[1]
        if len(boundary_bool) > 0:
            # otherwise the boundary does not exist in the freqs
            spectro_ds[i, :] = np.nansum(spectro[boundary_bool, :], axis=0)  # sum along the columns

    # avg_power = np.mean(spectro, axis=0)
    total_power = np.sum(spectro, axis=0)
    spectro_ds = np.divide(spectro_ds, total_power)  # turning the values to a percentage

    return spectro_ds, ordered_bands, total_power.reshape((1, -1))