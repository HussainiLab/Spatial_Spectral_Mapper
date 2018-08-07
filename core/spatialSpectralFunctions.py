from core.Tint_Matlab import *
import os, datetime, time
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from scipy import signal, interpolate
import scipy
from pyfftw.interfaces import scipy_fftpack as fftw
import mmap
import contextlib
import core.filtering as filt
import webcolors
import h5py


def create_colormap(color_list, boundary_list):
    cmap = matplotlib.colors.ListedColormap(color_list)
    # cmap.set_over('0.25')
    # cmap.set_under('0.75')

    bounds = boundary_list
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def get_spatialSpecData(filename):
    """This function will get all the necessary data for spatialSpectrogram"""
    with h5py.File(filename, 'r') as hf:

        for item in hf.attrs.keys():
            if 'Fs' in item:
                Fs = hf.attrs[item]

        for item, value in hf.items():
            # print(item, value)

            if item == 'bands':
                bands = hf.get(item)
            elif item == 'data':
                data = hf.get(item)

        # re-acquiring the frequency-boundaries
        frequency_boundaries = {}
        for item, value in bands.items():

            if item != 'band_order':
                frequency_boundaries[item] = bands.get(item)[:]
            else:
                # recreating the list that dictates the frequency band order
                # for the data matrix
                band_order = bands.get(item)[:]
                band_order = band_order.flatten()
                # convert from bytes to string (had to convert to bytes to save)
                band_order = [value.decode('utf-8') for value in band_order]

        # re-acquiring the data
        for item, value in data.items():
            if 'spectro' in item:
                spectro_ds = data.get(item)[:]
            elif 'peak' in item:
                f_peak = data.get(item)[:]
    return spectro_ds, f_peak, frequency_boundaries, band_order, Fs


def get_all_labels(geometry):
    rows, cols = geometry

    x = np.arange(cols) + 1
    y = np.arange(rows) + 1

    labels = np.zeros(geometry)

    for i, x_value in enumerate(x):
        x_col = np.repeat(x_value, rows).reshape((-1, 1))  # creating column vector
        labels[:, i] = (x_col + np.flipud(y.reshape((-1, 1)) / 100)).flatten()

    return labels


def track_to_xylabels(x_pos, y_pos, geometry):
    """
    returns the list of labels for each datapoint

     For example: 2x2 grid (geometry = (2,2))

              y=100 -------------------------------------------
                    |                    |                     |
                    |                    |                     |
                    |   Label = 1.2      |   Label = 2.2       |    y
                    |                    |                     |    |
                    |                    |                     |    a
               y=50 -------------------------------------------     x
                    |                    |                     |    i
                    |                    |                     |    s
                    |  Label = 1.1       |   Label = 2.1       |
                    |                    |                     |
                    |                    |                     |
                y=0 -------------------------------------------|
                    x=0                 x=50                  x=100
                                x-axis

    """

    # x is geometry[1] because the 2nd value pertains to number of columns
    x_ticks = np.linspace(np.min(x_pos), np.max(x_pos), geometry[1] + 1)[:-1]
    y_ticks = np.linspace(np.min(y_pos), np.max(y_pos), geometry[0] + 1)[:-1]

    x_lab = np.sum([x_pos >= t for t in x_ticks], 0)
    y_lab = np.sum([y_pos >= t for t in y_ticks], 0)

    return 1. * x_lab + y_lab / 100


def plot_map_labels(x_pos, y_pos, labels, ax=None, **args):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    for l, c in zip(np.unique(labels), colors):
        ax.scatter(x_pos[labels == l], y_pos[labels == l], color=c, **args)

    return ax


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


def spatialSpectroMap(F, posx, posy, labels=None, geometry=(32, 32)):
    """This will produce a ratemap-esque matrix (could be plotted)

    F: an array of frequencie
    posx: the position at that frequency
    posy: the y position at that frequency
    geometry: nxm grids"""

    spectroMap = np.zeros((geometry))

    all_labels = get_all_labels(geometry)

    if labels is None:
        labels = track_to_xylabels(posx, posy, geometry)

    for row, row_labels in enumerate(all_labels):
        for col, current_label in enumerate(row_labels):

            label_bool = np.where(labels == current_label)

            current_freqs = F[label_bool]
            if len(current_freqs) == 0:
                spectroMap[row, col] = np.NaN
            else:

                spectroMap[row, col] = np.mean(current_freqs)

    return spectroMap


def get_filename(filename, ext='pos'):
    """This returns various filenames given the input of the h5py file

    filename: the name of the output filename "c:\example\test_file.h5"
    file: options='pos', 'set', the filetype that it will outout
    """

    directory = os.path.dirname(os.path.dirname(os.path.dirname(filename)))  # need to go up 3 directories

    filename = os.path.basename(filename)  # return only the basename

    return os.path.join(directory, '%s.%s' % ('_'.join(filename.split('_')[:-1]), ext))


color_list = [[215 / 255, 217 / 255, 221 / 255],  # 0, to 1 Hz, light gray
                  [1 / 255, 0 / 255, 198 / 255],  # 1 to 3 Hz, navy blue
                  [215 / 255, 217 / 255, 221 / 255],  # 3 to 4 Hz, light gray
                  [66 / 255, 129 / 255, 255 / 255],  # 4 to 12 Hz, blue
                  [215 / 255, 217 / 255, 221 / 255],  # 12 to 13 Hz, light gray
                  'cyan',  # 13 to 20 Hz, light gray, beta
                  [215 / 255, 217 / 255, 221 / 255],  # 20 to 35 Hz, light gray
                  [59 / 255, 236 / 255, 26 / 255],  # 35 to 55 Hz, green
                  [215 / 255, 217 / 255, 221 / 255],  # 55 to 65 Hz, light gray
                  [175 / 255, 255 / 255, 191 / 255],  # 65 to 80 Hz, light green
                  'yellow',  # 80 to 120 Hz, yellow
                  'orange',  # 120 to 250 Hz, orange
                  'red'  # 250 to 500 Hz, red
                  ]


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


def spatialSpectroAnalyze(self, current_start=0, current_stop=None):

    self.analyzing = True
    # check if the file exists still
    filename = self.filename.text()
    if not os.path.exists(filename):
        self.choice = ''

        self.LogError.myGUI_signal.emit('H5ExistError')
        while self.choice == '':
            time.sleep(0.1)
        self.analyzing = False
        return

    # check if the geometry is valid
    geometry = self.get_geometry()
    if geometry is False:

        self.choice = ''

        self.LogError.myGUI_signal.emit('GeometryError')
        while self.choice == '':
            time.sleep(0.1)
        self.analyzing = False
        return

    if current_stop is not None:
        if current_start >= current_stop:
            self.analyzing = False
            return

    # self.analyzing = True

    self.spectroGraphAxis.clear()

    # read the data
    if not self.data_loaded:
        # this if statement is to make sure the data is only read once

        eeg_type = os.path.splitext(os.path.basename(filename).split('_')[-1])[0]
        self.eeg, Fs_EEG = ReadEEG(get_filename(filename, eeg_type))
        self.eeg_t = np.arange(len(self.eeg)) / Fs_EEG
        zero_ind = find_consec(np.where((self.eeg.flatten() == 0) | (self.eeg.flatten() == 0.0))[0])[-1]

        # check if the final index is in the list
        if len(self.eeg.flatten()) - 1 not in zero_ind:
            self.appended_zeros = False
            pass
        else:
            self.appended_zeros = True
            self.zero_t_start = self.eeg_t[zero_ind[0]-1]

        self.spectro_ds, self.f_peak, self.frequency_boundaries, self.band_order, Fs = get_spatialSpecData(filename)


        self.spatial_spectral_freq_bounds = {}

        for key, value in self.frequency_boundaries.items():
            if 'spindle' in key.lower():
                continue

            elif 'alpha' in key.lower():
                continue

            else:
                self.spatial_spectral_freq_bounds[key] = value


        self.t_axis = np.arange(self.spectro_ds.shape[1]) / Fs
        self.t_axis_bool = np.where(self.t_axis <= self.zero_t_start)[0]

        self.spectro_ds = self.spectro_ds[:, self.t_axis_bool]
        self.t_axis = self.t_axis[self.t_axis_bool]

        self.max_t = np.amax(self.t_axis)
        # self.max_t = np.amax(self.t_axis[self.t_axis_bool])

        current_start = current_start
        current_stop = self.max_t

        self.t_start.setText(str(current_start))
        self.t_stop.setText(str(current_stop))
        self.slice_size.setText(str(np.amax(self.t_axis)))

        ####### setting up the t-f plots ############
        self.spectro_yticks = MatlabNumSeq(0.5, len(self.band_order), 1)
        self.spectro_ydotted = np.arange(len(self.band_order))

        ########## read the positions ############
        pos_filename = get_filename(filename, 'pos')
        arena = self.arena.currentText()

        posx, posy, post, Fs_pos = getpos(pos_filename, arena)  # getting the mouse position

        # centering the positions
        center = centerBox(posx, posy)
        posx = posx - center[0]
        posy = posy - center[1]

        # Threshold for how far a mouse can move (100cm/s), in one sample (sampFreq = 50 Hz
        threshold = 100 / 50  # defining the threshold

        posx, posy, post = remBadTrack(posx, posy, post, threshold)  # removing bad tracks (faster than threshold)

        nonNanValues = np.where(np.isnan(posx) == False)[0]
        # removeing any NaNs
        post = post[nonNanValues]
        posx = posx[nonNanValues]
        posy = posy[nonNanValues]

        # box car smoothing, closest we could get to replicating Tint's speeds
        B = np.ones((int(np.ceil(0.4 * Fs_pos)), 1)) / np.ceil(0.4 * Fs_pos)
        # posx = scipy.ndimage.correlate(posx, B, mode='nearest')
        posx = scipy.ndimage.convolve(posx, B, mode='nearest')
        # posy = scipy.ndimage.correlate(posy, B, mode='nearest')
        posy = scipy.ndimage.convolve(posy, B, mode='nearest')

        # interpolating the positions so there's one position per time value

        Func_posx = interpolate.interp1d(post.flatten(), posx.flatten(), kind='linear')
        Func_posy = interpolate.interp1d(post.flatten(), posy.flatten(), kind='linear')

        # t_axis will have more points (and have larger max time so we need to only go to where post goes up to)

        position_bool = np.where(self.t_axis <= np.amax(post))

        self.posx_interp = Func_posx(self.t_axis[position_bool])
        self.posy_interp = Func_posy(self.t_axis[position_bool])

        # extend the interpolation with repeats of the last values of posx_interp, and posy_interp to ensure that
        # t_axis and posx_interp/posy_interp have the same length
        missing_positions = np.abs(len(self.posx_interp) - len(self.t_axis))

        if missing_positions > 0:
            # when we weren't removing the appended zeros there used to be some missing points as the t_axis
            # values extended larger than the post values (thus requiring extrapolation not interpolation)
            self.posx_interp = np.concatenate(
                (self.posx_interp, np.tile(self.posx_interp[-1], (missing_positions, 1)).flatten()))
            self.posy_interp = np.concatenate(
                (self.posy_interp, np.tile(self.posy_interp[-1], (missing_positions, 1)).flatten()))

        self.v = speed2D(self.posx_interp, self.posy_interp, self.t_axis)  # smoothed speed of the mouse
    else:
        arena = self.arena.currentText()

    if geometry != self.previous_geometry:
        ############ breaking up the positions into bins to analyze #################
        # we will keep this one out of the data_loaded if statement since the user can change the geometry
        self.labels = track_to_xylabels(self.posx_interp, self.posy_interp, geometry)  # each bin has its own label
        self.x_ticks = np.linspace(np.min(self.posx_interp), np.max(self.posx_interp), geometry[1] + 1)
        self.y_ticks = np.linspace(np.min(self.posy_interp), np.max(self.posy_interp), geometry[0] + 1)
        self.extent_vals = (np.amin(self.x_ticks), np.amax(self.x_ticks), np.amin(self.y_ticks), np.amax(self.y_ticks))

    # finding the indices for the current slice
    t_bool = np.where((self.t_axis >= current_start) * (self.t_axis <= current_stop))[0]

    t_axis = self.t_axis[t_bool]
    spectro_ds = self.spectro_ds[:, t_bool]
    posx_interp = self.posx_interp[t_bool]
    posy_interp = self.posy_interp[t_bool]
    labels = self.labels[t_bool]
    f_peak = self.f_peak[0, t_bool]
    v = self.v[t_bool]

    ####### plotting the spectro (t-f tab) ############

    self.spectro_extent_vals = (np.amin(t_axis), np.amax(t_axis), 0, len(self.band_order))
    plot = self.spectroGraphAxis.imshow(spectro_ds, origin='lower', aspect='auto', cmap='jet', interpolation='nearest',
                                        extent=self.spectro_extent_vals)
    self.spectroGraphAxis.set_yticks(self.spectro_yticks)
    self.spectroGraphAxis.set_yticklabels(self.band_order)  # Or we could use plt.xticks(...)
    self.spectroGraphAxis.hlines(self.spectro_ydotted, np.amin(t_axis), np.amax(t_axis), colors='w',
                                 linestyles='dashed')

    if self.spectroGraphColorbar is not None:
        pass  # this won't be changed since it will be normalized to 1
        # self.spectroGraphColorbar.update_normal(plot)
    else:
        self.spectroGraphColorbar = self.spectroGraph.colorbar(plot, ax=self.spectroGraphAxis)

    self.spectroGraphAxis.set_xlabel('Seconds (s)')


    ####### plotting the position bins ###############
    self.position_binsGraphAxis.clear()
    plot_map_labels(posx_interp, posy_interp, labels, ax=self.position_binsGraphAxis)
    # plt.plot(posx_interp, posy_interp, "k-", alpha=0.2)
    plt.plot(posx, posy, "k-", alpha=0.2)

    self.position_binsGraphAxis.vlines(self.x_ticks, np.min(self.posy_interp), np.max(self.posy_interp),
                                       linestyles='dashed')
    self.position_binsGraphAxis.hlines(self.y_ticks, np.min(self.posx_interp), np.max(self.posx_interp),
                                       linestyles='dashed')
    self.position_binsGraphAxis.set_xlabel('X-Position (cm)')
    self.position_binsGraphAxis.set_ylabel('Y-Position (cm)')

    # producing spectral map

    spatialMap_peak = spatialSpectroMap(f_peak.flatten(), posx_interp, posy_interp, labels, geometry)

    colorbar_ticks = list(np.unique(np.asarray(list(self.spatial_spectral_freq_bounds.values()))))
    colorbar_ticks.insert(0, 0)

    self.PeakFreqGraphAxis.clear()

    cm1, norm = create_colormap(color_list, colorbar_ticks)
    # C = create_colormap(color_dict, np.nanmin(spatialMap_peak), np.ceil(np.nanmax(spatialMap_peak)))
    peak_plot = self.PeakFreqGraphAxis.imshow(spatialMap_peak, aspect='auto', cmap=cm1, norm=norm,
                                              extent=self.extent_vals)
    # self.PeakFreqGraphAxis.plot(posx_interp, posy_interp, "k-", alpha=0.3)
    self.PeakFreqGraphAxis.plot(posx, posy, "k-", alpha=0.3)  # this has less data than the posx_interp, so should be easier to plot
    self.PeakFreqGraphAxis.set_title("Peak Frequencies")

    if self.PeakFreqGraphColorbar is not None:
        pass  # this won't be changed since it will be normalized to 500
        # self.PeakFreqGraphColorbar.update_normal(peak_plot)
    else:
        self.PeakFreqGraphColorbar = self.PeakFreqGraph.colorbar(peak_plot, ax=self.PeakFreqGraphAxis,
                                                                 ticks=colorbar_ticks)

    self.PeakFreqGraphAxis.set_xlabel('X-Position (cm)')
    self.PeakFreqGraphAxis.set_ylabel('Y-Position (cm)')

    # producing subplots that show band percentages within each bin

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # for band, ax in enumerate(axs.flatten()):

    # plot the velocity

    velocity_spectroMap = spatialSpectroMap(v, posx_interp, posy_interp, labels, geometry)

    # plot the frequency bands
    for band, ax in enumerate(self.band_percGraphAxis.flatten()):
        band = band - 1
        ax.clear()

        if ax == self.band_percGraphAxis[0, 0]:
            norm_vel = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(velocity_spectroMap))
            velocity_plot = ax.imshow(velocity_spectroMap, aspect='auto', cmap='jet', norm=norm_vel,
                                                                 extent=self.extent_vals)
            # ax.plot(posx_interp, posy_interp, "k-", alpha=0.3)
            ax.plot(posx, posy, "k-", alpha=0.3)  # this has less data than the posx_interp, so should be easier to plot
            ax.set_title("Speed (cm/s)")

            if len(self.bandpercGraphColorbar) == 10:
                current_colorbar = self.bandpercGraphColorbar[0]
                # current_colorbar.update_normal(velocity_plot)
                # velocity_plot.set_clim([0, np.nanmax(velocity_spectroMap)])
                current_colorbar.set_clim([0, np.nanmax(velocity_spectroMap)])
                current_colorbar.draw_all()
            else:
                current_colorbar = self.band_percGraph.colorbar(velocity_plot, ax=self.band_percGraphAxis[0, 0])
                self.bandpercGraphColorbar.append(current_colorbar)
            continue

        band_name = self.band_order[band]
        current_freqs = self.frequency_boundaries[band_name]
        current_spectroMap = spatialSpectroMap(spectro_ds[band, :], posx_interp, posy_interp, labels, geometry)
        peak_plot = ax.imshow(current_spectroMap, aspect='auto', cmap='jet', norm=norm, extent=self.extent_vals)
        # ax.plot(posx_interp, posy_interp, "k-", alpha=0.3)
        ax.plot(posx, posy, "k-", alpha=0.3)  # this has less data than the posx_interp, so should be easier to plot
        ax.set_title("%s (%d Hz - %d Hz)" % (band_name, current_freqs[0], current_freqs[1]))
        #ax.set_xlabel('X-Position (cm)')
        #ax.set_ylabel('Y-Position (cm)')
        if len(self.bandpercGraphColorbar) == 10:
            pass  # this won't be changed since it will be normalized to 1
            # current_colorbar = self.bandpercGraphColorbar[band + 1]
            # current_colorbar.update_normal(peak_plot)
        else:
            current_colorbar = self.band_percGraph.colorbar(peak_plot, ax=ax)
            self.bandpercGraphColorbar.append(current_colorbar)

    self.PeakFreqGraphCanvas.draw()
    self.spectroGraphCanvas.draw()
    self.band_percGraphCanvas.draw()
    self.position_binsGraphCanvas.draw()

    self.previous_t_start = current_start
    self.previous_t_stop = current_stop
    self.previous_geometry = geometry
    self.previous_arena = arena
    self.analyzing = False

    if not self.data_loaded:
        self.data_loaded = True
