from core.Tint_Matlab import *
import os, datetime
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from scipy import signal, interpolate
import scipy
from pyfftw.interfaces import scipy_fftpack as fftw
import mmap
import contextlib
import core.SignalProcessing as sp
import webcolors


def conj_nonzeros(X):
    ind = np.where(X.imag != 0)
    X[ind] = np.conj(X[ind])

    return X


def CWTGaborWavelet(data, Fs, f_min=0.1, f_max=500, FreqSeg=None, StDevCycles=3, TimeStep=None,
                    magnitudes=True, SquaredMag=False):
    """
    Will perform the continuous wavelet transform using the Gabor Wavelet

    Fourier transform is the most common tool for analyzing frequency properties, however the
    time-evolution of the resulting frequencies are not reflected in the Fourier transform
    (not directly). Essentially, it also does represent abrupt changes in the waveform efficiently.
    The fourier transform represents data as a sum of sine waves which are not localized in
    time or space, and oscillate forever. Thus we will be using wavelets in order to keep
    both the time and frequency aspects of the signal (and better handle the transient signals).

    Wavelet is a rapidly decaying wave-like oscillation with a mean of zero. There are many
    types of wavelets: Morlet, Daubechies, Mexican Hat, etc.

    The function that is going to be transformed is first multiplied by the Gaussian function,
    g(x), then is transformed using a Fourier Transform.

    Gaussian function, g(x) = (1/((σ^2 * π) ^(1/4))) * e^(-t^2 / (2*σ^2))
    Gabor Wavelet, ψ(t) = g(t)*e^(iηt)

    Guassian envelope windows are preferred as the same type of function would exist in either
    domain (if we did rectangular envelopes, the conjugate domain would have a sinx/x function).


    Continuous Wavelet Transform, C(s, t) = 1/sqrt(s)∫ f(t)*ψ((t-τ)/s), where s is scale,
    τ is translation, and η is angular frequency at scale s,

    σ = 6/η in order to satisfy the admissibility condition


    inputs:
    data: data to perform the trasnform on
    Fs: sampling frequency (Hz) of the collected data
    f_min: minimum frequency (Hz) to include in the resulting transform
    f_max: maximum frequency (Hz) to include in the resulting trasnform
    FreqSeg: the number of frequency segments to calculate
    StDevCycles: in the wavelet transform, the scale corresponding to each frequency defines
    the gaussian's standard dev used for calculating the transform. This parameter defines
    the number of cycles you want to include in the transform at every frequency
    TimeStep: This step value will determine the downsampling of the time value,
    default is None, in which case all time values are preserved
    Magnitudes: if True (default), it will return the magnitudes of the coefficients, False will
    return analytic values (complex values)
    SquaredMag:

    """

    if FreqSeg is None:
        FreqSeg = Fs

    if f_min >= f_max:
        print("The minimum frequency is larger than the maximum frequency, switching values")
        temp_values = (f_min, f_max)
        f_max, f_min = temp_values

    if f_max < 0 or f_min < 0:
        print("Inappropriate frequency value provided (does not accept negative values)")

    # checking that the max frequency is appropriate
    if f_max > Fs / 2:
        f_max = Fs / 2

    # checking that the minimum frequency is appropriate
    if f_min == 0:
        f_min = 0.1

    # if an improper segment value was provided, fix it
    if FreqSeg <= 0:
        FreqSeg = np.around(f_max - f_min)

    # calculate the frequency values

    step = (f_max - f_min) / (FreqSeg - 1)

    freqs = MatlabNumSeq(f_min, f_max, step).reshape((1, -1))
    # freqs = np.arange(f_min, f_max+step-f_min, step)

    # flip the matrix
    freqs = np.fliplr(freqs)

    if len(data) % 2 == 0:
        # need an odd number of data points
        data = data[:-1]

    t = np.arange(len(data)) / Fs  # creating the time array

    n = len(t)  # number of data points
    n_half = int(np.floor(n / 2) + 1)

    # creating an array from 0 -> 2 pi
    WAxis = np.multiply(np.arange(n), (2 * np.pi / n))

    WAxis *= Fs
    WAxis_half = WAxis[:n_half]

    if TimeStep is None:
        SampAve = 1  # no timestep was given, keep all the time values
    else:
        SampAve = np.around(TimeStep * Fs)  #

        if SampAve < 1:
            SampAve = 1

    SampAveFilt = np.array([])
    if SampAve > 1:
        # down sample the time
        # IndSamp = np.arange(0, len(t), SampAve)
        IndSamp = MatlabNumSeq(0, len(t) - 1, SampAve)
        t = t[IndSamp]
        SampAveFilt = np.ones(SampAve, 1)

    data_fft = fftw.fft(data)  # transforming the input signal to the S domain
    # data_fft = conj_nonzeros(data_fft)

    GaborWT = np.zeros((freqs.shape[1], len(t)), dtype=np.complex)

    FreqInd = -1

    # reshaped freqs so that the freq value would take on a single value, not the entire array
    for freq in freqs.reshape((-1, 1)):
        # Calculating each frequency
        StDevSec = (1 / freq) * StDevCycles

        WinFFT = np.zeros((n, 1))
        WinFFT[:n_half] = np.exp(-0.5 * np.power(
            WAxis_half - (2 * np.pi * freq), 2) * (StDevSec ** 2)
                                 ).reshape((-1, 1))
        WinFFT = np.multiply(WinFFT, (np.sqrt(n) / np.linalg.norm(WinFFT, 2)))
        # need to convert to complex otherwise imaginary numbers will be discarded
        WinFFT = np.asarray(WinFFT, dtype=np.complex).flatten()

        FreqInd += 1

        if SampAve > 1:

            GaborTemp = np.zeros(len(data_fft) + SampAve - 1, 1)
            GaborTemp[SampAve - 1:] = fftw.ifft(np.multiply(data_fft, WinFFT)) / sqrt(StDevSec)

            if magnitudes:
                # return only the magnitudes
                GaborTemp = np.absolute(GaborTemp)

            if SquaredMag:
                # return the squared magnitude
                if not magnitudes:
                    GaborTemp = np.absolute(GaborTemp)
                GaborTemp = GaborTemp ** 2

            GaborTemp[:SampAve - 1] = np.flipud(GaborTemp[SampAve, 2 * SampAve])
            GaborTemp = np.divide(signal.lfilter(SampAveFilt, 1, GaborTemp), SampAve)
            GaborTemp = GaborTemp[SampAve - 1:]

            GaborWT[FreqInd, :] = GaborTemp[IndSamp]
        else:
            GaborWT[FreqInd, :] = fftw.ifft(np.multiply(data_fft, WinFFT)) / np.sqrt(StDevSec)

    if SampAve > 1:
        return GaborWT, t, freqs

    if magnitudes:
        GaborWT = np.absolute(GaborWT)

    if SquaredMag:
        if not magnitudes:
            GaborWT = np.absolute(GaborWT)
        GaborWT = GaborWT ** 2

    # we flipped the GaborWT to have it in the same format as our
    # Stockwell Transform method
    return np.flipud(GaborWT), t, np.fliplr(freqs)


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


def data2wavelet(data):
    """

    """

    first_half_ind = int(np.around(len(data) * 0.5) + 1)  # indices of the first half of the data

    initial_window = data[:first_half_ind + 1]  # uses the first half of the data as the initial window
    end_window = data[-2 - first_half_ind:]  # defines the end window

    first = data[0]  # the first value
    last = data[-1]  # defines the last value

    initial_window = np.subtract(initial_window, first)  # wavelets start at 0
    end_window = np.subtract(end_window, last)  # wavelets end with a 0

    initial_window = np.flipud(-initial_window) + first
    end_window = np.flipud(-end_window) + last

    initial_window = initial_window[:-1]
    end_window = end_window[1:]

    return np.concatenate((initial_window, data, end_window)), initial_window, end_window