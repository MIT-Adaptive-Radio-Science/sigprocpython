"""
Various types of Time-Frequency estimators.
"""

from pathlib import Path
import argparse
import scipy.io.wavfile as wavio
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np


def create_sti(filename, nfft, decimation, secoffset):
    """This does the calculation for the STI plot

    Creates the spectral time intensity plot of the whole file. Outputs the sti in dB/Hz. Array is frequency x time.

    Parameters
    ----------
    filename : str
        Name of the wav file.
    nfft : int
        Number of fft bins after decimation.
    decimation : int
        Decimation factor.
    secoffset : float
        Number of seconds to offset the time vector.
    Returns
    -------
    freq : array_like
        Vector of frequency values in Hz
    t_ar : array_like
        Time array in seconds.
    len_s : float
        Number of seconds of data in file.
    Sxxlist : list
        List of the spectrogram arrays in dB/Hz.
    """
    rate, data = wavio.read(filename)
    len_s = float(data.shape[0]) / rate
    freq = np.fft.rfftfreq(nfft, 1.0 / rate)
    if data.ndim == 1:
        if decimation != 1:
            data = sig.decimate(data, decimation)
            rate = rate / decimation

        ntime = data.shape[0] // nfft
        t_ar = (np.arange(ntime) + 0.5) * nfft / rate
        datamat = data[: ntime * nfft].reshape(ntime, nfft)
        Sxx0 = np.abs(np.fft.rfft(datamat, axis=1)) ** 2 / rate
        # freq, t_ar, Sxx0 = sig.spectrogram(data, rate, nfft=nfft, scaling="density")
        Sxx0db = 10 * np.log10(Sxx0.transpose() + 1e-6)
        Sxxlist = [Sxx0db]
    elif data.ndim == 2:
        nchan = data.shape[1]
        Sxxlist = []
        if decimation != 1:
            data = sig.decimate(data, decimation, axis=0)
            rate = rate / decimation
        for ichan in range(nchan):
            ntime = data.shape[0] // nfft
            t_ar = (np.arange(ntime) + 0.5) * nfft / rate
            datamat = data[: ntime * nfft, ichan].reshape(ntime, nfft)
            Sxx0 = np.abs(np.fft.rfft(datamat, axis=1)) ** 2 / rate
            # freq, t_ar, Sxx0 = sig.spectrogram(
            #     data[:, ichan], rate, nfft=nfft, scaling="density"
            # )
            Sxx0db = 10 * np.log10(Sxx0.transpose() + 1e-12)
            Sxxlist.append(Sxx0db)
    t_ar = t_ar + secoffset
    return freq, t_ar, len_s, Sxxlist
