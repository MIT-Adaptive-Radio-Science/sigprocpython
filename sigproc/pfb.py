from pathlib import Path
import math
import numpy as np
import scipy.signal as sig
from scipy.signal._upfirdn_apply import _output_len
import matplotlib.pyplot as plt

chanstocoef = {64: "kaiser64.csv"}


def pfbresponse(taps, nchans, fs):
    """Creates the frequency response of a pfb given the taps.

    Parameters
    ----------
    taps : str or array_like
        If string will treat it as a csv file. If array treats it as taps to the filter.
    nchans : int
        The number of channesl for the pfb.
    fs : float
        Original sampling frequency of the signal in Hz.

    Returns
    -------
    freq_ar : array_like
        Frequency vector in Hz.
    filt_dict : dict
        Keys are filter number, values are frequency response in dB
    """
    # mod_path = Path(__file__).parent.parent
    #
    # coeffile = mod_path.joinpath('coeffs',chanstocoef[nchans])
    if isinstance(taps, str):
        pfb_coefs = np.genfromtxt(str(taps), delimeter=",")
    else:
        pfb_coefs = taps
    b = pfb_coefs / np.sqrt(np.sum(np.power(pfb_coefs, 2)))

    nfreq = 2 ** 11
    [_, h] = sig.freqz(b, 1, nfreq, fs=fs, whole=True)
    hdb = np.fft.fftshift(20 * np.log10(np.abs(h)))
    hdb = hdb - np.nanmax(hdb)
    freq_ar = np.fft.fftshift(np.fft.fftfreq(nfreq, d=1.0 / fs))

    # hpow = fftshift(20*np.log10(np.abs(h)));
    nsamps = nfreq // nchans
    filt_dict = {0: hdb}
    nplot = min(5, nchans // 2)

    for i in range(-nplot, nplot):
        filt_dict[i] = np.roll(hdb, i * nsamps)
    return freq_ar, filt_dict


def prefix(num):
    """Given a number will give you the metric prefix, symbol and muliplier, e.g. kilo, Mega etc.

    Parameters
    ----------
    num : float
        Number to be analyzed.

    Returns
    -------
    list :
        A list of the multiplier, symbol and prefix strings.
    """

    outex = int(np.floor(np.log10(np.abs(num)) - 0.3))

    prefix_dict = {
        0: [1.0, "", ""],
        3: [1e-3, "k", "kilo"],
        6: [1e-6, "M", "Mega"],
        9: [1e-9, "G", "Giga"],
    }
    return prefix_dict[outex]


def plotresponse(freq_ar, filt_dict, nchans):
    """Creates a figure and axis to plot frequency response of pfb

    First axis is a single pfb channel response, the bottom is the 5 lowest frequency.

    Parameters
    ----------
    freq_ar : array_like
        Frequency vector in Hz.
    filt_dict : dict
        Keys are filter number, values are frequency response in dB
    nchans : int
        The number of channesl for the pfb.

    Returns
    -------
    fig : figure
        Image of frequency response.
    axs : array_like
        List of the axes that teh frequency response plotted on.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    mult, ack, _ = prefix(np.abs(freq_ar).max())
    fs_h = np.max(np.abs(freq_ar))
    f_wid = 1.25 * fs_h / nchans

    filtnum = (0,)
    hpow = filt_dict[filtnum]
    hplot = axs[0].plot(freq_ar * mult, hpow, "LineWidth", 3)[0]
    axs[0].set_xlim([-f_wid, f_wid])
    axs[0].set_ylim([-65, 5])
    axs[0].set_xlabel("Frequency {0}Hz".format(ack))
    axs[0].set_ylabel("Magnitude dB")
    axs[0].grid(True)
    axs[0].set_title("Single Filter")

    hlist = []
    name_list = []
    for filtnum, hpow in filt_dict.items():
        hplot = axs[1].plot(freq_ar * mult, hpow, "LineWidth", 3)[0]
        hlist.append(hplot)
        name_list.append("Channel {}".format(filtnum))
        axs[1].set_xlim([-3 * f_wid, 3 * f_wid])
        axs[1].set_ylim([-65, 5])
        axs[1].set_xlabel("Frequency {0}Hz".format(ack))
        axs[1].set_ylabel("Magnitude dB")
        axs[1].set_title("Multiple Filters")
    axs[1].legend(hlist, name_list)

    return fig, axs


def pfb_reconstruct(data, nchans, coefs, mask, fillmethod, fillparams=[], realout=True):
    """Simple PFB reconstruction

    Parameters
    ----------
    data : array_like
        A numpy array to be processed.
    nchans : int
        Number of output channes
    mask : array_like
        List of channels to be kept
    fillmethod : string
        Type of filled in the data before the reconstrution.
    fillparams : list
        Parameters for filling in empty data

    Returns
    -------
    rec_array : array_like
        The output from the polyphase.
    """

    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    _, ntime, subchan = data.shape

    shp = (nchans, ntime, subchan)
    if fillmethod == "noise":
        if fillparams:
            npw = fillparams[0]
        else:
            npw = np.nanmedian(data.flatten().real ** 2 + data.flatten().imag ** 2)
            npw = npw / np.log(2)
        d1 = np.random.randn(shp, dtype=data.dtype) + 1j * np.random.randn(
            shp, dtype=data.dtype
        )
        d1 = np.sqrt(nwp / 2) * d1
        rec_input = d1
    elif fillmethod == "value":
        if fillparams:
            val = fillparams[0]
        else:
            val = 0.0
        rec_input = val * np.ones(shp, dtype=data.dtype)
    else:
        rec_input = np.zeros(shp, dtype=data.dtype)

    rec_input[mask] = data

    out_data = np.fft.fft(rec_input, n=nchans, axis=0)
    if realout:
        out_data = out_data.real

    rec_array = np.zeros((ntime * nchans, subchan), dtype=out_data.dtype)

    M = coefs.shape[0]
    # Determine padding for filter

    h_len = (M - 1) // nchans

    n_pre_pad = nchans - (h_len % nchans)
    n_post_pad = 0

    n_pre_remove = (h_len + n_pre_pad)

    n_samps = ntime * nchans

    # Make sure you have enough samples.
    while (
        _output_len(len(coefs) + n_pre_pad + n_post_pad, n_samps, 1, nchans)
        < ntime + n_pre_remove
    ):
        n_post_pad += 1
    # Make sure length of filter will be multiple of nchans
    n_post_pad += nchans - ((M + n_pre_pad + n_post_pad) % nchans)
    h_dt = coefs.dtype
    h = np.concatenate(
        [np.zeros(n_pre_pad, dtype=h_dt), coefs, np.zeros(n_post_pad, dtype=h_dt)]
    )
    # Number of filter coefficients per channel
    M_c = (M + n_pre_pad + n_post_pad) // nchans
    # Reshape filter
    h = h.reshape((M_c, nchans)).T
    # Number of data samples per channel
    W = int(math.ceil(n_samps / M_c / nchans))
    # Array to be filled up
    x_summed = np.zeros((nchans, M_c * (W + 1) - 1, subchan), dtype=rec_array.dtype)
    nfull = nchans * W * M_c

    zfill = np.zeros((nchans, (nfull - n_samps) // nchans), dtype=rec_array.dtype)

    for isub in range(subchan):
        x_p = out_data[:, :, isub]
        x_p = np.append(x_p, zfill, axis=1)
        for p_i, (x_i, h_i) in enumerate(zip(x_p, h)):
            # Use correlate for filtering. Due to orientation of how filter is broken up.
            # Also using full signal to make sure padding and removal is done right.
            x_summed[p_i, :, isub] = sig.correlate(x_i, h_i, mode="full")

    for isub in range(subchan):
        x_out = x_summed[:, :, isub].T
        rec_array[:, isub] = x_out.flatten()[n_pre_remove:n_samps+n_pre_remove]

    return rec_array


def pfb_decompose(data, nchans, coefs, mask):
    """Polyphase filter function

    Takes the sampled and applies polyphase filter bank to channelize frequency content. Padding
    for filter is similar to scipy.signal.resample_poly, so output samples are shifted toward middle
    of array. The only channels kept are those list in the mask variable.

    Parameters
    ----------
    data : array_like
        A numpy array to be processed.
    nchans : int
        Number of output channes
    coefs : array_like
        Filter coefficients
    mask : array_like
        List of channels to be kept

    Returns
    -------
    xout : array_like
        The output from the polyphase.
    """

    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_samps, subchan = data.shape
    M = coefs.shape[0]
    # Determine padding for filter
    nout = n_samps // nchans + (n_samps % nchans > 0)
    h_len = (M - 1) // nchans

    n_pre_pad = nchans - (h_len % nchans)
    n_post_pad = 0

    n_pre_remove = (h_len + n_pre_pad) // nchans
    # Make sure you have enough samples.
    while (
        _output_len(len(coefs) + n_pre_pad + n_post_pad, n_samps, 1, nchans)
        < nout + n_pre_remove
    ):
        n_post_pad += 1
    # Make sure length of filter will be multiple of nchans
    n_post_pad += nchans - ((M + n_pre_pad + n_post_pad) % nchans)
    h_dt = coefs.dtype
    h = np.concatenate(
        [np.zeros(n_pre_pad, dtype=h_dt), coefs, np.zeros(n_post_pad, dtype=h_dt)]
    )
    # Number of filter coefficients per channel
    M_c = (M + n_pre_pad + n_post_pad) // nchans
    # Reshape filter
    h = h.reshape((M_c, nchans)).T
    # Number of data samples per channel
    W = int(math.ceil(n_samps / M_c / nchans))
    # Array to be filled up
    x_summed = np.zeros((nchans, M_c * (W + 1) - 1, subchan), dtype=np.complex64)
    nfull = nchans * W * M_c
    zfill = np.zeros(nfull - n_samps, dtype=data.dtype)
    # HACK see if I can do with without
    for isub in range(subchan):
        x_p = data[:, isub]
        x_p = np.append(x_p, zfill, axis=0)
        # make x_p frequency x time orientation
        x_p = x_p.reshape((W * M_c, nchans)).T
        for p_i, (x_i, h_i) in enumerate(zip(x_p, h)):
            # Use correlate for filtering. Due to orientation of how filter is broken up.
            # Also using full signal to make sure padding and removal is done right.
            x_summed[p_i, :, isub] = sig.correlate(x_i, h_i, mode="full")
    # now fchan x time x phychan
    xout = np.fft.fft(x_summed, n=nchans, axis=0)[
        mask, n_pre_remove : (nout + n_pre_remove)
    ]
    return xout
