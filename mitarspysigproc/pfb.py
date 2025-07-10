from pathlib import Path
import math
from copy import copy
import numpy as np

import scipy.signal as sig
from scipy.signal._upfirdn_apply import _output_len
import matplotlib.pyplot as plt

chanstocoef = {64: "kaiser64.csv"}


def pfbresponse(taps, nchans, fs):
    """Creates the frequency response of a pfb given the taps.

    Parameters
    ----------
    taps : str or ndarray
        If string will treat it as a csv file. If array treats it as taps to the filter.
    nchans : int
        The number of channesl for the pfb.
    fs : float
        Original sampling frequency of the signal in Hz.

    Returns
    -------
    freq_ar : ndarray
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

    nfreq = 2**11
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
    freq_ar : ndarray
        Frequency vector in Hz.
    filt_dict : dict
        Keys are filter number, values are frequency response in dB
    nchans : int
        The number of channesl for the pfb.

    Returns
    -------
    fig : figure
        Image of frequency response.
    axs : ndarray
        List of the axes that teh frequency response plotted on.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    mult, ack, _ = prefix(np.abs(freq_ar).max())
    fs_h = np.max(np.abs(freq_ar))
    f_wid = 1.25 * fs_h / nchans

    filtnum = 0
    hpow = filt_dict[filtnum]
    hplot = axs[0].plot(freq_ar * mult, hpow, linewidth=3)[0]
    axs[0].set_xlim([-f_wid, f_wid])
    axs[0].set_ylim([-65, 5])
    axs[0].set_xlabel("Frequency {0}Hz".format(ack))
    axs[0].set_ylabel("Magnitude dB")
    axs[0].grid(True)
    axs[0].set_title("Single Filter")

    hlist = []
    name_list = []
    for filtnum, hpow in filt_dict.items():
        hplot = axs[1].plot(freq_ar * mult, hpow, linewidth=3)[0]
        hlist.append(hplot)
        name_list.append("Channel {}".format(filtnum))
        axs[1].set_xlim([-3 * f_wid, 3 * f_wid])
        axs[1].set_ylim([-65, 5])
        axs[1].set_xlabel("Frequency {0}Hz".format(ack))
        axs[1].set_ylabel("Magnitude dB")
        axs[1].set_title("Multiple Filters")

    axs[1].legend(hlist, name_list)
    axs[1].grid(True)

    return fig, axs


def pfb_dec_simp(data, nchans, coeffs):
    """
    Parameters
    ----------
    data : ndarray
        A numpy array to be processed.
    nchans : int
        Number of output channes
    coefs : ndarray
        Filter coefficients in nchan x tpc array.

    Returns
    -------
    xout : ndarray
        The output from the polyphase.
    """
    remove_sub = False
    if data.ndim == 1:
        data = data[:, np.newaxis]
        remove_sub = True

    n_samps, subchan = data.shape
    nout = n_samps // nchans + (n_samps % nchans > 0)

    h = coeffs.copy()
    nchans_check, M_c = h.shape
    if nchans_check != nchans:
        raise ValueError(
            "Filter coefficients axis 0 size must equal number of pfb channels."
        )

    n_pre_remove = (M_c) // 2
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
        x_p = x_p.reshape((W * M_c, nchans)).T[::-1]
        for p_i, (x_i, h_i) in enumerate(zip(x_p, h)):
            # Use correlate for filtering. Due to orientation of how filter is broken up.
            # Also using full signal to make sure padding and removal is done right.
            x_summed[p_i, :, isub] = sig.convolve(x_i, h_i, mode="full")
    # now fchan x time x phychan
    xout = np.fft.ifft(x_summed, n=nchans, axis=0)[
        :, n_pre_remove : (nout + n_pre_remove)
    ]
    if remove_sub:
        xout = xout[..., 0]

    return xout


def pfb_rec_simp(data, nchans, coeffs, mask, fillmethod, fillparams=[], realout=True):
    """
    Parameters
    ----------
    data : ndarray
        A numpy array to be processed.
    nchans : int
        Number of output channes
    coefs : ndarray
        Filter coefficients in nchan x tpc array.

    Returns
    -------
    rec_array : ndarray
        The output from the polyphase synthesis.
    """

    remove_sub = False
    if data.ndim == 2:
        data = data[..., np.newaxis]
        remove_sub = True

    _, ntime, subchan = data.shape
    shp = (nchans, ntime, subchan)
    if fillmethod == "noise":
        if fillparams:
            npw = fillparams[0]
        else:
            npw = np.nanmedian(data.flatten().real ** 2 + data.flatten().imag ** 2)
            npw = npw / np.log(2)
        d1r = np.random.randn(*shp, dtype=data.dtype)
        d1i =  np.random.randn(*shp, dtype=data.dtype)
        d1 = d1r + 1j*d1i

        d1 = np.sqrt(npw / 2) * d1
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

    out_data = np.fft.ifft(rec_input, n=nchans, axis=0)
    if realout:
        out_data = out_data.real
    n_samps = ntime * nchans

    rec_array = np.zeros((n_samps, subchan), dtype=out_data.dtype)

    h = coeffs.copy()
    nchans_check, M_c = h.shape
    if nchans_check != nchans:
        raise ValueError(
            "Filter coefficients axis 0 size must equal number of pfb channels."
        )

    n_pre_remove = (M_c - 1) // 2
    # Number of data samples per channel
    W = int(math.ceil(n_samps / M_c / nchans))
    # Array to be filled up
    x_summed = np.zeros((nchans, M_c * (W + 1) - 1, subchan), dtype=rec_array.dtype)
    nfull = nchans * W * M_c

    zfill = np.zeros((nchans, (nfull - n_samps) // nchans), dtype=rec_array.dtype)

    # import ipdb
    # ipdb.set_trace()
    for isub in range(subchan):
        x_p = out_data[:, :, isub]
        # x_odd = np.fft.fftshift(x_p[:,1::2],axes=0)
        # x_p[:,1::2] = x_odd
        x_p = np.append(x_p, zfill, axis=1)
        for p_i, (x_i, h_i) in enumerate(zip(x_p, h)):
            # Use correlate for filtering. Due to orientation of how filter is broken up.
            # Also using full signal to make sure padding and removal is done right.
            x_summed[p_i, :, isub] = sig.convolve(x_i, h_i, mode="full")

    for isub in range(subchan):
        # x_out = np.fft.fftshift(np.fft.fft(x_summed[:, n_pre_remove:(n_samps//nchans)+n_pre_remove, isub].T,axis=1).real,axes=1)
        # rec_array[:, isub] = x_out.flatten()
        x_out = x_summed[:, n_pre_remove : ntime + n_pre_remove, isub].T
        rec_array[:, isub] = x_out.flatten()

    if remove_sub:
        rec_array = rec_array[..., 0]

    return rec_array


def pfb_decompose(data, nchans, coefs, mask):
    """Polyphase filter function

    Takes the sampled and applies polyphase filter bank to channelize frequency content. Padding for filter is similar to scipy.signal.resample_poly, so output samples are shifted toward middle of the array. The only channels kept are those list in the mask variable.

    Parameters
    ----------
    data : ndarray
        A numpy array to be processed.
    nchans : int
        Number of output channes
    coefs : ndarray
        Filter coefficients
    mask : ndarray
        List of channels to be kept

    Returns
    -------
    xout : ndarray
        The output from the polyphase.
    """

    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_samps, subchan = data.shape
    M = coefs.shape[0]
    # Determine padding for filter
    nout = n_samps // nchans + (n_samps % nchans > 0)
    # Figure out half length
    if M % 2:
        h_len = (M - 1) // 2
    else:
        h_len = M // 2

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
    n_start = (M - 1 + n_pre_pad) // 2 // nchans + (
        (((M - 1 + n_pre_pad) // 2) % nchans) > 0
    )
    # Number of filter coefficients per channel
    M_c = (M + n_pre_pad + n_post_pad) // nchans
    # Reshape filter
    h = h.reshape((nchans, M_c), order="F")[:, ::-1]
    # Number of data samples per channel
    W = int(math.ceil(n_samps / M_c / nchans))
    # Array to be filled up
    x_summed = np.zeros((nchans, M_c * W, subchan), dtype=data.dtype)
    # x_summed = np.zeros((nchans, M_c * (W + 1) - 1, subchan), dtype=np.complex64)
    nfull = nchans * W * M_c
    zfill = np.zeros(nfull - n_samps, dtype=data.dtype)
    # HACK see if I can do with without
    for isub in range(subchan):
        x_p = data[:, isub]
        x_p = np.append(x_p, zfill, axis=0)
        # make x_p frequency x time orientation
        x_p = x_p.reshape((nchans, W * M_c), order="F")
        for p_i, (x_i, h_i) in enumerate(zip(x_p, h)):
            # Use correlate for filtering. Due to orientation of how filter is broken up.
            # Also using full signal to make sure padding and removal is done right.
            x_summed[p_i, :, isub] = sig.lfilter(
                h_i,
                1,
                x_i,
            )
    # now fchan x time x phychan
    xout = np.fft.ifft(x_summed, n=nchans, axis=0)[
        mask, :  # n_pre_remove : (nout + n_pre_remove)
    ]
    return xout


def pfb_reconstruct(data, nchans, coefs, mask, fillmethod, fillparams=[], realout=True):
    """Simple PFB reconstruction

    Parameters
    ----------
    data : ndarray
        A numpy array to be processed.
    nchans : int
        Number of output channes
    coefs : ndarray
        Filter coefficients
    mask : ndarray
        List of channels to be kept
    fillmethod : string
        Type of filled in the data before the reconstrution.
    fillparams : list
        Parameters for filling in empty data

    Returns
    -------
    rec_array : ndarray
        The output from the polyphase synthesis.
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
        d1 = np.sqrt(npw / 2) * d1
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

    rec_array = np.zeros((ntime * nchans, subchan), dtype=out_data.dtype)

    M = coefs.shape[0]
    # Determine padding for filter

    # Figure out half length
    if M % 2:
        h_len = (M - 1) // 2
    else:
        h_len = M // 2

    n_pre_pad = nchans - (h_len % nchans)
    n_pre_pad = 0
    n_post_pad = 0

    n_pre_remove = (h_len + n_pre_pad) // nchans
    n_samps = ntime * nchans

    # Make sure you have enough samples.
    while (
        _output_len(len(coefs) + n_pre_pad + n_post_pad, n_samps, 1, nchans)
        < ntime + n_pre_remove
    ):
        n_post_pad += 1

    # Make sure length of filter will be multiple of nchans
    n_post_pad += nchans - ((M + n_pre_pad + n_post_pad) % nchans)
    n_post_pad = n_post_pad % nchans
    h_dt = coefs.dtype

    h_full = np.concatenate(
        [
            np.zeros(n_pre_pad, dtype=data.dtype),
            coefs.astype(data.dtype),
            np.zeros(n_post_pad, dtype=data.dtype),
        ]
    )
    # Number of filter coefficients per channel
    M_c = (M + n_pre_pad + n_post_pad) // nchans
    # Reshape filter
    h_full = h_full.reshape((nchans, M_c), order="F")

    # Number of data samples per channel
    W = int(math.ceil(n_samps / M_c / nchans))
    # Array to be filled up
    x_summed = np.zeros((nchans, M_c * W, subchan), dtype=rec_array.dtype)
    nfull = nchans * W * M_c

    nadd = (nfull - n_samps) // nchans
    zfill = np.zeros((nchans, nadd), dtype=rec_array.dtype)

    # import ipdb
    # ipdb.set_trace()
    for isub in range(subchan):
        x_p = out_data[:, :, isub]
        # x_odd = np.fft.fftshift(x_p[:,1::2],axes=0)
        # x_p[:,1::2] = x_odd
        x_p = np.append(x_p, zfill, axis=1)
        for p_i, (x_i, h_i) in enumerate(zip(x_p, h_full)):
            # Use correlate for filtering. Due to orientation of how filter is broken up.
            # Also using full signal to make sure padding and removal is done right.
            x_summed[p_i, :, isub] = sig.lfilter(h_i, 1, x_i)

    for isub in range(subchan):
        # x_out = np.fft.fftshift(np.fft.fft(x_summed[:, n_pre_remove:(n_samps//nchans)+n_pre_remove, isub].T,axis=1).real,axes=1)
        # rec_array[:, isub] = x_out.flatten()
        x_out = x_summed[
            :, :, isub
        ]  # n_pre_remove: (n_samps // nchans) + n_pre_remove, isub]
        rec_array[:, isub] = x_out.flatten(order="F")[: (nfull - nadd * nchans)]

    if realout:
        rec_array = rec_array.real
    return rec_array


# %%
def npr_analysis(xin, nchans, coeffs):
    """PFB analysis using near perfect reconcsturction method.

    Performs a poly phase filter bank channelization that allows for near perfect reconstruction. This is done by creating overlaping channels to avoid any lossed from the filtering. The method is based off of this: Wessel Lubberhuizen (2024). Near Perfect Reconstruction Polyphase Filterbank (https://www.mathworks.com/matlabcentral/fileexchange/15813-near-perfect-reconstruction-polyphase-filterbank), MATLAB Central File Exchange

    Parameters
    ----------
    xin : ndarray
        The data that will be channelized along its first dimention.
    nchans : int
        Number of channels for the polyphase.
    coeffs : ndarray
        Filter coefficients, first dimension must be nchans//2.

    Returns
    -------
    yout : ndarray
        The output from the polyphase channelizer in the shape of nchans by xin.shape[0]/(nchans/2)
    """

    nfft = nchans // 2
    remove_sub = False
    if xin.ndim == 1:
        xin = xin[:, np.newaxis]
        remove_sub = True

    n_samps, subchan = xin.shape
    nout = n_samps // nfft + (n_samps % nfft > 0)

    h = coeffs.copy()

    nchans_check, M_c = h.shape
    if nchans_check != nfft:
        raise ValueError(
            "Filter coefficients axis 0 size must equal number of pfb channels."
        )

    # Change xin to work with the new type
    xin = xin.astype(np.complex128)
    # since xin has a sub channel use this
    # xin_osc = osc[:,np.newaxis]*xin

    h_comp = h.astype(xin.dtype)
    # Flip the filter coefficents in time around for the analysis
    h_comp = h_comp[:, ::-1]
    # Number of data samples per channel
    W = int(math.ceil(n_samps / M_c / nfft))
    x_summed0 = np.zeros((nfft, M_c * W, subchan), dtype=xin.dtype)
    x_summed1 = np.zeros((nfft, M_c * W, subchan), dtype=xin.dtype)

    nfull = nfft * W * M_c
    nslice = int(math.ceil(nfull / nfft))
    zfill = np.zeros(nfull - n_samps, dtype=xin.dtype)
    phi = np.mod(np.arange(nfull), 2 * nfft).astype(float) * np.pi / nfft
    osc = np.exp(1j * phi).astype(np.complex128)
    # HACK see if I can do with without the loop
    for isub in range(subchan):
        x_p0 = np.append(xin[:, isub], zfill, axis=0)
        x_p1 = x_p0 * osc
        # make x_p frequency x time orientation
        x_p0 = x_p0.reshape((nfft, nslice), order="F")
        x_p1 = x_p1.reshape((nfft, nslice), order="F")
        for p_i, (x_i0, x_i1, h_i) in enumerate(zip(x_p0, x_p1, h_comp)):
            # Use lfilter
            x_summed0[p_i, :, isub] = sig.lfilter(h_i, 1, x_i0)
            x_summed1[p_i, :, isub] = sig.lfilter(
                h_i, 1, x_i1
            )  # now fchan x time x phychan
    xout0 = np.fft.ifft(x_summed0, n=nfft, axis=0) * nfft
    xout1 = np.fft.ifft(x_summed1, n=nfft, axis=0) * nfft
    xout = np.zeros((nchans, nslice, subchan), dtype=xout0.dtype)
    xout[0:nchans:2] = xout0
    xout[1:nchans:2] = xout1
    if remove_sub:
        yout = xout[..., 0]

    return yout


# %%
def npr_synthesis(yin, nchans, coeffs, realout=False):
    """PFB synthesis using near perfect reconcsturction method.

    Performs a poly phase filter bank synthesis that allows for near perfect reconstruction. The method is based off of this: Wessel Lubberhuizen (2024). Near Perfect Reconstruction Polyphase Filterbank (https://www.mathworks.com/matlabcentral/fileexchange/15813-near-perfect-reconstruction-polyphase-filterbank), MATLAB Central File Exchange

    Parameters
    ----------
    yin : ndarray
        The data that will be channelized along its first dimension.
    nchans : int
        Number of channels for the polyphase.
    coeffs : ndarray
        Filter coefficients, first dimension must be nchans//2.
    realout : bool
        Will output a real signal if set to true.

    Returns
    -------
    xout : ndarray
        The output from the polyphase synthesis.
    """
    nfft = nchans // 2
    remove_sub = False
    if yin.ndim == 2:
        yin = yin[..., np.newaxis]
        remove_sub = True

    _, ntime, subchan = yin.shape

    shp = (nchans, ntime, subchan)
    yin_e = yin[0:nchans:2]
    yin_o = yin[1:nchans:2]

    xe = np.fft.fft(yin_e, n=nfft, axis=0) * nfft
    xo = np.fft.fft(yin_o, n=nfft, axis=0) * nfft

    n_samps = ntime * nfft

    h = coeffs.copy()
    nchans_check, M_c = h.shape
    if nchans_check != nfft:
        raise ValueError(
            "Filter coefficients axis 0 size must equal number of pfb channels."
        )

    # Number of data samples per channel
    W = int(math.ceil(n_samps / M_c / nfft))
    # Array to be filled up
    x_summede = np.zeros((nfft, M_c * W, subchan), dtype=xe.dtype)

    x_summedo = np.zeros((nfft, M_c * W, subchan), dtype=xe.dtype)

    nfull = nfft * W * M_c
    xout = np.zeros((nfull, subchan), dtype=xe.dtype)

    zfill = np.zeros((nfft, (nfull - n_samps) // nfft), dtype=xout.dtype)
    phi = np.mod(np.arange(nfull), nchans).astype(float) * np.pi / nfft
    osterm = np.exp(-1j * phi)
    # import ipdb
    # ipdb.set_trace()
    for isub in range(subchan):
        x_pe = xe[:, :, isub]
        x_po = xo[:, :, isub]
        # x_odd = np.fft.fftshift(x_p[:,1::2],axes=0)
        # x_p[:,1::2] = x_odd
        x_pe = np.append(x_pe, zfill, axis=1)
        x_po = np.append(x_po, zfill, axis=1)
        for p_i, (x_ie, x_io, h_i) in enumerate(zip(x_pe, x_po, h)):
            # Use correlate for filtering. Due to orientation of how filter is broken up.
            # Also using full signal to make sure padding and removal is done right.
            x_summedo[p_i, :, isub] = sig.lfilter(h_i, 1, x_io)
            x_summede[p_i, :, isub] = sig.lfilter(h_i, 1, x_ie)

    for isub in range(subchan):
        # x_out = np.fft.fftshift(np.fft.fft(x_summed[:, n_pre_remove:(n_samps//nchans)+n_pre_remove, isub].T,axis=1).real,axes=1)
        # xout[:, isub] = x_out.flatten()
        x_oute = x_summede[:, : nfull // nfft, isub].T
        x_outo = x_summedo[:, : nfull // nfft, isub].T
        xout[:, isub] = x_oute.flatten() - x_outo.flatten() * osterm

    if remove_sub:
        xout = xout[..., 0]
    if realout:
        xout = xout.real
    return xout
