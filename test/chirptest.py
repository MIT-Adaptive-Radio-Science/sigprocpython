"""
Create a chirp  and test the PFB.

"""

from pathlib import Path
import numpy as np
import scipy.signal as sig
from mitarspysigproc import (
    pfb_decompose,
    pfb_reconstruct,
    kaiser_coeffs,
    kaiser_syn_coeffs,
    npr_analysis,
    npr_synthesis,
    rref_coef,
)


def create_chirp(t_len, fs, bw, pad, nchans, nslice):
    """Creates a chirp signal

    Parameters
    ----------
    t_len : float
        Length of chirp in seconds
    fs : float
        Sampling frequency in Hz
    bw : float
        Bandwidth of chirp
    nzeros : tuple
        Number of zeros to pad in the begining and end of the array.
    nchans : int
        Number of channels for the PFB
    nslice : int
        Number of time samples from the pfb

    Returns
    -------
    tout : ndarray
        The time vector for the created signal
    xout : ndarray
        Created signal
    """
    nar = (
        np.arange(int(-nslice * nchans / 2), int(nslice * nchans / 2), dtype=float)
        / nslice
        / nchans
    )
    t = np.linspace(-t_len / 2, t_len / 2, int(t_len * fs))
    dphi = 2 * np.pi * nar * bw / fs
    phi = np.mod(np.cumsum(dphi), 2 * np.pi)
    x = np.exp(-1j * phi)
    # x = sig.chirp(t,t1=t_len,f0=0,f1=bw,method='linear')

    xout = np.concatenate((pad[0], x, pad[1]), axis=0)
    tp1 = -1 * np.arange(0, len(pad[0]), dtype=float)[::-1] / fs - t_len / 2
    tp2 = np.arange(0, len(pad[1]), dtype=float) / fs + t_len / 2
    tout = np.concatenate((tp1, t, tp2), axis=0)

    return tout, xout


def runchirptest(t_len, fs, bw, nzeros, nchans, nslice):
    """Creates a chirp and runs the standard PFB analysis and reconstruction

    Parameters
    ----------
    t_len : float
        Length of chirp in seconds
    fs : float
        Sampling frequency in Hz
    bw : float
        Bandwidth of chirp
    nzeros : int
        Number of zeros to pad
    nchans : int
        Number of channels for the PFB
    nslice : int
        Number of time samples from the pfb

    Returns
    -------
    x_rec : ndarray
        Reconstructed signal
    tin : ndarray
        The time vector for the input signal
    x : ndarray
        Input signal
    x_pfb : ndarray
        The result of the PFB analysis in an nchans x slice array.
    """
    pad = [np.zeros(nzeros), np.zeros(nzeros)]
    t, x = create_chirp(t_len, fs, bw, pad, nchans, nslice)

    coeffs = kaiser_coeffs(nchans, 8.0)
    mask = np.ones(nchans, dtype=bool)
    xout = pfb_decompose(x, nchans, coeffs, mask)
    fillmethod = ""
    fillparams = [0, 0]
    syn_coeffs = kaiser_syn_coeffs(nchans, 8)
    x_rec = pfb_reconstruct(
        xout, nchans, syn_coeffs, mask, fillmethod, fillparams=[], realout=False
    )
    return x_rec, t, x, xout


def runnprchirptest(t_len, fs, bw, nzeros, nchans, nslice, ntaps=64):
    """Creates a chirp and runs the near perfect PFB analysis and reconstruction

    Parameters
    ----------
    t_len : float
        Length of chirp in seconds
    fs : float
        Sampling frequency in Hz
    bw : float
        Bandwidth of chirp
    nchans : int
        Number of channels for the PFB
    nslice : int
        Number of time samples from the pfb

    Returns
    -------
    x_rec : ndarray
        Reconstructed signal
    tin : ndarray
        The time vector for the input signal
    x : ndarray
        Input signal
    x_pfb : ndarray
        The result of the PFB analysis in an nchans x slice array.
    """
    pad = [np.zeros(nzeros), np.zeros(nzeros)]
    t, x = create_chirp(t_len, fs, bw, pad, nchans, nslice)
    coeffs = rref_coef(nchans, ntaps)
    mask = np.ones(nchans, dtype=bool)
    xout = npr_analysis(x, nchans, coeffs)
    fillmethod = ""
    fillparams = [0, 0]
    x_rec = npr_synthesis(xout, nchans, coeffs)
    return x_rec, t, x, xout







def runexample():
    """Function for running each of the examples."""
    nchans = 64
    nslice = 2048
    fs = 10000
    t_len = nchans * nslice / fs
    bw = 2000
    ntaps = 64
    g_del = nchans * (ntaps - 1) // 2
    nzeros = 2048

    x_rec, t, x, xpfb = runchirptest(t_len, fs, bw, nzeros*2, nchans, nslice)
    x_rec = np.roll(x_rec, -1*nchans*ntaps)


    x_rec, t, x, xpfb = runnprchirptest(t_len, fs, bw, nzeros, nchans, nslice, ntaps)
    x_rec = x_rec[: len(x), np.newaxis]  # need to add new axis due to plotting issue
    x_rec = np.roll(x_rec, -1*g_del)

if __name__ == "__main__":
    runexample()
