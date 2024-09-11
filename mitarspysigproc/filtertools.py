from pathlib import Path
import scipy.signal as sig
import numpy as np
import scipy.special as scisp


def kaiser_coeffs(nchans, beta=1.7 * np.pi, pow2=True):
    """Creates a Kaiser window with a flat passband

    Parameters
    ----------
    nchans : int
        Number of frequency channels that the coefficients will be used for.
    beta : float
        Shape parameter for the filter.
    pow2 : bool
        Pad coefficients with zeros to the next power of 2.

    Returns
    -------
    taps : array_like
        Resulting filter taps.
    """

    ntaps = 64 * nchans
    # If you double this you can do this all with integers
    fs = nchans * 2
    if pow2:
        ntaps = int(np.power(2, np.ceil(np.log2(ntaps))))
    # Make odd number so you have type 1 filter.
    furry = sig.firwin(ntaps - 1, 1, window=("kaiser", beta), scale=True, fs=fs)
    taps = np.concatenate(([0], furry))
    return taps

    # sig.firwin(24*N, 0.5, window=('kaiser', 3*np.pi), scale=True, fs=N)


def kaiser_syn_coeffs(nchans, beta=1.7 * np.pi, pow2=True):
    """Creates a Kaiser window with a flat passband

    Parameters
    ----------
    nchans : int
        Number of frequency channels that the coefficients will be used for.
    beta : float
        Shape parameter for the filter.
    pow2 : bool
        Pad coefficients with zeros to the next power of 2.

    Returns
    -------
    taps : array_like
        Resulting filter taps.
    """

    ntaps = 64 * nchans
    # If you double this you can do this all with integers
    fs = nchans * 2
    if pow2:
        ntaps = int(np.power(2, np.ceil(np.log2(ntaps))))
    # Make odd number so you have type 1 filter.
    furry = sig.firwin(ntaps - 1, 1, window=("kaiser", beta), scale=True, fs=fs )
    taps = np.concatenate(([0], furry))
    taps = taps * nchans**2
    return taps


def kaiser_pfb_coefs(nchans, tpc=32, npr=False, beta=1.7 * np.pi):
    """Creates a Kaiser window with a flat passband with specific control for the

    Parameters
    ----------
    nchans : int
        Number of frequency channels that the coefficients will be used for.
    beta : float
        Shape parameter for the filter.
    npr : bool
        If yes then the number of rows is divided by 2 for the npr style filter.
    beta : float
        A shaping parameter for the kaiser functions. Default : 1.7*pi

    Returns
    -------
    taps : array_like
        Resulting filter taps as nchans x tpc array.
    """

    if npr:
        ncols = nchans // 2
    else:
        ncols = nchans

    ntaps = tpc * ncols
    # If you double this you can do this all with integers
    fs = nchans * 2

    # Make even number so you have type 2 filter.
    taps = sig.firwin(ntaps, 1, window=("kaiser", beta), scale=True, fs=fs)
    coeffs = taps.reshape((ncols, tpc), order="F")
    return coeffs


def createcoeffs(savedir):
    """Create a set of files for taps.

    Parameters
    ----------
    savedir : str
        Directory where the data will be saved.

    """
    chanarr = 2 ** np.arange(1, 11)
    maxchans = chanarr.max()
    maxchar = int(np.ceil(np.log10(maxchans)))
    suf_str = "{:0" + str(maxchar) + "}"
    savepath = Path(savedir)
    fstema = "kaiseranalysis" + suf_str + ".csv"
    fstems = "kaisersynthesis" + suf_str + ".csv"
    for ichans in chanarr:
        taps = kaiser_coeffs(ichans, pow2=False)

        fname = savepath.joinpath(fstema.format(ichans))
        np.savetxt(fname, taps, delimiter=",")

        if ichans >= 4:
            taps = kaiser_syn_coeffs(ichans, pow2=False) * ichans / 2
            fname = savepath.joinpath(fstems.format(ichans))
            np.savetxt(fname, taps, delimiter=",")


def rref_coef(N, L, K=None):
    """Creates a filter with the frequency response of the a root error function and sets it up for a npr filter bank. Based off the following: Wessel Lubberhuizen (2024). Near Perfect Reconstruction Polyphase Filterbank (https://www.mathworks.com/matlabcentral/fileexchange/15813-near-perfect-reconstruction-polyphase-filterbank), MATLAB Central File Exchange

    Parameters
    ----------
    N : int
        Number channels
    L : int
        Number of taps per channel.
    K : float
        A shaping factor but has default values

    Returns
    -------
    coeffs : ndarray
        A N//2 by L array of the coeffiecents.
    """
    k_dict = {
        8: 4.853,
        10: 4.775,
        12: 5.257,
        14: 5.736,
        16: 5.856,
        18: 7.037,
        20: 6.499,
        22: 6.483,
        24: 7.410,
        26: 7.022,
        28: 7.097,
        30: 7.755,
        32: 7.452,
        48: 8.522,
        64: 9.457,
        96: 10.785,
        128: 11.5,
        192: 11.5,
        256: 11.5,
    }

    if K is None:
        K = k_dict.get(L, 8.0)
    M = N // 2

    f = np.arange(0, L * M, dtype=np.float64) / (L * M)
    x = K * (2 * M * f - 0.5)
    A = np.sqrt(0.5 * scisp.erfc(x))
    a_n = len(A)
    n = np.arange(1, a_n / 2 + 1, dtype=int)
    A[-n] = np.conj(A[n])
    A[int(a_n / 2)] = 0
    B = np.fft.ifft(A)
    B = np.fft.fftshift(B).real

    return B.reshape((M, L), order="F")
