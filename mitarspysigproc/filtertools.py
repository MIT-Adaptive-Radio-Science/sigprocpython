from pathlib import Path
import scipy.signal as sig
import numpy as np


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
    furry = sig.firwin(ntaps - 1, 1, window=("kaiser", beta), scale=True, fs=fs)
    taps = np.concatenate(([0], furry))
    taps = taps*nchans**2
    return taps


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
