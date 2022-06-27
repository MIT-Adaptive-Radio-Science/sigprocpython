from pathlib import Path
import scipy.signal as sig
import numpy as np


def kaiser(nchans, beta=1.7 * np.pi, pow2=True):
    """Creates a Kaiser window with a flat passband"""

    ntaps = 24 * nchans
    # If you double this you can do this all with integers
    fs = nchans * 2
    if pow2:
        ntaps = int(np.power(2, np.ceil(np.log2(ntaps))))
    # Make odd number so you have type 1 filter.
    furry = sig.firwin(ntaps - 1, 1, window=("kaiser", beta), scale=True, fs=fs)
    taps = np.concatenate(([0], furry))
    return taps

    # sig.firwin(24*N, 0.5, window=('kaiser', 3*np.pi), scale=True, fs=N)


def kaisersyn(nchans, beta=1.7 * np.pi, pow2=True):
    """Creates a Kaiser window with a flat passband"""

    ntaps = 12 * nchans
    # If you double this you can do this all with integers
    fs = nchans * 2
    if pow2:
        ntaps = int(np.power(2, np.ceil(np.log2(ntaps))))
    # Make odd number so you have type 1 filter.
    furry = sig.firwin(ntaps - 1, 1, window=("kaiser", beta), scale=True, fs=nchans)
    taps = np.concatenate(([0], furry))
    return taps


def createcoeffs(savedir):
    """ """
    chanarr = 2 ** np.arange(1, 11)
    maxchans = chanarr.max()
    maxchar = int(np.ceil(np.log10(maxchans)))
    suf_str = "{:0" + str(maxchar) + "}"
    savepath = Path(savedir)
    fstema = "kaiseranalysis" + suf_str + ".csv"
    fstems = "kaisersynthesis" + suf_str + ".csv"
    for ichans in chanarr:
        taps = kaiser(ichans, pow2=False)

        fname = savepath.joinpath(fstema.format(ichans))
        np.savetxt(fname, taps, delimiter=",")

        if ichans >= 4:
            taps = kaisersyn(ichans, pow2=False) * ichans / 2
            fname = savepath.joinpath(fstems.format(ichans))
            np.savetxt(fname, taps, delimiter=",")
