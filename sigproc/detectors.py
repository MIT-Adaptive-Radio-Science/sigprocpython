#!/usr/bin/env python
"""
detectors.py
Code for constant false alarm rate and other detectors.
"""
import numpy as np
import scipy.special as sc


def cfar(data, cfar_width, thresh):
    """Run the cfar detector on a data set with the same threshold unit.

    Runs a constant false alarm detector in one dimension.
    Parameters
    ----------
    data : array_like
        Data array that will be examined.
    cfar_width : int
        Width of the CFAR array.
    thresh : float
        Threshold for making a detection.

    Returns
    -------
    outlist : list
        List of dictionaries with detections.
    """
    outlist = []
    ncells, nframes = data.shape
    # Get the median for the whole data set
    med_all = np.nanmedian(data.flatten())
    outdict = dict(cell=[], frame=[], snr=[], npow=[])
    for iframe in range(nframes):
        curframe = data[:, iframe]
        for icell in range(ncells):
            lbnd = min(ncells - cfar_width, max(icell - cfar_width // 2, 0))
            ubnd = max(cfar_width, min(icell + cfar_width // 2, ncells))
            val = curframe[icell]
            floor = max(np.nanmedian(curframe[lbnd:ubnd]), med_all)
            if val >= thresh + floor:

                # d1 = dict(cell=icell,frame=iframe,snr=val-floor,npow=floor)
                outdict["cell"].append(icell)
                outdict["frame"].append(iframe)
                outdict["snr"].append(val - floor)
                outdict["npow"].append(floor)
                # outlist.append(d1)
    return outdict


def erlang_ratio(pfa, k):
    """Get the multiplier for a detector for an Erlang distributed variable.

    Create the threshold for a erlang random variable by using inverse lower incomplete gamma functions.

    """

    gm_num = sc.gammaincinv(k, 1.0 - pfa)
    gm_den = sc.gammaincinv(k, 0.5)

    return gm_num / gm_den
