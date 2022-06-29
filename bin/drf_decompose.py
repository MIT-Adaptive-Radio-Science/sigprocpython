#!/usr/bin/env python
"""

"""
import argparse
from pathlib import Path

import sys
import shutil
import math
import numpy as np
import digital_rf as drf

from mitarspysigproc import pfb_decompose, kaiser_coeffs

import ipdb


def parse_command_line(str_input=None):
    """This will parse through the command line arguments

    Function to go through the command line and if given a list of strings all also output a namespace object.

    Parameters
    ----------
    str_input : list
        A list of strings from the command line. If none will read from commandline. Can just take command line inputs and do a split() on them.

    Returns
    -------
    input_args : Namespace
        An object holding the input arguments wrt the variables.
    """
    scriptname = Path(sys.argv[0]).name

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width
    title = "DRF PFB"
    shortdesc = (
        "Run a polyphase filter bank and break up data into frequency channels. "
    )
    desc = "\n".join(
        (
            "*" * width,
            "*{0:^{1}}*".format(title, width - 2),
            "*{0:^{1}}*".format("", width - 2),
            "*{0:^{1}}*".format(shortdesc, width - 2),
            "*" * width,
        )
    )
    # desc = "This is the run script for SimVSR."
    # if str_input is None:
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # else:
    #     parser = argparse.ArgumentParser(str_input)

    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        required=True,
        help="Name of the digital RF data set directory.",
    )
    parser.add_argument(
        "-n",
        "--pfbchans",
        dest="pfbchans",
        help="Number of pfb channels",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--channame",
        dest="channame",
        help="Name of channel with sub channel seperated by a colon.",
        default=None,
    )

    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        required=False,
        default="",
        type=str,
        help="Altitude in km for the average orbit.",
    )
    parser.add_argument(
        "-m", "--mask", dest="mask", required=False, type=str, default=""
    )
    parser.add_argument(
        "-s",
        "--starttime",
        dest="starttime",
        default=None,
        help="Use the provided start time instead of the first time in the data. format is ISO8601: 2015-11-01T15:24:00Z",
    )
    parser.add_argument(
        "-e",
        "--endtime",
        dest="endtime",
        default=None,
        help="Use the provided end time for the plot. format is ISO8601: 2015-11-01T15:24:00Z",
    )

    if str_input is None:
        return parser.parse_args()
    return parser.parse_args(str_input)


def setuppfbout(datadir, chan, signal_params, pfb_params, st):
    """Sets up the digital_rf writer objects for each pfb channel.

    Parameters
    ----------
    datadir : str
        Location of the director where the data will be stored.
    chan : str
        Name of the channel that will be read
    signal_params : dict
        Dictionary from the get_properties function of the drf reader.
    pfb_params : dict
        Dictionary holding info related to the pfb, filter coefiecents, number of channels, etc.
    st : datetime
        Start time of the decomposition.

    Returns
    -------
    pfb_out : dict
        Dictionary hold DigitalRFWriters.
    pfb_obj : dict
        Dictionary holding metadata writters.
    """

    datapath = Path(datadir)

    sr_nfs = signal_params["sample_rate_numerator"]
    sr_dfs = signal_params["sample_rate_denominator"]
    n_subchannels = signal_params["num_subchannels"]

    common_params = {
        "subdir_cadence_secs": 3600,  # Number of seconds of data in a subdirectory
        "file_cadence_millisecs": 10000,  # Each file will have up to 400 ms of data
        "compression_level": 0,  # no compression
        "checksum": False,  # no checksum
        "is_continuous": True,
        "num_subchannels": n_subchannels,
        "marching_periods": False,  # no marching periods when writing
        "sample_rate_numerator": sr_nfs,
    }
    # Common parameters
    sub_cadence_secs = common_params[
        "subdir_cadence_secs"
    ]  # Number of seconds of data in a subdirectory
    file_cadence_secs = (
        common_params["file_cadence_millisecs"] // 1000
    )  # Each file will have up to 400 ms of data
    compression_level = 0  # no compression
    checksum = False  # no checksum

    is_continuous = True
    marching_periods = False  # no marching periods when writing
    # get channels

    # pfb

    sr_div = pfb_params["nchans"]
    freq_map = np.fft.fftfreq(sr_div, d=float(sr_dfs) / sr_nfs)
    pfb_out = {}
    pfb_obj = {}
    ndig = int(np.log10(sr_div)) + 1
    chan_list = np.arange(sr_div)
    for ichan in chan_list:
        if not ichan in pfb_params["mask"]:
            continue
        # HACK how many channels will we need?
        iname = chan + "_" + str(ichan).zfill(ndig)
        cdir = datapath.joinpath(iname)
        if cdir.is_dir():
            shutil.rmtree(str(cdir))
        cdir.mkdir(parents=True, exist_ok=True)
        start_indx_fs = drf.util.datetime_to_timestamp(st) * sr_nfs / sr_dfs
        pfb_info = common_params.copy()
        pfb_info["directory"] = str(cdir)

        pfb_info["start_global_index"] = int(start_indx_fs / sr_div)
        pfb_info["sample_rate_denominator"] = sr_div * sr_dfs
        pfb_info["uuid_str"] = "pfb"
        pfb_info["dtype"] = np.complex64  # complex64
        pfb_info["is_complex"] = True  # complex values
        pfb_out[ichan] = drf.DigitalRFWriter(**pfb_info)

        # Create meta data
        pfbmetadir = cdir.joinpath("metadata")
        if pfbmetadir.is_dir():
            shutil.rmtree(str(pfbmetadir))
        pfbmetadir.mkdir(parents=True, exist_ok=True)

        pfb_obj[ichan] = drf.DigitalMetadataWriter(
            str(pfbmetadir),
            sub_cadence_secs,
            file_cadence_secs,
            sr_nfs,
            sr_div * sr_dfs,
            "pfb",
        )
        pfbap = pfb_params.copy()
        cur_freq = [freq_map[ichan]] * n_subchannels
        pfbap["center_frequencies"] = np.array(cur_freq)
        pfb_obj[ichan].write(pfb_info["start_global_index"], pfbap)
    return pfb_out, pfb_obj


def runpfb(path, pfbchans, channame, outdir="", mask="", starttime="", endtime=""):
    """Read in a Digital RF data set and decompose data using a polyphase filter bank.

    Parameters
    ----------
    path : str
        Path to the input data set.
    pfbchans : int
        Number of frequency channels the data is to be broken into
    channame : str
        Name of channel with sub channel seperated by a colon.
    outdir : str
        Directory that the decompsoed data will be written to.
    mask : str
        Which frequency channels, numbered 0 to pfbchans-1, to keep.
    starttime : str
        Start time instead of the first time in the data. format is ISO8601: 2015-11-01T15:24:00Z
    endtime : str
        End time instead of the last time in the data. format is ISO8601: 2015-11-01T15:24:00Z
    """

    # Parameter fixes
    if outdir == "":
        outdir = path.copy()

    if mask == "":
        mask = "0-{0}".format(pfbchans)
        mask_arr = np.arange(pfbchans)
    elif "-" in mask:
        beg, fin = mask.split("-")
        mask_arr = np.arange(int(beg), int(fin))
    elif "," in mask:
        maskstr = mask.split(",")
        mask_arr = np.array([int(i) for i in maskstr])

    drfObj = drf.DigitalRFReader(str(path))
    chans = drfObj.get_channels()
    if channame == "":
        chan = chans[0]
        subchan = 0
    else:
        chan, subchan = channame.split(":")
        subchan = int(subchan)

    # Get the signal properties
    signal_params = drfObj.get_properties(chan)
    st_f, et_f = drfObj.get_bounds(chan)
    sr = signal_params["samples_per_second"]

    if starttime == "":
        st = drf.util.sample_to_datetime(st_f, sr)
        st_samp = st_f
    else:
        st = drf.util.parse_identifier_to_time(starttime, sr)
        st_samp = int(drf.util.datetime_to_timestamp(st) * sr)

    if endtime == "":
        et = drf.util.sample_to_datetime(et_f, sr)
        et_samp = et_f
    else:
        et = drf.util.parse_identifier_to_time(endtime, sr)
        et_samp = int(drf.util.datetime_to_timestamp(et) * sr)

    sam_per_read = int(32 * 2**20 / 4)

    read_time = np.arange(st_samp, et_samp, sam_per_read)
    pfb_params = {
        "nchans": pfbchans,
        "coefs": kaiser_coeffs(pfbchans),
        "mask": mask_arr,
    }
    pfb_coefs = pfb_params["coefs"]

    # Normalize the coefiecents to try to keep the variances
    pfb_coefs = pfb_coefs / np.sqrt(np.sum(np.power(pfb_coefs, 2)))
    n_coefs = pfb_coefs.size
    M = n_coefs + ((n_coefs + 1) % 2)
    # Determine the amount of overlap needed after the filtering to keep things continuous between reads.
    max_overlap = M

    h_len = (M - 1) // 2
    ds_fac = pfb_params["nchans"]
    h_ds = h_len // ds_fac + (h_len % ds_fac > 0)
    channel_taps = math.ceil(n_coefs / ds_fac)
    nfull = ds_fac * channel_taps
    noutlist = sam_per_read // ds_fac + (sam_per_read % ds_fac > 0)
    pfb_params["n_del"] = h_ds
    pfb_params["nout"] = noutlist
    pfb_params["coefs"] = pfb_coefs

    pfb_out, pfb_obj = setuppfbout(outdir, chan, signal_params, pfb_params, st)
    print("Decomposing {0} into {1} channels".format(path, pfbchans))
    print("Keeping channels {0}".format(mask))

    for ind, ist in enumerate(read_time):
        print("Reading {0} out of {1}".format(ind + 1, len(read_time)))
        nsamps = min(sam_per_read + M, et_samp - ist)
        x = drfObj.read_vector(ist, nsamps, chan, subchan)
        x_out = pfb_decompose(
            x, pfb_params["nchans"], pfb_params["coefs"], pfb_params["mask"]
        )

        for ipchan, iwriter in pfb_out.items():
            iwriter.rf_write(x_out[ipchan].astype(iwriter.dtype))
    for ipchan, iwriter in pfb_out.items():
        iwriter.close()


if __name__ == "__main__":
    args_commd = parse_command_line()
    arg_dict = {k: v for k, v in args_commd._get_kwargs() if v is not None}
    runpfb(**arg_dict)
