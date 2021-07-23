from .estimators import create_sti
from .detectors import cfar
from .pfb import pfb_decompose, pfb_reconstruct, pfbresponse

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
