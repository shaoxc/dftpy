__author__ = "Pavanello Research Group"
__contact__ = "m.pavanello@rutgers.edu"
__license__ = "MIT"
__version__ = "1.1.0"
__date__ = "2021-09-08"

from .config import *
from .mpi import mp, sprint

try:
    from importlib.metadata import version # python >= 3.8
except Exception :
    try:
        from importlib_metadata import version
    except Exception :
        pass

try:
    __version__ = version("dftpy")
except Exception :
    pass
