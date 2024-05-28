__author__ = "Pavanello Research Group"
__contact__ = "m.pavanello@rutgers.edu"
__license__ = "MIT"
__version__ = "2.1.0"
__date__ = "2023-12-14"

from .config import *
from .mpi import mp, sprint
from .time_data import TimeData, timer

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
