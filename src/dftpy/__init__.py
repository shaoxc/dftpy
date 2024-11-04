__version__ = "2.1.2dev0"

from .config import *
from .mpi import mp, sprint
from .time_data import TimeData, timer

try:
    from importlib.metadata import version # python >= 3.8
    __version__ = version("dftpy")
except Exception :
    pass
