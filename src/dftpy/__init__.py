__author__ = "Pavanello Research Group"
__contact__ = "m.pavanello@rutgers.edu"
__license__ = "MIT"
__version__ = "1.1.0"
__date__ = "2021-09-08"

from .config import *
from .mpi import mp

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dftpy")
except PackageNotFoundError:
    pass
