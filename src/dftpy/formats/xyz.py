import numpy as np
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.cell import BaseCell, DirectCell
from dftpy.constants import LEN_CONV

BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
"""
Ref :
    http://atomsk.univ-lille1.fr/doc/en/format_xyz.html
"""


def read_xyz(infile, **kwargs):
    pass
