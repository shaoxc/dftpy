import os
import sys
import numpy as np
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.base import BaseCell, DirectCell
from dftpy.constants import LEN_CONV
from dftpy.formats.vasp import read_POSCAR
from dftpy.formats.qepp import PP


def read(infile, format=None, **kwargs):
    if format is None:
        format = guessType(infile)
    if format == "vasp":
        atom = read_POSCAR(infile, **kwargs)
    elif format == "qepp":
        qepp = PP(infile).read()
        atom = qepp.ions
    else:
        raise AttributeError("%s format not support yet" % format)
    return atom


def guessType(infile, **kwargs):
    basename = os.path.basename(infile)
    ext = os.path.splitext(infile)[1]
    if basename == "POSCAR" or basename == "CONTCAR":
        format = "vasp"
    elif ext == ".vasp":
        format = "vasp"
    elif ext == ".pp":
        format = "qepp"
    elif ext == ".xsf":
        format = "xsf"
    else:
        raise AttributeError("%s not support yet" % infile)

    return format

