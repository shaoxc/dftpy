import os
import sys
import numpy as np
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.base import BaseCell, DirectCell
from dftpy.field import DirectField
from dftpy.constants import LEN_CONV
from dftpy.formats.vasp import read_POSCAR
from dftpy.formats.qepp import PP
from dftpy.formats.xsf import XSF

def read(infile, format=None, **kwargs):
    if format is None:
        format = guessType(infile)
    if format == "vasp":
        atom = read_POSCAR(infile, **kwargs)
    elif format == "qepp":
        qepp = PP(infile).read(**kwargs)
        atom = qepp.ions
    elif format == "qepp":
        xsf = XSF(infile).read(**kwargs)
        atom = xsf.ions
    else:
        raise AttributeError("%s format not support yet" % format)
    return atom


def guessType(infile, **kwargs):
    basename = os.path.basename(infile)
    ext = os.path.splitext(infile)[1]
    if basename == "POSCAR" or basename == "CONTCAR" or ext == ".vasp":
        format = "vasp"
    elif ext == ".pp":
        format = "qepp"
    elif ext == ".xsf":
        format = "xsf"
    elif ext == ".den":
        format = "den"
    else:
        raise AttributeError("%s not support yet" % infile)

    return format

def read_density(infile, format=None, **kwargs):
    if format is None:
        format = guessType(infile)

    if format == "qepp":
        qepp = PP(infile).read(**kwargs)
        density = qepp.field
    elif format == "xsf":
        xsf = XSF(infile).read(**kwargs)
        density = xsf.field
    elif format == "den":
        density = read_data_den(infile, **kwargs)
    else:
        raise AttributeError("%s format not support yet" % format)

    return density

def write(outfile, data, ions = None, format=None, **kwargs):
    if format is None:
        format = guessType(outfile)

    system = None
    if isinstance(data, System):
        system = data
    elif isinstance(data, DirectField) and ions is not None :
        system = System(ions, data.grid, name="DFTpy", field=data)
    elif isinstance(data, Atom):
        ions = data

    if format == "qepp":
        PP(outfile).write(system)
    elif format == "xsf":
        XSF(outfile).write(system)
    elif format == "den":
        if system is not None :
            data = system.field
        write_data_den(outfile, data, **kwargs)
    else:
        raise AttributeError("%s format not support yet" % format)

    return 


def read_data_den(infile, order="F", **kwargs):
    with open(infile, "r") as fr:
        line = fr.readline()
        nr0 = list(map(int, line.split()))
        blocksize = 1024 * 8
        strings = ""
        while True:
            line = fr.read(blocksize)
            if not line:
                break
            strings += line
    density = np.fromstring(strings, dtype=float, sep=" ")
    density = density.reshape(nr0, order=order)
    return density

def write_data_den(outfile, density, order = "F", **kwargs):
    with open(outfile, "w") as fw:
        nr = density.shape
        print('nr', nr)
        if len(nr) == 3 :
            fw.write("{0[0]:10d} {0[1]:10d} {0[2]:10d}\n".format(nr))
        elif len(nr) == 4 :
            fw.write("{0[0]:10d} {0[1]:10d} {0[2]:10d} {0[3]:10d}\n".format(nr))
        size = np.size(density)
        nl = size // 3
        outrho = density.ravel(order="F")
        for line in outrho[: nl * 3].reshape(-1, 3):
            fw.write("{0[0]:22.15E} {0[1]:22.15E} {0[2]:22.15E}\n".format(line))
        for line in outrho[nl * 3 :]:
            fw.write("{0:22.15E}".format(line))
