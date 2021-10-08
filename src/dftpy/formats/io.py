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
from dftpy.formats import snpy

def guessType(infile, **kwargs):
    basename = os.path.basename(infile)
    ext = os.path.splitext(infile)[1].lower()
    if basename == "POSCAR" or basename == "CONTCAR" or ext == ".vasp":
        format = "vasp"
    elif ext == ".pp":
        format = "qepp"
    elif ext == ".xsf":
        format = "xsf"
    elif ext == ".den":
        format = "den"
    elif ext == ".snpy":
        format = "snpy"
    else:
        raise AttributeError("%s not support yet" % infile)

    return format

def read(infile, format=None, **kwargs):
    kind = kwargs.get('kind', 'cell')
    struct = read_system(infile, format=format, **kwargs)
    if kind == 'cell' :
        return struct.ions
    elif kind == 'field' :
        return struct.field
    else :
        return struct

def read_system(infile, format=None, **kwargs):
    if format is None:
        format = guessType(infile)

    if format == "snpy":
        struct = snpy.read(infile, **kwargs)
    elif format == "vasp":
        struct = read_POSCAR(infile, **kwargs)
        kwargs['kind'] = 'cell'
    elif format == "qepp":
        struct = PP(infile).read(**kwargs)
    elif format == "xsf":
        struct = XSF(infile).read(**kwargs)
    elif format == "den":
        density = read_data_den(infile, **kwargs)
        struct= System(None, field=density)
    else:
        raise AttributeError("%s format not support yet" % format)
    kind = kwargs.get('kind', 'all')
    if kind == 'cell' :
        struct= System(ions = struct)
    return struct

def read_density(infile, format=None, **kwargs):
    struct = read_system(infile, format=format, **kwargs)
    return struct.field

def write(outfile, data = None, ions = None, format=None, **kwargs):
    if format is None:
        format = guessType(outfile)

    if isinstance(data, Atom):
        from dftpy.formats import ase_io
        if ions is None :
            return ase_io.ase_write(outfile, ions, **kwargs)
        elif isinstance(ions, DirectField):
            ions, data = data, ions
            system = System(ions, name="DFTpy", field=data)
        else :
            raise AttributeError("Please check the input data")
    elif isinstance(data, System):
        system = data
    elif isinstance(data, DirectField):
        system = System(ions, name="DFTpy", field=data)
    else :
        raise AttributeError("Please check the input data")

    if format == "snpy":
        # Parallel IO
        return snpy.write(outfile, system, **kwargs)

    mp = system.field.mp

    if mp.is_mpi :
        total = system.field.gather()
        system = System(system.ions, name="DFTpy", field=total)

    if mp.is_root :
        if format == "qepp":
            PP(outfile).write(system, **kwargs)
        elif format == "xsf":
            XSF(outfile).write(system, **kwargs)
        elif format == "den":
            write_data_den(outfile, system.field, **kwargs)
        else:
            raise AttributeError("%s format not support yet" % format)
    mp.comm.Barrier()
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
