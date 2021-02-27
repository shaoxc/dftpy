"""
IO of snpy file

SNPY format
===========

snpy format is just contains some numpy NPY files, but has a definite order which contains structure information.

 - lattice matrix
 - symbols of atoms
 - positions of atoms
 - volumetric data
 - other data

snpy format also can contains multiframe, each frame is separate by a string matrix only contain one item, which I prefer
start with 'DFTPY'

Notes :
    snpy format also can be directly replace with numpy npz format, but it's need parallel compress and decompress
"""
import numpy as np
from dftpy.base import DirectCell
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.formats import npy
from dftpy.mpi import MP, MPIFile

MAGIC_PREFIX = b'\x93DFTPY'

def write(fname, system, mp = None):
    ions = system.ions
    data = system.field
    if mp is None :
        mp = data.grid.mp
    if isinstance(fname, str):
        if mp.size > 1 :
            # fh = mp.MPI.File.Open(mp.comm, fname, amode = mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
            fh = MPIFile(fname, mp, amode = mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
        else :
            fh = open(fname, "wb")
    else :
        fh = fname

    if mp.rank == 0 :
        # write cell
        npy.write(fh, ions.pos.cell.lattice, single = True)
        # write labels
        npy.write(fh, ions.Z, single = True)
        # write coordinates
        npy.write(fh, ions.pos, single = True)
    # write volumetric data
    npy.write(fh, data)
    return

def read(fname, mp=None, grid=None, kind="all", full=False, datarep='native', **kwargs):
    """
    Notes :
        Only support DirectField
    """
    if mp is None :
        if grid is None :
            mp = MP()
        else :
            mp = grid.mp
    if isinstance(fname, str):
        if mp.size > 1 :
            # fh = mp.MPI.File.Open(mp.comm, fname, amode = mp.MPI.MODE_RDONLY)
            fh = MPIFile(fname, mp, amode = mp.MPI.MODE_RDONLY)
        else :
            fh = open(fname, "rb")
    else :
        fh = fname

    # read cell
    lattice = npy.read(fh, single = True)
    # read labels
    labels = npy.read(fh, single = True)
    # read coordinates
    pos = npy.read(fh, single = True)
    cell = DirectCell(lattice)
    atoms = Atom(label=labels, pos=pos, cell=cell, basis="Cartesian")
    if kind == 'cell' :
        return atoms
    # read volumetric data
    if mp.size == 1 :
        data = npy.read(fh, single=True)
        if grid is None :
            grid = DirectGrid(lattice=lattice, nr=data.shape, full=full, mp=mp)
        data = DirectField(grid=grid, griddata_3d=data, rank=1)
    else :
        shape, fortran_order, dtype = npy._read_header(fh)
        if fortran_order :
            raise AttributeError("Not support Fortran order")
        if grid is None :
            grid = DirectGrid(lattice=lattice, nr=shape, full=full, mp=mp)
            data = DirectField(grid=grid, rank=1)
        elif not(np.all(shape == grid.nrR) or np.all(shape == grid.nrG)):
            raise AttributeError("The shape is not match with grid")
        npy._read_value(fh, data, datarep=datarep)
    return System(atoms, grid, name="snpy", field=data)
