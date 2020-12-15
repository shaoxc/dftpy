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

MAGIC_PREFIX = b'\x93DFTPY'

def write(fname, system, mp = None, units = 'Bohr'):
    ions = system.ions
    data = system.field
    if mp is None :
        mp = data.grid.mp
    if isinstance(fname, str):
        if mp.size > 1 :
            fh = mp.MPI.File.Open(mp.comm, fname, amode = mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
        else :
            fh = open(fname, "wb")
    else :
        fh = fname
    if mp.rank == 0 :
        # write cell
        npy.write(fh, ions.pos.cell.lattice, single = True, close = False)
        # write labels
        # npy.write(fh, ions.labels, single = True, close = False)
        npy.write(fh, ions.Z, single = True, close = False)
        # write coordinates
        npy.write(fh, ions.pos, single = True, close = False)
    # write volumetric data
    npy.write(fh, data)
    if hasattr(fh, 'close'):
        fh.close()
    elif hasattr(fh, 'Close'):
        fh.Close()
    return

def read(fname, mp=None, grid=None, kind="all", full=False, units='Bohr', datarep='native', **kwargs):
    """
    Notes :
        Only support DirectField
    """
    if mp is None :
        from dftpy.mpi import mp
    if isinstance(fname, str):
        if mp.size > 1 :
            fh = mp.MPI.File.Open(mp.comm, fname, amode = mp.MPI.MODE_RDONLY)
        else :
            fh = open(fname, "rb")
    else :
        fh = fname
    # read cell
    lattice = npy.read(fh, single = True, close = False)
    # read labels
    labels = npy.read(fh, single = True, close = False)
    # read coordinates
    pos = npy.read(fh, single = True, close = False)
    cell = DirectCell(lattice)
    atoms = Atom(label=labels, pos=pos, cell=cell, basis="Cartesian")
    if kind == 'cell' :
        if hasattr(fh, 'close'):
            fh.close()
        elif hasattr(fh, 'Close'):
            fh.Close()
        return atoms
    # read volumetric data
    if mp.size == 1 :
        data = npy.read(fh, single=True, close=False)
        if grid is None :
            grid = DirectGrid(lattice=lattice, nr=data.shape, units='Bohr', full=full, mp=mp)
        data = DirectField(grid=grid, griddata_3d=data, rank=1)
        fh.close()
    else :
        shape, fortran_order, dtype = npy._read_header(fh)
        if fortran_order :
            raise AttributeError("Not support Fortran order")
        if grid is None :
            grid = DirectGrid(lattice=lattice, nr=shape, units='Bohr', full=full, mp=mp)
            data = DirectField(grid=grid, rank=1)
        elif not(np.all(shape == grid.nrR) or np.all(shape == grid.nrG)):
            raise AttributeError("The shape is not match with grid")
        npy._read_value(fh, data, datarep=datarep)
        fh.Close()
    return System(atoms, grid, name="snpy", field=data)
