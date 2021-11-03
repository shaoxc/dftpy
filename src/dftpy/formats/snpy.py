"""
IO of snpy file

SNPY format
===========
snpy format is just contains some numpy NPY files, but has a definite order which contains structure information.
 - description of snpy (1d integer array)
   1. lattice matrix  (3x3 float array)
   2. symbols of atoms (1d integer array)
   3. positions of atoms (Nx3 float array)
   4. volumetric data (3d float array)
   5. other data
 - repeat

The first item of snpy is the description of all items of this frame. snpy format can contains multiframe, and each
frame is start with the description.

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
from dftpy.mpi import MP, MPIFile, sprint

MAGIC_PREFIX = b'\x93DFTPY'

def write(fname, system, kind = 'all', desc = None, mp = None, **kwargs):
    ions = system.ions
    data = system.field
    if mp is None :
        mp = data.grid.mp
    if hasattr(fname, 'close'):
        fh = fname
    else :
        if mp.size > 1 :
            fh = MPIFile(fname, mp, amode = mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
        else :
            fh = open(fname, "wb")

    if desc is None :
        if kind == 'cell' :
            desc = np.arange(1, 4)
        else :
            desc = np.arange(1, 5)

    if mp.rank == 0 : # write description
        npy.write(fh, desc, single = True)

    for key in desc :
        if key == 1 : # write cell
            if mp.rank == 0 : npy.write(fh, ions.pos.cell.lattice, single = True)
        elif key == 2 : # write labels
            if mp.rank == 0 : npy.write(fh, ions.Z, single = True)
        elif key == 3 : # write coordinates
            if mp.rank == 0 : npy.write(fh, ions.pos, single = True)
        elif key == 4 : # write volumetric data
            npy.write(fh, data)

    if isinstance(fname, str): fh.close()
    return

def read(fname, mp=None, grid=None, kind="all", full=False, datarep='native', desc=None, **kwargs):
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

    atoms = None
    if desc is None :
        # read description
        desc = npy.read(fh, single = True)
    else :
        sprint('WARN : You set the description by yourself {}'.format(desc), comm = mp.comm, level = 3)

    for key in desc :
        if key == 1 : # read cell
            lattice = npy.read(fh, single = True)
        elif key == 2 : # read labels
            labels = npy.read(fh, single = True)
        elif key == 3 : # read coordinates
            pos = npy.read(fh, single = True)
            cell = DirectCell(lattice)
            atoms = Atom(label=labels, pos=pos, cell=cell, basis="Cartesian")
            if kind == 'cell' :
                if isinstance(fname, str): fh.close()
                return atoms
        elif key == 4 : # read volumetric data
            if mp.size == 1 :
                data = npy.read(fh, single=True)
                if grid is None :
                    grid = DirectGrid(lattice=lattice, nr=data.shape, full=full, mp=mp)
                data = DirectField(grid=grid, griddata_3d=data, rank=1)
            else :
                shape, fortran_order, dtype = npy._read_header(fh)
                # if fortran_order :
                #     raise AttributeError("Not support Fortran order")
                if grid is None :
                    grid = DirectGrid(lattice=lattice, nr=shape, full=full, mp=mp)
                elif not(np.all(shape == grid.nrR) or np.all(shape == grid.nrG)):
                    raise AttributeError("The shape is not match with grid")
                order = 'F' if fortran_order else 'C'
                data = DirectField(grid=grid, rank=1, order = order)
                npy._read_value(fh, data, datarep=datarep, fortran_order=fortran_order)

    if isinstance(fname, str): fh.close()
    return System(atoms, grid, name="DFTpy", field=data)
