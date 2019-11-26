import os
import sys
import numpy as np
import ase
import ase.io
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.base import BaseCell, DirectCell
from dftpy.constants import LEN_CONV

BOHR2ANG   = LEN_CONV['Bohr']['Angstrom']

def ase_read(infile, index=None, format=None, **kwargs):
    struct = ase.io.read(infile, index=index, format=format, **kwargs)
    lattice = struct.cell[:]
    lattice = np.asarray(lattice).T/BOHR2ANG
    Z = struct.numbers
    cell = DirectCell(lattice)
    pos = struct.get_positions()/BOHR2ANG
    atoms = Atom(Z = Z, pos=pos, cell=cell, basis = 'Cartesian')
    return atoms

def ase_write(outfile, ions, format = None, pbc = None,  **kwargs):
    cell = ions.pos.cell.lattice * BOHR2ANG
    numbers = ions.Z
    pos = ions.pos[:] * BOHR2ANG
    if pbc is None :
        pbc = [1, 1, 1]
    struct = ase.Atoms(positions = pos, numbers = numbers, cell = cell, pbc = pbc)
    ase.io.write(outfile, struct, format = format, **kwargs)
    return
