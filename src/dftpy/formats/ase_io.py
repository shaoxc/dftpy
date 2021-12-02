import numpy as np
import ase
import ase.io
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.base import DirectCell
from dftpy.constants import LEN_CONV

BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]


def ase_read(infile, index=None, format=None, **kwargs):
    if isinstance(infile, ase.Atoms):
        struct = infile
    else :
        struct = ase.io.read(infile, index=index, format=format, **kwargs)
    atoms = ase2ions(struct)
    return atoms

def ase_write(outfile, ions, format=None, pbc=None, **kwargs):
    struct = ions2ase(ions)
    if pbc is not None :
        struct.set_pbc(pbc)
    ase.io.write(outfile, struct, format=format, **kwargs)
    return

def ase2ions(ase_atoms, wrap = True):
    lattice = ase_atoms.cell[:]
    lattice = np.asarray(lattice).T / BOHR2ANG
    lattice = np.ascontiguousarray(lattice)
    Z = ase_atoms.numbers
    cell = DirectCell(lattice)
    pos = ase_atoms.get_scaled_positions()
    if wrap :
        pos %= 1.0
    ions = Atom(Z=Z, pos=pos, cell=cell, basis="Crystal")
    return ions

def ions2ase(ions, pbc = True):
    cell = ions.pos.cell.lattice.T * BOHR2ANG
    cell = np.ascontiguousarray(cell)
    numbers = ions.Z
    pos = ions.pos[:] * BOHR2ANG
    struct = ase.Atoms(positions=pos, numbers=numbers, cell=cell, pbc = pbc)
    return struct

def read_ase(infile, **kwargs):
    ions = ase_read(infile, **kwargs)
    system = System(ions, name="DFTpy", field=None)
    return system

def write_ase(outfile, system, **kwargs):
    ase_write(outfile, system.ions, **kwargs)
