"""
API for pymatgen

Notes :
    From v2021.3.4, pymatgen removed the root imports.
"""
import numpy as np
from dftpy.atom import Atom
from dftpy.cell import DirectCell
from dftpy.constants import LEN_CONV
from dftpy.system import System
from pymatgen.core import Structure

BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]


def pmg_read(infile, index=None, format=None, **kwargs):
    struct = Structure.from_file(infile)
    ions = pmg2ions(struct, **kwargs)
    return ions


def pmg_write(outfile, ions, format=None, pbc=None, **kwargs):
    struct = ions2pmg(ions)
    struct.to(filename=outfile)
    return

def pmg2ions(pmg_atoms, wrap = True, **kwargs):
    lattice = pmg_atoms.lattice.matrix
    lattice = np.asarray(lattice).T / BOHR2ANG
    lattice = np.ascontiguousarray(lattice)
    labels = [item.symbol for item in pmg_atoms.species]
    cell = DirectCell(lattice)
    pos = pmg_atoms.frac_coords
    if wrap :
        pos %= 1.0
    ions = Atom(label=labels, pos=pos, cell=cell, basis="Crystal")
    return ions

def ions2pmg(ions):
    lattice = ions.pos.cell.lattice * BOHR2ANG
    pos = ions.pos.to_crys()
    labels = ions.labels
    struct = Structure(lattice, labels, pos)
    return struct

def read_pmg(infile, **kwargs):
    ions = pmg_read(infile, **kwargs)
    system = System(ions, name="DFTpy", field=None)
    return system

def write_pmg(outfile, system, **kwargs):
    pmg_write(outfile, system.ions, **kwargs)
