import os
import sys
import numpy as np
from ..system import System
from ..atom import Atom
from ..base import BaseCell, DirectCell
import pymatgen as pmg
from dftpy.constants import LEN_CONV

BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]


def pmg_read(infile, index=None, format=None, **kwargs):
    struct = pmg.Structure.from_file(infile)
    lattice = struct.lattice.matrix
    lattice = np.asarray(lattice).T / BOHR2ANG
    labels = [item.symbol for item in struct.species]
    cell = DirectCell(lattice)
    pos = struct.cart_coords / BOHR2ANG
    atoms = Atom(label=labels, pos=pos, cell=cell, basis="Cartesian")
    return atoms


def pmg_write(outfile, ions, format=None, pbc=None, **kwargs):
    lattice = ions.pos.cell.lattice
    pos = ions.pos.to_crys()
    labels = ions.labels
    struct = pmg.Structure(lattice, labels, pos)
    struct.to(filename=outfile)
    return
