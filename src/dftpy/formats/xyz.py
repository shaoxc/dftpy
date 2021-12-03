import numpy as np
import re
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.base import DirectCell
from dftpy.constants import LEN_CONV

BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
"""
Ref :
    http://atomsk.univ-lille1.fr/doc/en/format_xyz.html
"""


def read_xyz(infile, **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else :
        fh = open(infile, "r")

    line = fh.readline()
    natom = int(line)
    line = fh.readline().strip()
    p = re.compile(r'(Lattice)'+r'\s*=\s*["\{]([^"\{\}]+)["\}]\s*')
    m = p.match(line)
    if m is None :
        lattice = np.zeros((3, 3))
    else :
        lattice = np.fromstring(m.group(2), dtype=float, sep=" ")
        lattice = np.asarray(lattice).reshape((3, 3)) / BOHR2ANG
    labels = []
    pos = []
    for i in range(natom):
        line = fh.readline().split()
        labels.append(line[0])
        pos.append(list(map(float, line[1:4])))
    pos = np.asarray(pos) / BOHR2ANG

    cell = DirectCell(lattice)
    atoms = Atom(label=labels, pos=pos, cell=cell, basis='Cartesian')

    if not hasattr(infile, 'close'): fh.close()
    system = System(ions = atoms)
    return system

def write_xyz(infile, system = None, comment = None, fmt = '%22.15f', **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else :
        fh = open(infile, "w")

    ions = system.ions

    if comment is None :
        comment = 'Lattice="' + " ".join(map(str, (ions.pos.cell.lattice*BOHR2ANG).ravel(order = 'F').tolist())) + '"'
        comment += ' Properties=species:S:1:pos:R:3'

    comment = comment.rstrip()
    natoms = len(ions.pos)
    fh.write(f'{natoms}\n{comment}\n')
    pos = ions.pos*BOHR2ANG
    for s, ps in zip(ions.labels, pos):
        fh.write(f'{s:3s} {fmt%ps[0]} {fmt%ps[1]} {fmt%ps[2]}\n')

    if not hasattr(infile, 'close'): fh.close()
    return
