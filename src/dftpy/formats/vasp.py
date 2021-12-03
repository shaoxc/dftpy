import numpy as np
from dftpy.system import System
from dftpy.atom import Atom
from dftpy.base import BaseCell, DirectCell
from dftpy.constants import LEN_CONV

BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]


def read_POSCAR(infile, names=None, **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else :
        fh = open(infile, "r")

    title = fh.readline()
    scale = list(map(float, fh.readline().split()))
    if len(scale) == 1:
        scale = np.ones(3) * scale
    elif len(scale) == 3:
        scale = np.asarray(scale)
    lat = []
    for i in range(2, 5):
        lat.append(list(map(float, fh.readline().split())))
    lat = np.asarray(lat).T / BOHR2ANG
    lat = np.ascontiguousarray(lat)
    for i in range(3):
        lat[i] *= scale[i]
    lineL = fh.readline().split()
    if lineL[0].isdigit():
        typ = list(map(int, lineL))
    else:
        names = lineL
        typ = list(map(int, fh.readline().split()))
    if names is None:
        raise AttributeError("Must input the atoms names")
    Format = fh.readline().strip()[0]
    if Format.lower() == "d" :
        Format = "Crystal"
    elif Format.lower() in ['f', 'c'] :
        Format = "Cartesian"
    nat = sum(typ)
    pos = []
    i = 0
    for line in fh:
        i += 1
        if i > nat:
            break
        else:
            pos.append(list(map(float, line.split()[:3])))
    # pos = np.asarray(pos)
    if Format == "Cartesian" :
        pos = np.asarray(pos) / BOHR2ANG

    labels = []
    for i in range(len(names)):
        labels.extend([names[i]] * typ[i])

    cell = DirectCell(lat)
    atoms = Atom(label=labels, pos=pos, cell=cell, basis=Format)

    if not hasattr(infile, 'close'): fh.close()
    return atoms

def read_vasp(infile, **kwargs):
    atoms = read_POSCAR(infile, **kwargs)
    system = System(ions = atoms)
    return system

def write_vasp(infile, system, direct = False, fmt = '%22.15f', header = 'DFTpy', **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else :
        fh = open(infile, "w")

    ions = system.ions

    fh.write(header + '\n')
    fh.write('1.0\n')
    for i in range(3):
        vec = ions.pos.cell.lattice[:, i] * BOHR2ANG
        for x in vec : fh.write(fmt % x)
        fh.write('\n')

    if direct :
        pos = ions.pos.to_crys()
    else :
        pos = ions.pos.to_cart()

    names, indices, counts = np.unique(ions.labels, return_inverse=True, return_counts=True)

    for x in names : fh.write(f' {x:<3s}')
    fh.write('\n')
    for x in counts : fh.write(f' {x:<3d}')
    fh.write('\n')

    ind = np.argsort(indices)
    pos = pos[ind]

    if direct:
        fh.write('Direct\n')
    else:
        fh.write('Cartesian\n')
        pos = pos*BOHR2ANG

    for i, ps in enumerate(pos):
        fh.write(f'{fmt%ps[0]} {fmt%ps[1]} {fmt%ps[2]}\n')

    if not hasattr(infile, 'close'): fh.close()
    return
