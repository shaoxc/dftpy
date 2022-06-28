import numpy as np
from dftpy.base import DirectCell
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.system import System
from dftpy.atom import Atom

def read_cube(infile, kind="all", full=False, pbc=True, data_type='density', **kwargs):
    # http://gaussian.com/cubegen/
    # http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/cubeplugin.html
    with open(infile, "r") as fh:
        fh.readline()
        fh.readline()
        line = fh.readline().split()
        natom = int(line[0])
        origin = np.asarray(list(map(float, line[1:])))

        nr = []
        lattice = []
        for i in range(3):
            line = fh.readline().split()
            nr.append(int(line[0]))
            lattice.append(list(map(float, line[1:])))
        nr = np.asarray(nr)
        lattice = np.asarray(lattice)
        lattice = np.ascontiguousarray(lattice.T)  # cell = [a, b, c]
        for i in range(3):
            lattice[:, i] *= nr[i]
        cell = DirectCell(lattice)

        Z = []
        pos = []
        for i in range(natom):
            line = fh.readline().split()
            Z.append(int(line[0]))
            pos.append(list(map(float, line[2:])))

        atoms = Atom(Z=Z, pos=pos, cell=cell, basis="Cartesian")

        if kind == "cell":
            system = System(ions = atoms)
        else :
            data = []
            for line in fh :
                line = line.split()
                l = list(map(float, line))
                data.extend(l)
            data = np.asarray(data)
            grid = DirectGrid(lattice=lattice, nr=nr, full=full, origin=origin)
            plot = DirectField(grid=grid, data=data, rank=1)
            system = System(atoms, grid, name="cube", field=plot)
    return system

def write_cube(filename, system, data_type = 'density', header = None, origin = None, long = True, **kwargs):
    if long :
        fmt = "{0:15}{1:22.15e}{2:22.15e}{3:22.15e}"
        fmt2 = fmt + "{4:22.15e}"
        fmt3 = "%22.15e"
    else :
        fmt = "{0:5}{1:12.6f}{2:12.6f}{3:12.6f}" # original
        fmt2 = fmt + "{4:12.6f}"
        fmt3 = "%13.5e"
    fh = open(filename, "w")
    if header is None :
        header = 'DFTpy : cube file for {}'.format(data_type)
    header += "\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n"
    fh.write(header)

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin)

    fh.write(fmt.format(system.ions.nat, *origin) + '\n')
    data = system.field
    ions = system.ions
    lattice = ions.pos.cell.lattice

    shape = data.shape
    for i in range(3):
        v = lattice[:, i] / shape[i]
        fh.write(fmt.format(shape[i], *v) + '\n')

    for i in range(ions.nat):
        z = ions.Z[i]
        c = ions.Zval.get(ions.labels[i], 0.0)
        p = ions.pos[i]
        fh.write(fmt2.format(z, c, *p) + '\n')

    # data.tofile(fh, sep="\n", format="%e")
    val_per_line = 6
    data = data.ravel()
    nnr = data.size
    nlines = nnr // val_per_line
    for iline in range(nlines):
        i = iline * val_per_line
        data[i : i + val_per_line].tofile(fh, sep=" ", format=fmt3)
        fh.write('\n')
    i = nlines * val_per_line
    if i < nnr : data[i : nnr].tofile(fh, sep=" ", format=fmt3)
    fh.close()
