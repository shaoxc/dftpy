import numpy as np
from dftpy.constants import LEN_CONV
from dftpy.base import BaseCell, DirectCell
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.system import System
from dftpy.atom import Atom


def read_xsf(infile, kind="All", full=False, pbc=True, **kwargs):
    # http ://www.xcrysden.org/doc/XSF.html
    with open(infile, "r") as fr:

        def readline():
            for line in fr:
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                else:
                    return line

        celltype = readline()
        lattice = []
        line = readline()
        if line.startswith("PRIMVEC"):
            for i in range(3):
                l = list(map(float, fr.readline().split()))
                lattice.append(l)
            lattice = np.asarray(lattice) / LEN_CONV["Bohr"]["Angstrom"]
            lattice = lattice.T  # cell = [a, b, c]
            line = readline()

        label = []
        pos = []
        if line.startswith("CONVVEC"):
            for i in range(4):
                line = readline()

        if line.startswith("PRIMCOORD"):
            natom = int(fr.readline().split()[0])
            for i in range(natom):
                line = readline().split()
                label.append(line[0])
                p = list(map(float, line[1:4]))
                pos.append(p)
            pos = np.asarray(pos) / LEN_CONV["Bohr"]["Angstrom"]
            line = readline()

        if len(lattice) > 0 :
            cell = DirectCell(lattice)
            atoms = Atom(label=label, pos=pos, cell=cell, basis="Cartesian")
            if kind == "cell":
                return atoms

        if line.startswith("BEGIN_BLOCK_DATAGRID_3D"):
            line = readline()
            line = readline()
        data = []
        if line.startswith("BEGIN_DATAGRID_3D"):
            nrx = np.empty(3, dtype=int)
            nrx[0], nrx[1], nrx[2] = map(int, readline().split())
            readline()  # read one useless line
            if len(lattice) == 0 :
                for i in range(3):
                    l = list(map(float, readline().split()))
                    lattice.append(l)
                lattice = np.asarray(lattice)
                lattice = lattice.T  # cell = [a, b, c]
                cell = DirectCell(lattice)
                atoms = Atom(label=label, pos=pos, cell=cell, basis="Cartesian")
                if kind == "cell":
                    return atoms
            else:
                for i in range(3):
                    line = readline()
            # for speed, we assume in the data block no blank line
            for line in fr:
                line = line.split()
                if not line:
                    continue
                if line[0][0] == "E":
                    break
                else:
                    l = list(map(float, line))
                    data.extend(l)

        if not data:
            raise AttributeError("!!!ERROR : XSF file have some problem")
        data = np.asarray(data)
        if np.size(data) > np.prod(nrx):  # double xsf grid data
            data = data[: np.prod(nrx)]
        data = np.reshape(data, nrx, order="F")
        if pbc:
            bound = nrx.copy()
            for i in range(len(nrx)):
                if nrx[i] > 1:
                    bound[i] = nrx[i] - 1
                else:
                    bound[i] = nrx[i]
            data = data[: bound[0], : bound[1], : bound[2]]
            nrx = bound.copy()

        grid = DirectGrid(lattice=lattice, nr=nrx, units=None, full=full)
        plot = DirectField(grid=grid, griddata_3d=data, rank=1)
        # plot = DirectField(grid=grid, griddata_F=data, rank=1)
        return System(atoms, grid, name="xsf", field=plot)


class XSF(object):

    xsf_units = "Angstrom"

    def __init__(self, filexsf):
        self.filexsf = filexsf
        self.title = ""
        self.cutoffvars = {}

    def write(self, system, field=None):
        """
        Write a system object into an xsf file.
        Not all specifications of the xsf file format are implemented, they will
        be added as needed.
        So far it can:
            - write the system cell and atoms
            - write the 1D/2D/3D grid data
        """

        title = system.name
        cell = system.cell
        ions = system.ions

        # it can be useful to override the plot inside the system object,
        # for example if we want to plot a 2D/3D custom cut of the density grid
        if field is None:
            field = system.field

        with open(self.filexsf, "w") as fileout:
            self._write_header(fileout, title)
            self._write_cell(fileout, cell)
            self._write_coord(fileout, ions)
            self._write_datagrid(fileout, field)

        return 0

    def read(self, kind="All", full=False, **kwargs):
        return read_xsf(self.filexsf, kind=kind, full=full, **kwargs)

    def _write_header(self, fileout, title):
        mywrite(fileout, ("# ", title))
        mywrite(fileout, "CRYSTAL \n", True)

    def _write_cell(self, fileout, cell):
        mywrite(fileout, "PRIMVEC", True)
        for ilat in range(3):
            latt = cell.lattice[:, ilat] * LEN_CONV["Bohr"][self.xsf_units]
            mywrite(fileout, latt, True)

    def _write_coord(self, fileout, ions):
        mywrite(fileout, "PRIMCOORD", True)
        mywrite(fileout, (len(ions.pos), 1), True)
        for i in range(len(ions.pos)):
            mywrite(fileout, (ions.labels[i], ions.pos[i] * LEN_CONV["Bohr"][self.xsf_units]), True)
        # for iat, atom in enumerate(ions):
        # mywrite(fileout, (atom.label, atom.pos*LEN_CONV["Bohr"][self.xsf_units]), True)

    def _write_datagrid(self, fileout, plot):
        ndim = plot.span  # 2D or 3D grid?
        if ndim < 2:
            return  # XSF format doesn't support one data grids
        val_per_line = 5
        values = plot.get_values_flatarray(pad=1, order="F")

        mywrite(fileout, "BEGIN_BLOCK_DATAGRID_{}D".format(ndim), True)
        mywrite(fileout, "{}d_datagrid_from_pbcpy".format(ndim), True)
        mywrite(fileout, "BEGIN_DATAGRID_{}D".format(ndim), True)
        nnr = len(values)
        origin = plot.grid.origin * LEN_CONV["Bohr"][self.xsf_units]
        if ndim == 3:
            mywrite(fileout, (plot.grid.nr[0] + 1, plot.grid.nr[1] + 1, plot.grid.nr[2] + 1), True)
        elif ndim == 2:
            mywrite(fileout, (plot.grid.nr[0] + 1, plot.grid.nr[1] + 1), True)
        mywrite(
            fileout, origin, True
        )  # TODO, there might be an actual origin if we're dealing with a custom cut of the grid
        for ilat in range(ndim):
            latt = plot.grid.lattice[:, ilat] * LEN_CONV["Bohr"][self.xsf_units]
            mywrite(fileout, latt, True)

        nlines = nnr // val_per_line

        for iline in range(nlines):
            igrid = iline * val_per_line
            mywrite(fileout, values[igrid : igrid + val_per_line], True)
        igrid = nlines * val_per_line
        mywrite(fileout, values[igrid:nnr], True)

        mywrite(fileout, "END_DATAGRID_{}D".format(ndim), True)
        mywrite(fileout, "END_BLOCK_DATAGRID_{}D".format(ndim), True)


def mywrite(fileobj, iterable, newline=False):
    if newline:
        fileobj.write("\n  ")
    if isinstance(iterable, (np.ndarray, list, tuple)):
        for ele in iterable:
            mywrite(fileobj, ele)
            # fileobj.write(str(ele)+'    ')
    else:
        fileobj.write(str(iterable) + "    ")
