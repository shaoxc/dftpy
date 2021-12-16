import numpy as np
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.base import DirectCell
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.system import System
from dftpy.atom import Atom

def xsf_readline(fr):
    for line in fr:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        else :
            yield line

def read_xsf(infile, kind="all", full=False, pbc=True, units='Angstrom', data_type='density', **kwargs):
    # http ://www.xcrysden.org/doc/XSF.html
    if isinstance(units, str):
        xsf_units = [units, units]
    elif isinstance(units, list):
        xsf_units = units
    else :
        raise AttributeError("!!!ERROR : Wrong type of the `units`")

    with open(infile, "r") as fr:
        fh = xsf_readline(fr)
        celltype = next(fh).upper()
        if celltype != 'CRYSTAL' :
            raise AttributeError("Only support crystal xsf.")
        lattice = []
        line = next(fh).upper()
        if line.startswith("PRIMVEC"):
            for i in range(3):
                l = list(map(float, fr.readline().split()))
                lattice.append(l)
            lattice = np.asarray(lattice) / LEN_CONV["Bohr"][xsf_units[0]]
            lattice = np.ascontiguousarray(lattice.T)  # cell = [a, b, c]
            line = next(fh).upper()

        label = []
        pos = []
        if line.startswith("CONVVEC"):
            for i in range(3):
                line = next(fh)
            line = next(fh).upper()

        if line.startswith("PRIMCOORD"):
            natom = int(fr.readline().split()[0])
            for i in range(natom):
                line = next(fh).split()
                label.append(line[0])
                p = list(map(float, line[1:4]))
                pos.append(p)
            pos = np.asarray(pos) / LEN_CONV["Bohr"][xsf_units[0]]
            line = next(fh).upper()

        cell = DirectCell(lattice)
        atoms = Atom(label=label, pos=pos, cell=cell, basis="Cartesian")

        if kind == "cell":
            system = System(ions = atoms)
        else :
            if line.startswith("BEGIN_BLOCK_DATAGRID_"):
                line = next(fh)
            blocks = []
            for line in fh :
                data = []
                line = line.upper()
                if line.startswith("BEGIN_DATAGRID"):
                    nrx = np.ones(3, dtype=int)
                    npbc = 3
                    if line.startswith("BEGIN_DATAGRID_3D"):
                        nrx[0], nrx[1], nrx[2] = map(int, next(fh).split())
                        npbc = 3
                    elif line.startswith("BEGIN_DATAGRID_2D"):
                        nrx[0], nrx[1] = map(int, next(fh).split())
                        npbc = 2
                    elif line.startswith("BEGIN_DATAGRID_1D"):
                        nrx[0] = map(int, next(fh).split())
                        npbc = 1
                    next(fh)  # read the origin
                    vlat = np.zeros((3, 3))
                    for i in range(npbc):
                        l = list(map(float, next(fh).split()))
                        vlat[i] = np.asarray(l)
                    if npbc == 1:
                        for i in range(3):
                            if abs(vlat[0][i]) > 1e-4:
                                j = i - 1
                                vlat[1][j] = vlat[0][i]
                                vlat[1][i] = -vlat[0][j]
                                vlat[1] = vlat[1] / np.sqrt(np.dot(vlat[1], vlat[1]))
                                break
                        vlat[2] = np.cross(vlat[0], vlat[1])
                        vlat[2] = vlat[2] / np.sqrt(np.dot(vlat[2], vlat[2]))
                    elif npbc == 2:
                        vlat[2] = np.cross(vlat[0], vlat[1])
                        vlat[2] = vlat[2] / np.sqrt(np.dot(vlat[2], vlat[2]))
                    data_lat = np.ascontiguousarray(vlat.T)  # cell = [a, b, c]
                    data_lat /= LEN_CONV["Bohr"][xsf_units[0]]
                    for line in fh:
                        if line[0] == "E":
                            break
                        else:
                            line = line.split()
                            l = list(map(float, line))
                            data.extend(l)
                    data = np.asarray(data)
                    if np.size(data) > np.prod(nrx):  # double xsf grid data
                        data = data[: np.prod(nrx)]
                    data = np.reshape(data, nrx, order="F")
                    blocks.append(data)
                if line.strip().startswith('END_BLOCK_DATAGRID'): break

            if not blocks:
                raise AttributeError("!!!ERROR : XSF file have some problem")

            nrx_prev = None
            rank = len(blocks)
            for spin in range(rank):
                data = blocks[spin]
                nrx = np.array(data.shape)
                if nrx_prev is not None :
                    if not np.all(nrx_prev == nrx):
                        raise AttributeError("All DATAGRID should have same shape.")
                else :
                    nrx_prev = nrx.copy()
                if pbc:
                    for i in range(len(nrx)):
                        if nrx[i] > 1: nrx[i] -= 1
                    data = data[: nrx[0], : nrx[1], : nrx[2]]
                data *= LEN_CONV["Bohr"][xsf_units[1]] ** 3
                blocks[spin] = data

            grid = DirectGrid(lattice=data_lat, nr=nrx, units=None, full=full)
            plot = DirectField(grid=grid, griddata_3d=blocks, rank=rank)
            if data_type == 'potential' :
                plot *= ENERGY_CONV["eV"]["Hartree"]
            system = System(atoms, grid, name="xsf", field=plot)
    return system

def write_xsf(filexsf, system, field = None, **kwargs):
    return XSF(filexsf).write(system, field, **kwargs)


class XSF(object):

    xsf_units = "Angstrom"

    def __init__(self, filexsf, title = 'DFTpy'):
        self.filexsf = filexsf
        self.title = title
        self.cutoffvars = {}

    def write(self, system, field=None, data_type = 'density', **kwargs):
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
            self._write_datagrid(fileout, field, data_type = data_type)

        return

    def read(self, kind="all", full=False, **kwargs):
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

    def _write_datagrid(self, fileout, plot, data_type = 'density', **kwargs):
        ndim = plot.span  # 2D or 3D grid?
        if ndim < 2:
            return  # XSF format doesn't support one data grids
        val_per_line = 5
        rank = plot.rank
        grid = plot.grid
        if rank == 1 :
            plot = [plot]
        data = []
        for p in plot :
            values = p.get_values_flatarray(pad=1, order="F") / LEN_CONV["Bohr"][self.xsf_units] ** 3
            data.append(values)
        if data_type == 'potential' :
            values = values * ENERGY_CONV["Hartree"]["eV"]

        mywrite(fileout, "BEGIN_BLOCK_DATAGRID_{}D".format(ndim), True)
        mywrite(fileout, "{}d_datagrid_{}".format(ndim, data_type), True)
        for i, values in enumerate(data) :
            mywrite(fileout, "BEGIN_DATAGRID_{}D#{}".format(ndim, i), True)
            origin = grid.origin * LEN_CONV["Bohr"][self.xsf_units]
            if ndim == 3:
                mywrite(fileout, (grid.nr[0] + 1, grid.nr[1] + 1, grid.nr[2] + 1), True)
            elif ndim == 2:
                mywrite(fileout, (grid.nr[0] + 1, grid.nr[1] + 1), True)
            mywrite(
                fileout, origin, True
            )  # TODO, there might be an actual origin if we're dealing with a custom cut of the grid
            for ilat in range(ndim):
                latt = grid.lattice[:, ilat] * LEN_CONV["Bohr"][self.xsf_units]
                mywrite(fileout, latt, True)

            nnr = len(values)
            nlines = nnr // val_per_line
            for iline in range(nlines):
                igrid = iline * val_per_line
                mywrite(fileout, values[igrid : igrid + val_per_line], True)
            igrid = nlines * val_per_line
            if igrid < nnr : mywrite(fileout, values[igrid:nnr], True)
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
