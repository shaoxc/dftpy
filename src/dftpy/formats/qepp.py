import numpy as np
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.system import System
from dftpy.atom import Atom


class PP(object):
    def __init__(self, filepp):
        self.filepp = filepp
        self.title = ""
        self.cutoffvars = {}

    def read(self, full=True, data_type='density', **kwargs):

        with open(self.filepp) as filepp:
            # title
            self.title = filepp.readline()

            # nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp
            nrx = np.empty(3, dtype=int)
            nr = np.empty(3, dtype=int)
            nrx[0], nrx[1], nrx[2], nr[0], nr[1], nr[2], nat, ntyp = (int(x) for x in filepp.readline().split())

            # ibrav, celldm
            celldm = np.zeros(6, dtype=float)
            linesplt = filepp.readline().split()
            ibrav = int(linesplt[0])
            celldm = np.asarray(linesplt[1:], dtype=float)

            # at(:,i) three times
            if ibrav == 0:
                at = np.zeros((3, 3), dtype=float)
                for ilat in range(3):
                    linesplt = filepp.readline().split()
                    at[:, ilat] = np.asarray(linesplt, dtype=float)
                    # at[:,i] = (float(x) for x in filepp.readline().split())
                at *= celldm[0]
            else:
                at = self.celldm2at(ibrav, celldm)
            grid = DirectGrid(lattice=at, nr=nrx, units=None, full=full)

            # gcutm, dual, ecut, plot_num
            # gcutm, dual, ecut, plot_num = (float(x) for x in filepp.readline().split())
            # plot_num = int(plot_num)
            filepp.readline()
            gcutm, dual, ecut, plot_num = 1.0, 1.0, 1.0, 0
            self.cutoffvars["ibrav"] = ibrav
            self.cutoffvars["celldm"] = celldm
            self.cutoffvars["gcutm"] = gcutm
            self.cutoffvars["dual"] = dual
            self.cutoffvars["ecut"] = ecut

            # ntyp
            atm = []
            zv = np.empty(ntyp, dtype=float)
            Zval = {}
            for ity in range(ntyp):
                linesplt = filepp.readline().split()
                atm.append(linesplt[1])
                zv[ity] = float(linesplt[2])
                Zval[linesplt[1]] = float(linesplt[2])
            # tau
            # tau = np.zeros((nat,3), dtype=float)
            # tau = np.zeros(3, dtype=float)
            # ityp = np.zeros(nat, dtype=int)
            label = []
            pos = []
            for iat in range(nat):
                linesplt = filepp.readline().split()
                # tau[iat,:] = np.asarray(linesplt[1:4],dtype=float)
                # ityp[iat] = int(linesplt[4]) -1
                tau = np.asarray(linesplt[1:4], dtype=float)
                ity = int(linesplt[4]) - 1
                label.append(atm[ity])
                pos.append(tau)
                # atoms.append(Atom(Zval=zv[ity], label=atm[
                # ity], pos=tau, cell=grid, basis = 'Crystal'))
                # ity], pos=tau * celldm[0], cell=grid))
            pos = np.asarray(pos)
            if ibrav == 0 :
                pos *= celldm[0]
                atoms = Atom(Zval=Zval, label=label, pos=pos, cell=grid, basis="Cartesian")
            else :
                atoms = Atom(Zval=Zval, label=label, pos=pos, cell=grid, basis="Crystal")

            # self.atoms = Ions( nat, ntyp, atm, zv, tau*celldm[0], ityp, self.grid)

            # plot
            blocksize = 1024 * 8
            strings = ""
            while True:
                line = filepp.read(blocksize)
                if not line:
                    break
                strings += line
            ppgrid = np.fromstring(strings, dtype=float, sep=" ")

            # igrid = 0
            # nnr = nrx[0] * nrx[1] * nrx[2]
            # ppgrid = np.zeros(nnr, dtype=float)
            # for line in filepp:
            # line = line.split()
            # npts = len(line)
            # ppgrid[igrid:igrid + npts] = np.asarray(line, dtype=float)
            # igrid += npts

            plot = DirectField(grid=grid, griddata_F=ppgrid, rank=1)

            if data_type == 'potential' :
                plot *= 0.5 # Ry to Hartree

            return System(atoms, grid, name=self.title, field=plot)

    def writepp(self, system, **kwargs):

        with open(self.filepp, "w") as filepp:
            val_per_line = 5

            # title
            filepp.write(system.name)

            # nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp
            mywrite(filepp, system.cell.nrx, False)
            mywrite(filepp, system.cell.nr, False)
            mywrite(filepp, [len(self.atoms.ions), len(self.atoms.species)], False)

            # ibrav, celldm
            mywrite(filepp, self.cell.ibrav, True)
            mywrite(filepp, self.cell.celldm, False)

            # at(:,i) three times
            if self.cell.ibrav == 0:
                for ilat in range(3):
                    mywrite(filepp, self.cell.at[:, ilat], True)

            # gcutm, dual, ecut, plot_num
            mywrite(filepp, self.cutoffvars["gcutm"], True)
            mywrite(filepp, self.cutoffvars["dual"], False)
            mywrite(filepp, self.cutoffvars["ecut"], False)
            mywrite(filepp, self.plot.plot_num, False)

            # ntyp
            for ity, spc in enumerate(self.atoms.species):
                mywrite(filepp, [ity + 1, spc[0], spc[1]], True)

            # tau
            for iat, ion in enumerate(self.atoms.ions):
                mywrite(filepp, iat + 1, True)
                mywrite(filepp, ion.pos, False)
                mywrite(filepp, ion.typ + 1, False)

            # plot
            nlines = system.field.grid.nnr // val_per_line
            grid_pp = system.field.get_values_1darray(order="F")
            for iline in range(nlines):
                igrid = iline * val_per_line
                mywrite(filepp, grid_pp[igrid : igrid + val_per_line], True)
            igrid = (iline + 1) * val_per_line
            mywrite(filepp, grid_pp[igrid : self.grid.nnr], True)
            # pass

    def celldm2at(self, ibrav, celldm):

        at = np.zeros((3, 3), dtype=float)

        if ibrav == 1:
            at = celldm[0] * np.identity(3)
        elif ibrav == 2:
            at[:, 0] = 0.5 * celldm[0] * np.array([-1.0, 0.0, 1.0])
            at[:, 1] = 0.5 * celldm[0] * np.array([0.0, 1.0, 1.0])
            at[:, 2] = 0.5 * celldm[0] * np.array([-1.0, 1.0, 0.0])
        else:
            # implement all the other Bravais lattices
            raise NotImplementedError("celldm2at is only implemented for ibrav = 0 and ibrav = 1")

        return at

    def write(self, system, data_type = 'density', **kwargs):
        info = {
                'gcutm' : 1.0,
                'dual' : 1.0,
                'ecut' : 1.0,
                }
        info.update(kwargs)
        with open(self.filepp, "w") as filepp:
            val_per_line = 5
            grid = system.field.grid

            # title
            filepp.write(system.name)

            # nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp
            mywrite(filepp, grid.nrR, True)
            mywrite(filepp, grid.nrR, False)
            mywrite(filepp, [system.ions.nat, len(system.ions.nsymbols)], False)

            # ibrav, celldm
            celldm = np.zeros(6)
            celldm[0] = grid.latparas[0]
            mywrite(filepp, 0, True)
            mywrite(filepp, celldm, False)

            for ilat in range(3):
                mywrite(filepp, grid.lattice[:, ilat]/celldm[0], True)

            # gcutm, dual, ecut, plot_num
            mywrite(filepp, info["gcutm"], True)
            mywrite(filepp, info["dual"], False)
            mywrite(filepp, info["ecut"], False)
            mywrite(filepp, 0, False)

            # ntyp
            for ity, spc in enumerate(system.ions.nsymbols):
                mywrite(filepp, [ity + 1, spc, system.ions.Zval.get(spc, 1.0)], True)

            # tau
            tau = system.ions.pos.to_cart() / celldm[0]
            for iat in range(system.ions.nat):
                mywrite(filepp, iat + 1, True)
                mywrite(filepp, tau[iat], False)
                mywrite(filepp, np.where(system.ions.nsymbols == system.ions.labels[iat])[0] + 1, False)

            # plot
            nlines = system.field.grid.nnr // val_per_line
            grid_pp = system.field.get_values_flatarray(order="F")
            if data_type == 'potential' :
                grid_pp = grid_pp*2.0 # Hartree to Ry
            for iline in range(nlines):
                igrid = iline * val_per_line
                mywrite(filepp, grid_pp[igrid : igrid + val_per_line], True)
            igrid = (iline + 1) * val_per_line
            mywrite(filepp, grid_pp[igrid : grid.nnr], True)


class Ions(object):
    def __init__(self, nat, ntyp, atm, zv, tau, ityp, cell):
        self.species = []
        self.ions = []
        for ity in range(ntyp):
            self.species.append([atm[ity], zv[ity]])

        for iat in range(nat):
            self.ions.append(Atom(Zval=zv[ityp[iat]], pos=tau[:, iat], typ=ityp[iat], label=atm[ityp[iat]], cell=cell))


def mywrite(fileobj, iterable, newline):
    if newline:
        fileobj.write("\n  ")
    # if len(iterable) > 1 :
    try:
        for ele in iterable:
            fileobj.write(str(ele) + "    ")
    except:
        fileobj.write(str(iterable) + "    ")
