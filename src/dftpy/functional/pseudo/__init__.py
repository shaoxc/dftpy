import os
from abc import abstractmethod

import numpy as np
import scipy.special as sp
from scipy.interpolate import splrep, splev

from dftpy.ewald import CBspline
from dftpy.field import ReciprocalField, DirectField
from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.grid import DirectGrid
from dftpy.math_utils import quartic_interpolation
from dftpy.mpi import sprint, SerialComm, MP
from dftpy.time_data import TimeData

from dftpy.functional.pseudo.psp import PSP
from dftpy.functional.pseudo.recpot import RECPOT
from dftpy.functional.pseudo.upf import UPF
from dftpy.functional.pseudo.usp import USP
from dftpy.functional.pseudo.xml import PAWXML


# NEVER TOUCH THIS CLASS
# NEVER TOUCH THIS CLASS
class AbstractLocalPseudo(AbstractFunctional):
    """
    This is a pseudo potential template class and should never be touched.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def local_PP(self):
        pass

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def force(self):
        pass

    @abstractmethod
    def stress(self):
        pass

    @property
    @abstractmethod
    def v(self):
        pass

    @property
    @abstractmethod
    def vreal(self):
        pass

    @property
    @abstractmethod
    def vlines(self):
        pass


# Only touch this class if you know what you are doing
class LocalPseudo(AbstractLocalPseudo):
    """
    LocalPseudo class handles local pseudo potentials.
    This is a template class and should never be touched.
    """

    def __init__(self, grid=None, ions=None, PP_list=None, PME=True, readpp = None, comm = None, **kwargs):

        self.type = 'PSEUDO'
        self.name = 'PSEUDO'

        if comm is None:
            if grid is not None :
                comm = grid.mp.comm
            else :
                comm = SerialComm()

        # Read PP first, then initialize other variables.
        if PP_list is not None:
            self.readpp = ReadPseudo(PP_list, comm=comm, **kwargs)
        elif readpp is not None :
            self.readpp = readpp
        else:
            raise AttributeError("Must specify PP_list for Pseudopotentials")

        self.usePME = PME
        # if not PME :
            # sprint("Using N^2 method for strf!", comm=comm)
        self.restart(grid, ions)

    @property
    def _vloc_interp(self):
        return self.readpp.vloc_interp

    @property
    def _gp(self):
        return self.readpp.gp

    @property
    def _vp(self):
        return self.readpp.vp

    @property
    def PP_list(self):
        return self.readpp.PP_list

    @property
    def zval(self):
        return self.readpp.zval

    def __repr__(self):
        return 'LOCALPSEUDO'

    def restart(self, grid=None, ions=None, full=False, duplicate=False):
        """
        Clean all private data and resets the ions and grid.
        This will prompt the computation of a new pseudo
        without recomputing the local pp on the atoms.
        """
        if duplicate :
            pseudo = self.__class__(grid = grid, ions = ions, readpp = self.readpp)
            return pseudo

        self._vlines = {}  # PP for each atomic species on 3D PW grid
        self._v = None  # PP for atom on 3D PW grid
        self._vreal = None  # PP for atom on 3D real space
        self._ions = None
        self._grid = None
        if ions is not None:
            self.ions = ions
        if grid is not None:
            self.grid = grid

    @property
    def grid(self):
        if self._grid is not None:
            return self._grid
        else:
            raise AttributeError("Must specify grid for Pseudopotentials")

    @grid.setter
    def grid(self, value):
        if not isinstance(value, (DirectGrid)):
            raise TypeError("Grid must be DirectGrid")
        self._grid = value

    @property
    def ions(self):
        if self._ions is not None:
            return self._ions
        else:
            raise AttributeError("Must specify ions for Pseudopotentials")

    @ions.setter
    def ions(self, value):
        for key in value.symbols_uniq :
            if key not in self.PP_list:
                raise ValueError("There is no pseudopotential for {:4s} atom".format(key))
        self._ions = value
        # update zval in ions
        self._ions.set_charges(self.zval)

    def compute(self, density, calcType={"E", "V"}, **kwargs):
        if self._vreal is None:
            self.local_PP()
        pot = self._vreal
        if 'E' in calcType:
            if density.rank > 1:
                rho = np.sum(density, axis=0)
            else:
                rho = density
            ene = np.einsum("ijk, ijk->", self._vreal, rho) * self.grid.dV
        else:
            ene = 0.0

        if density.rank > 1:
            pot = pot.tile((density.rank, 1, 1, 1))
        return FunctionalOutput(name="eN", energy=ene, potential=pot)

    def local_PP(self, BsplineOrder=10):
        """
        """
        # if BsplineOrder is not 10:
        # warnings.warn("BsplineOrder not 10. Do you know what you are doing?")
        self.BsplineOrder = BsplineOrder

        if self._v is None:
            if self.usePME:
                self._PP_Reciprocal_PME()
            else:
                self._PP_Reciprocal()
        if self._vreal is None:
            self._vreal = self._v.ifft(force_real=True)

    def stress(self, rho, energy=None):
        if rho is None:
            raise AttributeError("Must specify rho")
        if not isinstance(rho, (DirectField)):
            raise TypeError("rho must be DirectField")
        if self.usePME:
            s = self._StressPME(rho, energy)
        else:
            s = self._Stress(rho, energy)
        return s

    def force(self, rho):
        if rho is None:
            raise AttributeError("Must specify rho")
        if not isinstance(rho, (DirectField)):
            raise TypeError("rho must be DirectField")
        if self.usePME:
            f = self._ForcePME(rho)
        else:
            f = self._Force(rho)
        return f

    @property
    def vreal(self):
        """
        The vloc represented on the real space grid.
        """
        if self._vreal is not None:
            return self._vreal
        else:
            return Exception("Must load PP first")

    @property
    def v(self):
        """
        The vloc represented on the reciprocal space grid.
        """
        if self._v is not None:
            return self._v
        else:
            return Exception("Must load PP first")

    @property
    def vlines(self):
        """
        The vloc for each atom type represented on the reciprocal space grid.
        """
        if not self._vlines:
            reciprocal_grid = self.grid.get_reciprocal()
            q = reciprocal_grid.q
            vloc = np.empty_like(q)
            for key in sorted(self._vloc_interp):
                vloc_interp = self._vloc_interp[key]
                vloc[:] = 0.0
                mask = q < self._gp[key][-1]
                qmask = q[mask]
                if len(qmask)>0 :
                    vloc[mask] = splev(qmask, vloc_interp, der=0)
                # quartic interpolation for small q
                # -----------------------------------------------------------------------
                mask = q < self._gp[key][1]
                vp = self._vp[key]
                dp = vp[1] - vp[0]
                f = [vp[2], vp[1], vp[0], vp[1], vp[2]]
                dx = q[mask] / dp
                vloc[mask] = quartic_interpolation(f, dx)
                # -----------------------------------------------------------------------
                self._vlines[key] = vloc.copy()
        return self._vlines

    def _PP_Reciprocal(self):
        TimeData.Begin("Vion")
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype=np.complex128)
        for key in sorted(self._vloc_interp):
            for i in range(len(self.ions.positions)):
                if self.ions.symbols[i] == key:
                    strf = self.ions.strf(reciprocal_grid, i)
                    v += self.vlines[key] * strf
        self._v = ReciprocalField(reciprocal_grid, griddata_3d=v)
        TimeData.End("Vion")
        return "PP successfully interpolated"

    def _PP_Reciprocal_PME(self):
        TimeData.Begin("Vion_PME")
        self.Bspline = CBspline(ions=self.ions, grid=self.grid, order=self.BsplineOrder)
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype=np.complex128)
        QA = np.empty(self.grid.nr)
        scaled_postions=self.ions.get_scaled_positions()
        for key in sorted(self._vloc_interp):
            QA[:] = 0.0
            for i in range(len(self.ions.positions)):
                if self.ions.symbols[i] == key:
                    QA = self.Bspline.get_PME_Qarray(scaled_postions[i], QA)
            Qarray = DirectField(grid=self.grid, griddata_3d=QA, rank=1)
            v = v + self.vlines[key] * Qarray.fft()
        v = v * self.Bspline.Barray * self.grid.nnrR / self.grid.volume
        self._v = v
        TimeData.End("Vion_PME")
        return "PP successfully interpolated"

    def _PP_Derivative_One(self, key=None):
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        vloc_interp = self._vloc_interp[key]
        vloc_deriv = np.zeros(np.shape(q))
        vloc_deriv[q < np.max(self._gp[key])] = splev(q[q < np.max(self._gp[key])], vloc_interp, der=1)
        return ReciprocalField(reciprocal_grid, griddata_3d=vloc_deriv)

    def _PP_Derivative(self, symbols=None):
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype=np.complex128)
        vloc_deriv = np.empty_like(q, dtype=np.complex128)
        if symbols is None:
            symbols = sorted(self._gp)
        for key in symbols:
            vloc_interp = self._vloc_interp[key]
            vloc_deriv[:] = 0.0
            vloc_deriv[q < np.max(self._gp[key])] = splev(q[q < np.max(self._gp[key])], vloc_interp, der=1)
            for i in range(len(self.ions.positions)):
                if self.ions.symbols[i] == key:
                    strf = self.ions.strf(reciprocal_grid, i)
                    v += vloc_deriv * np.conjugate(strf)
        return v

    def _Stress(self, density, energy=None):
        if density.rank > 1:
            rho = np.sum(density, axis=0)
        else:
            rho = density
        if energy is None:
            energy = self(rho, calcType={"E"}).energy
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        mask = reciprocal_grid.mask
        invq = reciprocal_grid.invq
        rhoG = rho.fft()
        stress = np.zeros((3, 3))
        v_deriv = self._PP_Derivative()
        rhoGV_q = rhoG * v_deriv * invq
        for i in range(3):
            for j in range(i, 3):
                # den = (g[i]*g[j])[np.newaxis] * rhoGV_q
                # stress[i, j] = (np.einsum('ijk->', den)).real / rho.grid.volume
                den = (g[i][mask] * g[j][mask]) * rhoGV_q[mask]
                stress[i, j] = stress[j, i] = -(np.einsum("i->", den)).real / self.grid.volume * 2.0
                if i == j:
                    stress[i, j] -= energy
        stress /= self.grid.volume
        return stress

    def _Force(self, density):
        if density.rank > 1:
            rho = np.sum(density, axis=0)
        else:
            rho = density
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        Forces = np.zeros((self.ions.nat, 3))
        mask = reciprocal_grid.mask
        for i in range(self.ions.nat):
            strf = self.ions.istrf(reciprocal_grid, i)
            den = self.vlines[self.ions.symbols[i]][mask] * (rhoG[mask] * strf[mask]).imag
            for j in range(3):
                Forces[i, j] = np.einsum("i, i->", g[j][mask], den)
        Forces *= 2.0 / self.grid.volume
        return Forces

    def _ForcePME(self, density):
        if density.rank > 1:
            rho = np.sum(density, axis=0)
        else:
            rho = density
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        Bspline = self.Bspline
        Barray = Bspline.Barray
        Barray = np.conjugate(Barray)
        denG = rhoG * Barray
        nrR = self.grid.nrR
        # cell_inv = np.linalg.inv(self.ions.positions[0].cell.lattice)
        cell_inv = reciprocal_grid.lattice.T / 2 / np.pi
        Forces = np.zeros((self.ions.nat, 3))
        ixyzA = np.mgrid[: self.BsplineOrder, : self.BsplineOrder, : self.BsplineOrder].reshape((3, -1))
        Q_derivativeA = np.zeros((3, self.BsplineOrder * self.BsplineOrder * self.BsplineOrder))
        scaled_postions=self.ions.get_scaled_positions()
        for key in self.ions.symbols_uniq :
            denGV = denG * self.vlines[key]
            if rho.mp.rank == 0 :
                denGV[0, 0, 0] = 0.0 + 0.0j
            rhoPB = denGV.ifft(force_real=True)
            for i in range(self.ions.nat):
                if self.ions.symbols[i] == key:
                    Up = scaled_postions[i] * nrR
                    if self.Bspline.check_out_cell(Up):
                        continue
                    Mn = []
                    Mn_2 = []
                    for j in range(3):
                        Mn.append(Bspline.calc_Mn(Up[j] - np.floor(Up[j])))
                        Mn_2.append(Bspline.calc_Mn(Up[j] - np.floor(Up[j]), order=self.BsplineOrder - 1))
                    Q_derivativeA[0] = nrR[0] * np.einsum(
                        "i, j, k -> ijk", Mn_2[0][1:] - Mn_2[0][:-1], Mn[1][1:], Mn[2][1:]
                    ).reshape(-1)
                    Q_derivativeA[1] = nrR[1] * np.einsum(
                        "i, j, k -> ijk", Mn[0][1:], Mn_2[1][1:] - Mn_2[1][:-1], Mn[2][1:]
                    ).reshape(-1)
                    Q_derivativeA[2] = nrR[2] * np.einsum(
                        "i, j, k -> ijk", Mn[0][1:], Mn[1][1:], Mn_2[2][1:] - Mn_2[2][:-1]
                    ).reshape(-1)
                    l123A = np.mod(1 + np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nrR.reshape((3, 1)))
                    mask = self.Bspline.get_Qarray_mask(l123A)
                    Forces[i] = -np.sum(np.matmul(Q_derivativeA.T, cell_inv)[mask] * rhoPB[
                                                                                         l123A[0][mask], l123A[1][mask],
                                                                                         l123A[2][mask]][:, np.newaxis],
                                        axis=0)
        return Forces

    def _StressPME(self, density, energy=None):
        if density.rank > 1:
            rho = np.sum(density, axis=0)
        else:
            rho = density

        if energy is None:
            energy = self(rho, calcType={"E"}).energy
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        invq = reciprocal_grid.invq
        mask = reciprocal_grid.mask
        Bspline = self.Bspline
        Barray = Bspline.Barray
        rhoGB = np.conjugate(rhoG) * Barray
        nr = self.grid.nr
        stress = np.zeros((3, 3))
        QA = np.empty(nr)
        scaled_postions=self.ions.get_scaled_positions()
        for key in self.ions.symbols_uniq :
            rhoGBV = rhoGB * self._PP_Derivative_One(key=key)
            QA[:] = 0.0
            for i in range(self.ions.nat):
                if self.ions.symbols[i] == key:
                    QA = self.Bspline.get_PME_Qarray(scaled_postions[i], QA)
            Qarray = DirectField(grid=self.grid, griddata_3d=QA, rank=1)
            rhoGBV = rhoGBV * Qarray.fft()
            for i in range(3):
                for j in range(i, 3):
                    den = (g[i][mask] * g[j][mask]) * rhoGBV[mask] * invq[mask]
                    stress[i, j] -= (np.einsum("i->", den)).real / self.grid.volume ** 2
        stress *= 2.0 * self.grid.nnrR
        for i in range(3):
            for j in range(i, 3):
                stress[j, i] = stress[i, j]
            stress[i, i] -= energy
        stress /= self.grid.volume
        return stress


PPEngines = {
            "recpot" : RECPOT,
            "usp"    : USP,
            "uspcc"  : USP,
            "uspso"  : USP,
            "upf"    : UPF,
            "psp"    : PSP,
            "psp8"   : PSP,
            "lps"    : PSP,
            "xml"    : PAWXML,
        }


class ReadPseudo(object):
    """
    Support class for LocalPseudo.
    """

    def __init__(self, PP_list=None, comm=None, parallel = True, **kwargs):
        self._gp = {}  # 1D PP grid g-space
        self._vp = {}  # PP on 1D PP grid
        self._r = {}  # 1D PP grid r-space
        self._v = {}  # PP on 1D PP grid r-space
        self._info = {}
        self._vloc_interp = {}  # Interpolates recpot PP
        self._core_density_grid = {}
        self._core_density = {}  # the radial core charge density for the non-linear core correction in g-space
        self._atomic_density_grid = {}
        self._atomic_density = {}  # the radial atomic charge density in g-space
        self._zval = {}

        self.PP_list = PP_list
        #-----------------------------------------------------------------------
        if comm is None: comm = SerialComm()
        self.comm = comm
        self.parallel = parallel and comm.size > 1
        #-----------------------------------------------------------------------

        for key in self.PP_list:
            sprint("setting key: {} -> {}".format(key, self.PP_list[key]), comm=comm)
            if not os.path.isfile(self.PP_list[key]):
                raise FileNotFoundError("'{}' PP file for atom type {} not found".format(self.PP_list[key], key))
            self._init_pp(key, **kwargs)

    @staticmethod
    def _real2recip(r, v, zval=0, MaxPoints=10000, Gmax=30, Gmin=1E-4, method='simpson', comm=None, mp=None):
        if mp is None : mp = MP(comm = comm)
        gp = np.logspace(np.log10(Gmin), np.log10(Gmax), num=MaxPoints)
        gp[0] = 0.0
        vp = np.zeros_like(gp)
        if method == 'simpson':
            try:
                # new version of scipy=1.6.0 change the name to simpson
                from scipy.integrate import simpson
            except Exception:
                from scipy.integrate import simps as simpson
            vr = (v * r + zval) * r

            lb, ub = mp.split_number(len(gp))
            if mp.is_root : lb = 1

            for k in range(lb, ub):
                y = sp.spherical_jn(0, gp[k] * r) * vr
                vp[k] = (4.0 * np.pi) * simpson(y, r)

            if mp.is_mpi : vp = mp.vsum(vp)

            vp[0] = (4.0 * np.pi) * simpson(vr, r)
        else:
            dr = np.empty_like(r)
            dr[1:] = r[1:] - r[:-1]
            dr[0] = r[0]
            vr = (v * r + zval) * r * dr

            lb, ub = mp.split_number(len(gp))
            if mp.is_root : lb = 1

            for k in range(lb, ub):
                vp[k] = (4.0 * np.pi) * np.sum(sp.spherical_jn(0, gp[k] * r) * vr)

            vp = mp.vsum(vp)

            vp[0] = (4.0 * np.pi) * np.sum(vr)
        vp[1:] -= 4.0 * np.pi * zval / (gp[1:] ** 2)
        return gp, vp

    @staticmethod
    def _recip2real(gp, vp, MaxPoints=1001, Rmax=10, r=None, comm=None, mp=None):
        if mp is None : mp = MP(comm = comm)
        if r is None:
            r = np.linspace(0, Rmax, MaxPoints)
        v = np.zeros_like(r)
        dg = np.empty_like(gp)
        dg[1:] = gp[1:] - gp[:-1]
        dg[0] = gp[0]
        vr = vp * gp * gp * dg

        lb, ub = mp.split_number(len(r))

        for i in range(lb, ub):
            v[i] = (0.5 / np.pi ** 2) * np.sum(sp.spherical_jn(0, r[i] * gp) * vr)

        v = mp.vsum(v)
        return r, v

    @staticmethod
    def _vloc2rho(r, v, r2=None):
        if r2 is None: r2 = r
        tck = splrep(r, v)
        dv1 = splev(r2[1:], tck, der=1)
        dv2 = splev(r2[1:], tck, der=2)
        rhop = np.empty_like(r2)
        rhop[1:] = 1.0 / (4 * np.pi) * (2.0 / r2[1:] * dv1 + dv2)
        rhop[0] = rhop[1]
        return rhop

    def _self_energy(self, r, vr, rhop):
        dr = np.empty_like(r)
        dr[1:] = r[1:] - r[:-1]
        dr[0] = r[0]
        ene = np.sum(r * r * vr * rhop * dr) * 4 * np.pi
        # sprint('Ne ', np.sum(r *r * rhop * dr) * 4 * np.pi)
        return ene

    def _init_pp(self, key, engine = None, k = 3, **kwargs):
        """ initialize the pseudopotential

        Parameters
        ----------
        key :
            The key of pseudopotential, which usually is the symbol of ion.
        engine :
            The engine for reading pseudopotential file
        k :
            k: The degree of the spline fit of splrep.
        kwargs :
            kwargs
        """
        if engine is None :
            suffix = os.path.splitext(self.PP_list[key])[1][1:].lower()
            engine = PPEngines.get(suffix, None)

        if engine is None :
            raise AttributeError("Pseudopotential '{}' is not supported".format(self.PP_list[key]))
        else :
            if self.parallel :
                if self.comm.rank == 0 :
                    pp = engine(self.PP_list[key])
                else :
                    pp = None
                pp = self.comm.bcast(pp)
            else :
                pp = engine(self.PP_list[key])

        self.pp = pp

        self._gp[key] = pp.radial_grid
        self._vp[key] = pp.local_potential
        self._zval[key] = pp.zval
        self._info[key] = pp.info
        self._core_density[key] = pp.core_density
        self._core_density_grid[key]= pp.core_density_grid
        self._atomic_density[key] = pp.atomic_density
        self._atomic_density_grid[key] = pp.atomic_density_grid

        if pp.direct :
            if self.parallel :
                comm = self.comm
            else :
                comm = None
            self._r[key] = self._gp[key]
            self._v[key] = self._vp[key]
            self._gp[key], self._vp[key] = self._real2recip(self._r[key], self._v[key], self._zval[key], comm=comm, **kwargs)
            if self._core_density[key] is not None :
                self._core_density_grid[key], self._core_density[key] = self._real2recip(self._core_density_grid[key], self._core_density[key], 0, comm=comm, **kwargs)
            if self._atomic_density[key] is not None :
                self._atomic_density_grid[key], self._atomic_density[key] = self._real2recip(self._atomic_density_grid[key], self._atomic_density[key], 0, comm=comm, **kwargs)
        self._vloc_interp[key] = splrep(self._gp[key][1:], self._vp[key][1:], k=k)

    @property
    def vloc_interp(self):
        if not self._vloc_interp:
            raise AttributeError("Must init ReadPseudo")
        return self._vloc_interp

    @property
    def gp(self):
        if not self._gp:
            raise AttributeError("Must init ReadPseudo")
        return self._gp

    @property
    def vp(self):
        if not self._vp:
            raise AttributeError("Must init ReadPseudo")
        return self._vp

    @property
    def zval(self):
        if not self._zval:
            raise AttributeError("Must init ReadPseudo")
        return self._zval

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value
