import os
import numpy as np
from scipy.interpolate import splrep, splev
import scipy.special as sp
import importlib.util

from dftpy.mpi import sprint
from dftpy.field import ReciprocalField, DirectField
from dftpy.functional_output import Functional
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.ewald import CBspline
from dftpy.time_data import TimeData
from dftpy.atom import Atom
from abc import ABC, abstractmethod
from dftpy.grid import DirectGrid
from dftpy.math_utils import quartic_interpolation
from dftpy.pseudo.upf import UPF, UPFJSON
from dftpy.pseudo.usp import USP
from dftpy.pseudo.psp import PSP
from dftpy.pseudo.recpot import RECPOT


# NEVER TOUCH THIS CLASS
# NEVER TOUCH THIS CLASS
class AbstractLocalPseudo(ABC):
    """
    This is a pseudo potential template class and should never be touched.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
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

    def __init__(self, grid=None, ions=None, PP_list=None, PME=True, **kwargs):

        if PP_list is not None:
            self.PP_list = PP_list
        else:
            raise AttributeError("Must specify PP_list for Pseudopotentials")
        # Read PP first, then initialize other variables.
        if grid is None :
            comm = None
        else :
            comm = grid.mp.comm
        readPP = ReadPseudo(PP_list, comm=comm, **kwargs)
        self.readpp = readPP

        self.restart(grid, ions, full=True)

        # if not PME :
            # sprint("Using N^2 method for strf!")

        self.usePME = PME

        self._vloc_interp = readPP.vloc_interp
        self._gp = readPP.gp
        self._vp = readPP.vp
        self.zval = {}

    def restart(self, grid=None, ions=None, full=False):
        """
        Clean all private data and resets the ions and grid.
        This will prompt the computation of a new pseudo
        without recomputing the local pp on the atoms.
        """
        if full:
            self._gp = {}  # 1D PP grid g-space
            self._vp = {}  # PP on 1D PP grid
            self._vloc_interp = {}  # Interpolates recpot PP
        self._vlines = {} # PP for each atomic species on 3D PW grid
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
        self._grid= value

    @property
    def ions(self):
        if self._ions is not None:
            return self._ions
        else:
            raise AttributeError("Must specify ions for Pseudopotentials")

    @ions.setter
    def ions(self, value):
        if not isinstance(value, (Atom)):
            raise TypeError("Ions must be an array of Atom classes")
        for key in set(value.labels[:]):
            if key not in self.PP_list :
                raise ValueError("There is no pseudopotential for {:4s} atom".format(key))
        self._ions = value
        # update Zval in ions
        self.readpp.get_Zval(self._ions)

    def __call__(self, density=None, calcType={"E", "V"}, **kwargs):
        if self._vreal is None:
            self.local_PP()
        pot = self._vreal
        if 'E' in calcType:
            if density.rank > 1 :
                rho = np.sum(density, axis = 0)
            else :
                rho = density
            ene = np.einsum("ijk, ijk->", self._vreal, rho) * self.grid.dV
        else:
            ene = 0.0

        if density.rank > 1 :
            pot = np.tile(pot, (density.rank, 1, 1, 1))
        return Functional(name="eN", energy=ene, potential=pot)

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
            for key in sorted(self._vloc_interp) :
                vloc_interp = self._vloc_interp[key]
                vloc[:] = 0.0
                mask = q < self._gp[key][-1]
                vloc[mask] = splev(q[mask], vloc_interp, der=0)
                # quartic interpolation for small q
                #-----------------------------------------------------------------------
                mask = q < self._gp[key][1]
                vp = self._vp[key]
                dp = vp[1]-vp[0]
                f = [vp[2], vp[1], vp[0], vp[1], vp[2]]
                dx = q[mask]/dp
                vloc[mask] = quartic_interpolation(f, dx)
                #-----------------------------------------------------------------------
                # mask[0, 0, 0] = False
                # vloc[mask] = splev(q[mask], vloc_interp, der=0)-self.zval[key]/gg[mask]
                self._vlines[key] = vloc.copy()
        return self._vlines

    def _PP_Reciprocal(self):
        TimeData.Begin("Vion")
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype=np.complex128)
        for key in sorted(self._vloc_interp):
            for i in range(len(self.ions.pos)):
                if self.ions.labels[i] == key:
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
        for key in sorted(self._vloc_interp):
            QA[:] = 0.0
            for i in range(len(self.ions.pos)):
                if self.ions.labels[i] == key:
                    QA = self.Bspline.get_PME_Qarray(i, QA)
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

    def _PP_Derivative(self, labels=None):
        reciprocal_grid = self.grid.get_reciprocal()
        q = reciprocal_grid.q
        v = np.zeros_like(q, dtype=np.complex128)
        vloc_deriv = np.empty_like(q, dtype=np.complex128)
        if labels is None:
            labels = sorted(self._gp)
        for key in labels:
            vloc_interp = self._vloc_interp[key]
            vloc_deriv[:] = 0.0
            vloc_deriv[q < np.max(self._gp[key])] = splev(q[q < np.max(self._gp[key])], vloc_interp, der=1)
            for i in range(len(self.ions.pos)):
                if self.ions.labels[i] == key:
                    strf = self.ions.strf(reciprocal_grid, i)
                    v += vloc_deriv * np.conjugate(strf)
        return v

    def _Stress(self, density, energy=None):
        if density.rank > 1 :
            rho = np.sum(density, axis = 0)
        else :
            rho = density
        if energy is None:
            energy = self(density=rho, calcType={"E"}).energy
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
        if density.rank > 1 :
            rho = np.sum(density, axis = 0)
        else :
            rho = density
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        Forces = np.zeros((self.ions.nat, 3))
        mask = reciprocal_grid.mask
        for i in range(self.ions.nat):
            strf = self.ions.istrf(reciprocal_grid, i)
            den = self.vlines[self.ions.labels[i]][mask] * (rhoG[mask] * strf[mask]).imag
            for j in range(3):
                Forces[i, j] = np.einsum("i, i->", g[j][mask], den)
        Forces *= 2.0 / self.grid.volume
        return Forces

    def _ForcePME(self, density):
        if density.rank > 1 :
            rho = np.sum(density, axis = 0)
        else :
            rho = density
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        Bspline = self.Bspline
        Barray = Bspline.Barray
        Barray = np.conjugate(Barray)
        denG = rhoG * Barray
        nrR = self.grid.nrR
        # cell_inv = np.linalg.inv(self.ions.pos[0].cell.lattice)
        cell_inv = reciprocal_grid.lattice.T / 2 / np.pi
        Forces = np.zeros((self.ions.nat, 3))
        ixyzA = np.mgrid[: self.BsplineOrder, : self.BsplineOrder, : self.BsplineOrder].reshape((3, -1))
        Q_derivativeA = np.zeros((3, self.BsplineOrder * self.BsplineOrder * self.BsplineOrder))
        for key in sorted(self.ions.Zval):
            denGV = denG * self.vlines[key]
            denGV[0, 0, 0] = 0.0 + 0.0j
            rhoPB = denGV.ifft(force_real=True)
            for i in range(self.ions.nat):
                if self.ions.labels[i] == key:
                    Up = np.array(self.ions.pos[i].to_crys()) * nrR
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
                    Forces[i] = -np.sum(np.matmul(Q_derivativeA.T, cell_inv)[mask] * rhoPB[l123A[0][mask], l123A[1][mask], l123A[2][mask]][:, np.newaxis], axis=0)
        return Forces

    def _StressPME(self, density, energy=None):
        if density.rank > 1 :
            rho = np.sum(density, axis = 0)
        else :
            rho = density

        if energy is None:
            energy = self(density=rho, calcType={"E"}).energy
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
        for key in sorted(self.ions.Zval):
            rhoGBV = rhoGB * self._PP_Derivative_One(key=key)
            QA[:] = 0.0
            for i in range(self.ions.nat):
                if self.ions.labels[i] == key:
                    QA = self.Bspline.get_PME_Qarray(i, QA)
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


class ReadPseudo(object):
    """
    Support class for LocalPseudo.
    """

    def __init__(self, PP_list=None, MaxPoints = 10000, Gmax = 30, Gmin = 1E-4, Rmax = 10, comm = None):
        self._gp = {}  # 1D PP grid g-space
        self._vp = {}  # PP on 1D PP grid
        self._r = {}  # 1D PP grid r-space
        self._v = {}  # PP on 1D PP grid r-space
        self._info = {}
        self._vloc_interp = {}  # Interpolates recpot PP
        self._core_density = {}  # the radial core charge density for the non-linear core correction

        self.PP_list = PP_list
        self.comm = comm

        for key in self.PP_list:
            sprint("setting key: {} -> {}".format(key, self.PP_list[key]), comm = comm)
            if not os.path.isfile(self.PP_list[key]):
                raise FileNotFoundError("'{}' PP file for atom type {} not found".format(self.PP_list[key], key))
            if PP_list[key].lower().endswith("recpot"):
                self._init_PP_recpot(key)
            elif PP_list[key].lower().endswith(("usp", "uspcc", "uspso")):
                self._init_PP_usp(key)
            elif PP_list[key].lower().endswith("upf"):
                self._init_PP_upf(key, MaxPoints = MaxPoints, Gmax = Gmax, Gmin = Gmin)
            elif PP_list[key].lower().endswith(("psp", "psp8")):
                self._init_PP_psp(key, MaxPoints = MaxPoints, Gmax = Gmax, Gmin = Gmin)
            else:
                raise AttributeError("Pseudopotential not supported")

            self.get_vloc_interp(key)

    def get_vloc_interp(self, key, k = 3):
        """get the representation of PP

        Args:
            key: Atomic symbol
            k: The degree of the spline fit of splrep, should keep use 3.
        """
        vloc_interp = splrep(self._gp[key][1:], self._vp[key][1:], k=k)
        self._vloc_interp[key] = vloc_interp

    @staticmethod
    def _real2recip(r, v, zval, MaxPoints=10000, Gmax=30, Gmin=1E-4, method='simpson'):
        gp = np.logspace(np.log10(Gmin), np.log10(Gmax), num = MaxPoints)
        vp = np.empty_like(gp)
        if method == 'simpson' :
            try:
                #new version of scipy=1.6.0 change the name to simpson
                from scipy.integrate import simpson
            except Exception :
                from scipy.integrate import simps as simpson
            vr = (v * r + zval) * r
            for k in range(1, len(gp)):
                y = sp.spherical_jn(0, gp[k] * r) * vr
                vp[k] = (4.0 * np.pi) * simpson(y, r)
            vp[0] = (4.0 * np.pi) * simpson(vr, r)
        else :
            dr = np.empty_like(r)
            dr[1:] = r[1:]-r[:-1]
            dr[0] = r[0]
            vr = (v * r + zval) * r * dr
            for k in range(1, len(gp)):
                vp[k] = (4.0 * np.pi) * np.sum(sp.spherical_jn(0, gp[k] * r) * vr)
            vp[0] = (4.0 * np.pi) * np.sum(vr)
        vp[1:] -= 4.0 * np.pi * zval / (gp[1:] ** 2)
        return gp, vp

    @staticmethod
    def _recip2real(gp, vp, MaxPoints=1001, Rmax=10, r=None):
        if r is None :
            r = np.linspace(0, Rmax, MaxPoints)
        v = np.empty_like(r)
        dg = np.empty_like(gp)
        dg[1:] = gp[1:]-gp[:-1]
        dg[0] = gp[0]
        vr = vp * gp * gp * dg
        for i in range(0, len(r)):
            v[i] = (0.5 / np.pi ** 2) * np.sum(sp.spherical_jn(0, r[i] * gp) * vr)
        return r, v

    @staticmethod
    def _vloc2rho(r, v, r2 = None):
        if r2 is None : r2 = r
        tck = splrep(r, v)
        dv1 = splev(r2[1:], tck, der = 1)
        dv2 = splev(r2[1:], tck, der = 2)
        rhop = np.empty_like(r2)
        rhop[1:] = 1.0/(4 * np.pi) * (2.0/r2[1:] * dv1 + dv2)
        rhop[0] = rhop[1]
        return rhop

    def get_Zval(self, ions):
        for key in self._gp :
                ions.Zval[key] = self._info[key].zval
        self.zval = ions.Zval.copy()

    def _self_energy(self, r, vr, rhop):
        dr = np.empty_like(r)
        dr[1:] = r[1:]-r[:-1]
        dr[0] = r[0]
        ene = np.sum(r *r * vr * rhop * dr) * 4 * np.pi
        # sprint('Ne ', np.sum(r *r * rhop * dr) * 4 * np.pi)
        return ene

    def _init_PP_recpot(self, key):
        self._info[key] = RECPOT(self.PP_list[key])
        self._gp[key] = self._info[key].r_g
        self._vp[key] = self._info[key].v_g

    def _init_PP_usp(self, key):
        """
        !!! NOT FULL TEST !!!
        """
        self._info[key] = USP(self.PP_list[key])
        self._gp[key] = self._info[key].r_g
        self._vp[key] = self._info[key].v_g

    def _init_PP_upf(self, key, **kwargs):
        """
        Note :
            Prefer xmltodict which is more robust
            xmltodict not work for UPF v1
        """
        has_xml = importlib.util.find_spec("xmltodict")
        has_json = importlib.util.find_spec("upf_to_json")
        if has_xml :
            try :
                self._info[key] = UPF(self.PP_list[key])
            except :
                if has_json :
                    try :
                        self._info[key] = UPFJSON(self.PP_list[key])
                    except :
                        raise ModuleNotFoundError("Please use standard 'UPF' file")
        elif has_json :
            try :
                self._info[key] = UPFJSON(self.PP_list[key])
            except :
                raise ModuleNotFoundError("Maybe you can try install xmltodict or use standard 'UPF' file")
        else :
            raise ModuleNotFoundError("Must pip install xmltodict or upf_to_json")

        self._r[key] = self._info[key].r
        self._v[key] = self._info[key].v
        self._gp[key], self._vp[key] = self._real2recip(self._r[key], self._v[key], self._info[key].zval, **kwargs)
        self._core_density[key] = self._info[key].core_density

    def _init_PP_psp(self, key, **kwargs):
        self._info[key] = PSP(self.PP_list[key])
        self._r[key] = self._info[key].r
        self._v[key] = self._info[key].v
        self._gp[key], self._vp[key] = self._real2recip(self._r[key], self._v[key], self._info[key].zval, **kwargs)

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
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value
