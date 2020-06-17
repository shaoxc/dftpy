import os
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
import scipy.special as sp
from dftpy.base import Coord
from dftpy.field import ReciprocalField, DirectField
from dftpy.functional_output import Functional
from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.ewald import CBspline
from dftpy.time_data import TimeData
from dftpy.atom import Atom
from abc import ABC, abstractmethod
from dftpy.grid import DirectGrid, ReciprocalGrid
import warnings
from dftpy.math_utils import quartic_interpolation


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
        readPP = ReadPseudo(PP_list)
        self.readpp = readPP

        self.restart(grid, ions, full=True)

        if not PME :
            warnings.warn("Using N^2 method for strf!")

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
            if key not in self.PP_list.keys():
                raise ValueError("There is no pseudopotential for {:4s} atom".format(key))
        self._ions = value
        # update Zval in ions
        self.readpp.get_Zval(self._ions)

    def __call__(self, density=None, calcType=["E", "V"]):
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
            return self._StressPME(rho, energy)
        else:
            return self._Stress(rho, energy)

    def force(self, rho):
        if rho is None:
            raise AttributeError("Must specify rho")
        if not isinstance(rho, (DirectField)):
            raise TypeError("rho must be DirectField")
        if self.usePME:
            return self._ForcePME(rho)
        else:
            return self._Force(rho)

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
            gg = reciprocal_grid.gg
            vloc = np.empty_like(q)
            for key in self._vloc_interp.keys():
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
        for key in self._vloc_interp.keys():
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
        for key in self._vloc_interp.keys():
            QA[:] = 0.0
            for i in range(len(self.ions.pos)):
                if self.ions.labels[i] == key:
                    QA = self.Bspline.get_PME_Qarray(i, QA)
            Qarray = DirectField(grid=self.grid, griddata_3d=QA, rank=1)
            v = v + self.vlines[key] * Qarray.fft()
        v = v * self.Bspline.Barray * self.grid.nnr / self.grid.volume
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
            labels = self._gp.keys()
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
            energy = self(density=rho, calcType=["E"]).energy
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        # gg= reciprocal_grid.gg
        mask = reciprocal_grid.mask
        q = reciprocal_grid.q
        q[0, 0, 0] = 1.0
        rhoG = rho.fft()
        stress = np.zeros((3, 3))
        v_deriv = self._PP_Derivative()
        rhoGV_q = rhoG * v_deriv / q
        for i in range(3):
            for j in range(i, 3):
                # den = (g[i]*g[j])[np.newaxis] * rhoGV_q
                # stress[i, j] = (np.einsum('ijk->', den)).real / rho.grid.volume
                den = (g[i][mask] * g[j][mask]) * rhoGV_q[mask]
                stress[i, j] = stress[j, i] = -(np.einsum("i->", den)).real / self.grid.volume * 2.0
                if i == j:
                    stress[i, j] -= energy
        stress /= self.grid.volume
        q[0, 0, 0] = 0.0
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
        nr = self.grid.nr
        # cell_inv = np.linalg.inv(self.ions.pos[0].cell.lattice)
        cell_inv = reciprocal_grid.lattice / 2 / np.pi
        Forces = np.zeros((self.ions.nat, 3))
        ixyzA = np.mgrid[: self.BsplineOrder, : self.BsplineOrder, : self.BsplineOrder].reshape((3, -1))
        Q_derivativeA = np.zeros((3, self.BsplineOrder * self.BsplineOrder * self.BsplineOrder))
        for key in self.ions.Zval.keys():
            denGV = denG * self.vlines[key]
            denGV[0, 0, 0] = 0.0 + 0.0j
            rhoPB = denGV.ifft(force_real=True)
            for i in range(self.ions.nat):
                if self.ions.labels[i] == key:
                    Up = np.array(self.ions.pos[i].to_crys()) * nr
                    Mn = []
                    Mn_2 = []
                    for j in range(3):
                        Mn.append(Bspline.calc_Mn(Up[j] - np.floor(Up[j])))
                        Mn_2.append(Bspline.calc_Mn(Up[j] - np.floor(Up[j]), order=self.BsplineOrder - 1))
                    Q_derivativeA[0] = nr[0] * np.einsum(
                        "i, j, k -> ijk", Mn_2[0][1:] - Mn_2[0][:-1], Mn[1][1:], Mn[2][1:]
                    ).reshape(-1)
                    Q_derivativeA[1] = nr[1] * np.einsum(
                        "i, j, k -> ijk", Mn[0][1:], Mn_2[1][1:] - Mn_2[1][:-1], Mn[2][1:]
                    ).reshape(-1)
                    Q_derivativeA[2] = nr[2] * np.einsum(
                        "i, j, k -> ijk", Mn[0][1:], Mn[1][1:], Mn_2[2][1:] - Mn_2[2][:-1]
                    ).reshape(-1)
                    l123A = np.mod(1 + np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nr.reshape((3, 1)))
                    Forces[i] = -np.sum(
                        np.matmul(Q_derivativeA.T, cell_inv) * rhoPB[l123A[0], l123A[1], l123A[2]][:, np.newaxis],
                        axis=0,
                    )
        return Forces

    def _StressPME(self, density, energy=None):
        if density.rank > 1 :
            rho = np.sum(density, axis = 0)
        else :
            rho = density

        if energy is None:
            energy = self(density=rho, calcType=["E"]).energy
        rhoG = rho.fft()
        reciprocal_grid = self.grid.get_reciprocal()
        g = reciprocal_grid.g
        q = reciprocal_grid.q
        q[0, 0, 0] = 1.0
        mask = reciprocal_grid.mask
        Bspline = self.Bspline
        Barray = Bspline.Barray
        rhoGB = np.conjugate(rhoG) * Barray
        nr = self.grid.nr
        stress = np.zeros((3, 3))
        QA = np.empty(nr)
        for key in self.ions.Zval.keys():
            rhoGBV = rhoGB * self._PP_Derivative_One(key=key)
            QA[:] = 0.0
            for i in range(self.ions.nat):
                if self.ions.labels[i] == key:
                    QA = self.Bspline.get_PME_Qarray(i, QA)
            Qarray = DirectField(grid=self.grid, griddata_3d=QA, rank=1)
            rhoGBV = rhoGBV * Qarray.fft()
            for i in range(3):
                for j in range(i, 3):
                    den = (g[i][mask] * g[j][mask]) * rhoGBV[mask] / q[mask]
                    stress[i, j] -= (np.einsum("i->", den)).real / self.grid.volume ** 2
        stress *= 2.0 * self.grid.nnr
        for i in range(3):
            for j in range(i, 3):
                stress[j, i] = stress[i, j]
            stress[i, i] -= energy
        stress /= self.grid.volume
        q[0, 0, 0] = 0.0
        return stress


class ReadPseudo(object):
    """
    Support class for LocalPseudo.
    """

    def __init__(self, PP_list=None, MaxPoints = 15000, Gmax = 30, Rmax = 10):
        self._gp = {}  # 1D PP grid g-space
        self._vp = {}  # PP on 1D PP grid
        self._vloc_interp = {}  # Interpolates recpot PP
        self._info = {}

        self.PP_list = PP_list
        self.PP_type = {}

        for key in self.PP_list:
            print("setting key: " + key)
            if not os.path.isfile(self.PP_list[key]):
                raise Exception("PP file for atom type " + str(key) + " not found")
            if PP_list[key].lower().endswith("recpot"):
                self.PP_type[key] = "recpot"
                self._init_PP_recpot(key)
            elif PP_list[key].lower().endswith("usp"):
                self.PP_type[key] = "usp"
                self._init_PP_usp(key)
            elif PP_list[key].lower().endswith("uspcc"):
                self.PP_type[key] = "usp"
                self._init_PP_usp(key)
            elif PP_list[key].lower().endswith("uspso"):
                self.PP_type[key] = "uspso"
                self._init_PP_usp(key, 'uspso')
            elif PP_list[key].lower().endswith("upf"):
                self.PP_type[key] = "upf"
                self._init_PP_upf(key, MaxPoints, Gmax)
            elif PP_list[key].lower().endswith(("psp", "psp8")):
                self.PP_type[key] = "psp"
                self._init_PP_psp(key, MaxPoints, Gmax)
            else:
                raise Exception("Pseudopotential not supported")

    def _real2recip(self, r, v, zval, MaxPoints=15000, Gmax=30):
        gp = np.linspace(start=0, stop=Gmax, num=MaxPoints)
        vp = np.empty_like(gp)
        dr = np.empty_like(r)
        dr[1:] = r[1:]-r[:-1]
        dr[0] = r[0]
        vr = (v * r + zval) * r * dr
        for k in range(1, len(gp)):
            vp[k] = (4.0 * np.pi) * np.sum(sp.spherical_jn(0, gp[k] * r) * vr)
        vp[1:] -= 4.0 * np.pi * zval / (gp[1:] ** 2)
        vp[0] = (4.0 * np.pi) * np.sum(vr)
        return gp, vp

    def _recip2real(self, gp, vp, MaxPoints=1001, Rmax=10, r=None):
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

    def _vloc2rho(self, r, v, r2 = None):
        if r2 is None : r2 = r
        tck = splrep(r, v)
        dv1 = splev(r2[1:], tck, der = 1)
        dv2 = splev(r2[1:], tck, der = 2)
        rhop = np.empty_like(r2)
        rhop[1:] = 1.0/(4 * np.pi) * (2.0/r2[1:] * dv1 + dv2)
        rhop[0] = rhop[1]
        return rhop

    def get_Zval(self, ions):
        for key in self._gp.keys():
            if self.PP_type[key] == "upf":
                ions.Zval[key] = self._info[key]["pseudo_potential"]["header"]["z_valence"]
            elif self.PP_type[key] == "psp":
                ions.Zval[key] = self._info[key]["Zval"]
            elif self.PP_type[key] in ["recpot", "usp", "uspso"] :
                gp = self._gp[key]
                vp = self._vp[key]
                val = (vp[0] - vp[1]) * (gp[1] ** 2) / (4.0 * np.pi)
                # val = (vp[0] - vp[1]) * (gp[-1] / (gp.size - 1)) ** 2 / (4.0 * np.pi)
                ions.Zval[key] = round(val)
        self.zval = ions.Zval.copy()

    def _init_PP_recpot(self, key):
        """
        This is a private method used only in this specific class. 
        """

        def set_PP(Single_PP_file):
            """Reads CASTEP-like recpot PP file
            Returns tuple (g, v)"""
            # HARTREE2EV = 27.2113845
            # BOHR2ANG   = 0.529177211
            HARTREE2EV = ENERGY_CONV["Hartree"]["eV"]
            BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
            with open(Single_PP_file, "r") as outfil:
                lines = outfil.readlines()

            ibegin = 0
            for i in range(0, len(lines)):
                line = lines[i]
                if "END COMMENT" in line:
                    ibegin = i + 3
                elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 1) :
                    iend = i
                    break

            line = " ".join([line.strip() for line in lines[ibegin:iend]])

            if "1000" in lines[iend] or len(lines[iend].strip()) == 1 :
                print("Recpot pseudopotential " + Single_PP_file + " loaded")
            else:
                return Exception
            gmax = np.float(lines[ibegin - 1].strip()) * BOHR2ANG
            v = np.array(line.split()).astype(np.float) / HARTREE2EV / BOHR2ANG ** 3
            g = np.linspace(0, gmax, num=len(v))
            # self._recip2real(g, v)
            return g, v

        gp, vp = set_PP(self.PP_list[key])
        self._gp[key] = gp
        self._vp[key] = vp
        vloc_interp = splrep(gp, vp)
        self._vloc_interp[key] = vloc_interp

    def _init_PP_usp(self, key, ext = 'usp'):
        """
        !!! NOT FULL TEST !!! 
        This is a private method used only in this specific class. 
        """

        def set_PP(Single_PP_file):
            """Reads CASTEP-like usp PP file
            Returns tuple (g, v)"""
            HARTREE2EV = ENERGY_CONV["Hartree"]["eV"]
            BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
            with open(Single_PP_file, "r") as outfil:
                lines = outfil.readlines()

            ibegin = 0
            for i in range(0, len(lines)):
                line = lines[i]
                if ext == 'usp' :
                    if "END COMMENT" in line:
                        ibegin = i + 4
                    elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 1) and i - ibegin > 4:
                        iend = i
                        break
                elif ext == 'uspso' :
                    if "END COMMENT" in line:
                        ibegin = i + 5
                    elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 5) and i - ibegin > 4:
                        iend = i
                        break

            line = " ".join([line.strip() for line in lines[ibegin:iend]])
            info = {}

            Zval = np.float(lines[ibegin - 2].strip())
            info['Zval'] = Zval

            if "1000" in lines[iend] or len(lines[iend].strip()) == 1 or len(lines[iend].strip()) == 5 :
                print("Recpot pseudopotential " + Single_PP_file + " loaded")
            else:
                raise AttributeError("Error : Check the PP file")
            gmax = np.float(lines[ibegin - 1].split()[0]) * BOHR2ANG
                
            # v = np.array(line.split()).astype(np.float) / (HARTREE2EV*BOHR2ANG ** 3 * 4.0 * np.pi)
            v = np.array(line.split()).astype(np.float) / (HARTREE2EV*BOHR2ANG ** 3)
            g = np.linspace(0, gmax, num=len(v))
            v[1:] -= Zval * 4.0 * np.pi / g[1:] ** 2
            #-----------------------------------------------------------------------
            nlcc = int(lines[ibegin - 1].split()[1])
            if nlcc == 2 and ext == 'usp' :
                #num_projectors
                for i in range(iend, len(lines)):
                    l = lines[i].split()
                    if len(l) == 2 and all([item.isdigit() for item in l]):
                        ibegin = i + 1
                        ngrid = int(l[1])
                        break
                core_grid = []
                for i in range(ibegin, len(lines)):
                    l = list(map(float, lines[i].split()))
                    core_grid.extend(l)
                    if len(core_grid) >= ngrid :
                        core_grid = core_grid[:ngrid]
                        break
                info['core_grid'] = np.asarray(core_grid) * BOHR2ANG
                line = " ".join([line.strip() for line in lines[ibegin:]])
                data = np.array(line.split()).astype(np.float)
                info['core_value'] = data[-ngrid:]
            #-----------------------------------------------------------------------
            return g, v, info

        gp, vp, self._info[key] = set_PP(self.PP_list[key])
        self._gp[key] = gp
        self._vp[key] = vp
        vloc_interp = splrep(gp, vp)
        self._vloc_interp[key] = vloc_interp

    def _init_PP_upf(self, key, MaxPoints=15000, Gmax=30):
        """
        This is a private method used only in this specific class. 
        """

        def set_PP(Single_PP_file):
            """Reads QE UPF type PP"""
            import importlib.util

            upf2json = importlib.util.find_spec("upf_to_json")
            found = upf2json is not None
            if found:
                from upf_to_json import upf_to_json
            else:
                raise ModuleNotFoundError("Must pip install upf_to_json")
            with open(Single_PP_file, "r") as outfil:
                upf = upf_to_json(upf_str=outfil.read(), fname=Single_PP_file)
            r = np.array(upf["pseudo_potential"]["radial_grid"], dtype=np.float64)
            # v = np.array(upf["pseudo_potential"]["local_potential"], dtype=np.float64) 
            v = np.array(upf["pseudo_potential"]["local_potential"], dtype=np.float64)
            return r, v, upf

        r, vr, self._info[key] = set_PP(self.PP_list[key])
        zval = self._info[key]["pseudo_potential"]["header"]["z_valence"]
        gp, vp = self._real2recip(r, vr, zval, MaxPoints, Gmax)
        self._gp[key] = gp
        self._vp[key] = vp
        vloc_interp = splrep(gp, vp)
        self._vloc_interp[key] = vloc_interp

    def _self_energy(self, r, vr, rhop):
        dr = np.empty_like(r)
        dr[1:] = r[1:]-r[:-1]
        dr[0] = r[0]
        ene = np.sum(r *r * vr * rhop * dr) * 4 * np.pi
        print('Ne ', np.sum(r *r * rhop * dr) * 4 * np.pi)
        return ene

    # def _init_PP_psp(self, MaxPoints=15000, Gmax=30):
    def _init_PP_psp(self, key, MaxPoints=200000, Gmax=30):
        """
        """

        def set_PP(Single_PP_file):
            # Only support psp8 format
            # HARTREE2EV = ENERGY_CONV["Hartree"]["eV"]
            # BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
            with open(Single_PP_file, "r") as outfil:
                lines = outfil.readlines()
            info = {}

            # line 2 :atomic number, pseudoion charge, date
            values = lines[1].split()
            atomicnum = int(float(values[0]))
            Zval = float(values[1])
            # line 3 :pspcod,pspxc,lmax,lloc,mmax,r2well
            values = lines[2].split()
            if int(values[0]) != 8 :
                raise AttributeError("Only support psp8 format pseudopotential with psp")
            info['info'] = lines[:6]
            info['atomicnum'] = atomicnum
            info['Zval'] = Zval
            info['pspcod'] = 8
            info['pspxc'] = int(values[1])
            info['lmax'] = int(values[2])
            info['lloc'] = int(values[3])
            info['r2well'] = int(values[5])
            # pspxc = int(value[1])
            mmax = int(values[4])

            ibegin = 7
            iend = ibegin + mmax
            # line = " ".join([line for line in lines[ibegin:iend]])
            # data = np.fromstring(line, dtype=float, sep=" ")
            # data = np.array(line.split()).astype(np.float) / HARTREE2EV / BOHR2ANG ** 3
            data = [line.split()[1:3] for line in lines[ibegin:iend]]
            data = np.asarray(data, dtype = float)

            r = data[:, 0] 
            v = data[:, 1] 
            print("psp pseudopotential " + Single_PP_file + " loaded")
            info['grid'] = r
            info['local_potential'] = v
            return r, v, info

        r, v, self._info[key] = set_PP(self.PP_list[key])
        zval = self._info[key]['Zval']
        gp, vp = self._real2recip(r, v, zval, MaxPoints, Gmax)
        self._gp[key] = gp
        self._vp[key] = vp
        vloc_interp = splrep(gp, vp)
        self._vloc_interp[key] = vloc_interp

    @property
    def vloc_interp(self):
        if not self._vloc_interp:
            raise Exception("Must init ReadPseudo")
        return self._vloc_interp

    @property
    def gp(self):
        if not self._gp:
            raise Exception("Must init ReadPseudo")
        return self._gp

    @property
    def vp(self):
        if not self._vp:
            raise Exception("Must init ReadPseudo")
        return self._vp
