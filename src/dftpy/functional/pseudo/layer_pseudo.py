from dftpy.functional.external_potential import ExternalPotential
from dftpy.atom import Atom
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from typing import List
import numpy as np
from dftpy.math_utils import PowerInt


class LayerPseudo(ExternalPotential):

    def __init__(self, vr=None, r=None, grid: DirectGrid = None, ions: List[Atom] = None, **kwargs):
        super().__init__()
        self.grid = grid
        self.ions = ions
        self.vr = vr
        self.r = r
        if r is not None:
            self.r_cut = r[-1]
        self.dis = None
        self.distance()

    def distance(self):
        self.dis = DirectField(self.grid, griddata_3d=np.zeros(self.grid.nr))
        for ion in self.ions:
            dis_ion = np.sqrt(
                PowerInt((self.dis.grid.r[0] - ion.pos[0]), 2) +
                PowerInt((self.dis.grid.r[1] - ion.pos[1]), 2) +
                PowerInt((self.dis.grid.r[2] - ion.pos[2]), 2))
            self.dis[dis_ion <= self.r_cut] = dis_ion[dis_ion <= self.r_cut]

    @property
    def v(self):
        if self._v is not None:
            return self._v

        from scipy.interpolate import splrep, splev
        spl = splrep(self.r, self.vr)
        self._v = splev(self.dis, spl)
        self._v = DirectField(self.grid, griddata_3d=np.zeros(self._v))
        return self._v


    # def __init__(self, **kwargs):
    #     self._gp = {}  # 1D PP grid g-space
    #     self._vp = {}  # PP on 1D PP grid
    #     self._vloc_interp = {}  # Interpolates recpot PP
    #     self._zval = {}
    #
    # @property
    # def vloc_interp(self):
    #     if not self._vloc_interp:
    #         """get the representation of PP
    #
    #                 Args:
    #                     key: Atomic symbol
    #                     k: The degree of the spline fit of splrep, should keep use 3.
    #                 """
    #         from scipy.interpolate import splrep
    #         vloc_interp = splrep(self._gp[key][1:], self._vp[key][1:], k=k)
    #         self._vloc_interp[key] = vloc_interp
    #     return self._vloc_interp
    #
    # @property
    # def gp(self):
    #     if not self._gp:
    #         raise AttributeError("Must init ReadPseudo")
    #     return self._gp
    #
    # @property
    # def vp(self):
    #     if not self._vp:
    #         raise AttributeError("Must init ReadPseudo")
    #     return self._vp
    #
    # def get_Zval(self, ions):
    #     pass
