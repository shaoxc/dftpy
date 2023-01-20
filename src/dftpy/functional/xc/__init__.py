import copy
from dftpy.functional.abstract_functional import AbstractFunctional

from .semilocal_xc import LDA, PBE, LibXC, CheckLibXC, XCStress
from .rvv10 import RVV10, RVV10NL

class XC(AbstractFunctional):
    def __init__(self, xc=None, core_density=None, libxc=None, name=None, pseudo = None, **kwargs):
        self.type = 'XC'
        self.name = name or 'XC'
        self.energy = None
        xc = xc or name
        self.options = kwargs
        self.options['xc'] = xc
        self._core_density = core_density
        self.pseudo = pseudo # For NLCC
        if libxc is False :
            if xc and xc.lower() == 'lda' :
                self.xcfun = LDA
            else :
                raise AttributeError("Default one only support 'LDA', others please try with pylibxc.")
        else :
            if isinstance(libxc, bool) : libxc = None
            self.options['libxc'] = libxc
            if CheckLibXC(False):
                if xc and xc.lower() == 'rvv10' :
                    self.xcfun = RVV10(**kwargs)
                else :
                    self.xcfun = LibXC
            elif xc and xc.lower() == 'lda':
                self.xcfun = LDA
            else :
                raise ModuleNotFoundError("Install pylibxc to use this functionality")

    @property
    def core_density(self):
        if self._core_density is None and self.pseudo is not None :
            self._core_density = self.pseudo.core_density
        return self._core_density

    @core_density.setter
    def core_density(self, value):
        self._core_density = value

    def compute(self, density, calcType={"E", "V"}, **kwargs):
        options = copy.deepcopy(self.options)
        options.update(kwargs)
        core_density = self.core_density
        if core_density is None:
            new_density = density
        elif density.rank == core_density.rank:
            new_density = density + core_density
        elif density.rank == 2 and core_density.rank == 1:
            new_density = density.copy()
            new_density[0] += 0.5 * core_density
            new_density[1] += 0.5 * core_density

        functional = self.xcfun(new_density, calcType=calcType, **options)
        if 'E' in calcType : self.energy = functional.energy
        return functional

    def forces(self, density, pseudo = None, **kwargs):
        if pseudo is None : pseudo = self.pseudo
        if pseudo is None : return None
        pot = self.compute(density, calcType={"V"}, **kwargs).potential
        forces = pseudo.calc_force_cc(pot)
        return forces

    def stress(self, density, **kwargs):
        options = copy.deepcopy(self.options)
        options.update(kwargs)
        energy = self.energy
        stress=XCStress(density, energy=energy, **options)
        return stress
