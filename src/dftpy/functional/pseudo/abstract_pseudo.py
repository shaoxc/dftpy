from abc import ABC, abstractmethod

class AbstractPseudo(ABC):
    @abstractmethod
    def __init__(self, fname, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def zval(self):
        pass

    @property
    @abstractmethod
    def radial_grid(self):
        pass

    @property
    @abstractmethod
    def local_potential(self):
        pass

    @property
    @abstractmethod
    def core_density_grid(self):
        pass

    @property
    @abstractmethod
    def core_density(self):
        pass

    @property
    @abstractmethod
    def atomic_density_grid(self):
        pass

    @property
    @abstractmethod
    def atomic_density(self):
        pass

    @property
    @abstractmethod
    def direct(self):
        pass

class BasePseudo(AbstractPseudo):
    def __init__(self, fname, direct = True, **kwargs):
        self.fname = fname
        #-----------------------------------------------------------------------
        self.r = None
        self.v = None
        self.info = {}
        self._zval = None
        self._direct = direct
        self._core_density = None
        self._core_density_grid = None
        self._atomic_density = None
        self._atomic_density_grid = None
        #-----------------------------------------------------------------------
        self.read(fname, **kwargs)

    def read(self, fname, *args, **kwargs):
        pass

    @property
    def zval(self):
        return self._zval

    @property
    def radial_grid(self):
        return self.r

    @property
    def local_potential(self):
        return self.v

    @property
    def core_density_grid(self):
        if self._core_density_grid is None :
            return self.radial_grid
        else :
            return self._core_density_grid

    @property
    def core_density(self):
        return self._core_density

    @property
    def atomic_density_grid(self):
        if self._atomic_density_grid is None :
            return self.radial_grid
        else :
            return self._atomic_density_grid

    @property
    def atomic_density(self):
        return self._atomic_density

    @property
    def direct(self):
        return self._direct
