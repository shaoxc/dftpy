import numpy as np

from dftpy.field import BaseField, DirectField
from dftpy.grid import DirectGrid

class System(object):
    def __init__(self, ions=None, cell=None, name=None, field=None):
        self.ions = ions
        self.field = field
        self.cell = cell
        self.name = name

    @property
    def cell(self):
        if self._cell is not None:
            return self._cell
        elif self.ions is not None:
            return self.ions.pos.cell
        elif self.field is not None and isinstance(self.field, BaseField):
            return self.field.grid
        else:
            return None

    @cell.setter
    def cell(self, new_cell):
        self._cell = new_cell
        if self._cell is not None and not isinstance(self.field, BaseField):
            nr = np.shape(self.field)
            grid = DirectGrid(self._cell.lattice, nr)
            self.field = DirectField(grid, griddata_3d=self.field)

    @property
    def natoms(self):
        return np.shape(self.ions.pos)[0]

    def copy(self):
        kws = {}
        for key in ['ions', 'cell', 'field'] :
            value = getattr(self, key, None)
            if value is not None : value = value.copy()
            kws[key] = value
        system = self.__class__(**kws)
        return system

    def __add__(self, other):
        system = self.copy()
        system += other
        return system

    def __iadd__(self, other):
        if other.ions is not None :
            if self.ions is not None :
                self.ions = self.ions + other.ions
            else :
                self.ions = other.ions.copy()

        if other.field is not None :
            if self.field is not None :
                self.field+= other.field
            else :
                self.field = other.field.copy()
        return self
