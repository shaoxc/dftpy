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
        if self._cell is not None and self.field is not None and not isinstance(self.field, BaseField):
            nr = np.shape(self.field)
            grid = DirectGrid(self._cell.lattice, nr)
            self.field = DirectField(grid, griddata_3d=self.field)

    @property
    def natoms(self):
        return np.shape(self.ions.pos)[0]
