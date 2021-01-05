import numpy as np


class System(object):
    def __init__(self, ions, cell = None, name=None, field=None):
        self.ions = ions
        self._cell = cell
        self.name = name
        self.field = field

    @property
    def cell(self):
        if self._cell is not None :
            return self._cell
        else :
            return self.ions.pos.cell

    @property
    def natoms(self):
        return np.shape(self.ions.pos)[0]
