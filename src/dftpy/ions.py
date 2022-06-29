import numpy as np
from ase import Atoms
from dftpy.constants import Units

class Ions(Atoms):
    """Ions object based on ase.Atoms
    Units :
        positions : Bohr
        cell : Bohr
    """
    def __init__(self,
            symbols=None,
            positions=None,
            numbers=None,
            tags=None,
            momenta=None,
            masses=None,
            magmoms=None,
            charges=None,
            scaled_positions=None,
            cell=None,
            pbc=None,
            celldisp=None,
            constraint=None,
            calculator=None,
            info=None,
            velocities=None,
            units = 'au'):

        super().__init__(
            symbols=symbols,
            positions=positions,
            numbers=numbers,
            tags=tags,
            momenta=momenta,
            masses=masses,
            magmoms=magmoms,
            charges=charges,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=pbc,
            celldisp=celldisp,
            constraint=constraint,
            calculator=calculator,
            info=info,
            velocities=velocities)

        if units != 'au' :
            self.convert_units()

    def to_ase(self):
        atoms = self.copy()
        atoms.convert_units(backward=True)
        return atoms

    @staticmethod
    def from_ase(atoms):
        ions = Ions(atoms, units = 'ase')
        return ions

    def convert_units(self, backward = False):
        """"Convert the units to atomic units or ase units."""
        if not backward :
            self.cell.array[:] /= Units.Bohr
            if self.has('positions'):
                self.positions[:] /= Units.Bohr
            self._celldisp /= Units.Bohr
            # if self.has('momenta'):
                # self.arrays['momenta'] /= Units._auv
        else :
            self.cell.array[:] *= Units.Bohr
            if self.has('positions'):
                self.positions[:] *= Units.Bohr
            self._celldisp *= Units.Bohr
            # if self.has('momenta'):
                # self.arrays['momenta'] *= Units._auv

    def get_ncharges(self):
        """Get total number of charges."""
        if not self.has('initial_charges'):
            raise Exception("Please initialize or set the 'charge'")
        return self.arrays['initial_charges'].sum()

    def get_charges(self, charges=None):
        """Get the atomic charges."""
        self.get_initial_charges(charges=charges)

    def set_charges(self, charges=None):
        """Set the atomic charges."""
        if isinstance(charges, dict):
            values = []
            for s in self.symbols :
                if s not in charges :
                    raise AttributeError(f"{s} not in the charges")
                values.append(charges[s])
            charges = values
        self.set_initial_charges(charges=charges)

    @property
    def charges(self):
        """Get the atomic charges."""
        return self.arrays['initial_charges']

    @charges.setter
    def charges(self, value):
        """Set the atomic charges."""
        self.arrays['initial_charges'][:] = value

    def strf(self, reciprocal_grid, iatom):
        """Returns the Structure Factor associated to i-th ion."""
        a = np.exp(-1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.positions[iatom]))
        return a

    def istrf(self, reciprocal_grid, iatom):
        """Returns the Structure-Factor-like property associated to i-th ion."""
        a = np.exp(1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.positions[iatom]))
        return a

    @property
    def symbols_uniq(self):
        return sorted(np.unique(self.symbols))

    @property
    def nat(self):
        return len(self)

    @property
    def zval(self):
        zval = dict.fromkeys(self.symbols_uniq, 0)
        symbols = self.get_chemical_symbols()
        try:
            self.charges[0]
        except Exception :
            return zval

        for k in zval :
            for i in range(self.nat):
                if symbols[i] == k :
                    zval[k] = self.charges[i]
                    break
        return zval