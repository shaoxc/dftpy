import numpy as np
from dftpy.atom import Atom
from dftpy.base import BaseCell, DirectCell
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.interface import OptimizeDensityConf

class DFTpyCalculator(object):
    """DFTpy calculator for ase"""
    def __init__(self, config = None):
        self.config = config
        self.results = None
        self.atoms = {}

    def check_restart(self, atoms=None):
        if self.atoms and np.allclose(self.atoms['lattice'], atoms.cell[:]) and \
            np.allclose(self.atoms['position'], atoms.get_scaled_positions()) and self.results is not None:
            return False
        else :
            return True

    def get_potential_energy(self, atoms = None,  **kwargs):
        if self.check_restart(atoms):
            lattice = atoms.cell[:]
            Z = atoms.numbers
            # pos = atoms.get_positions()
            # pos /= LEN_CONV['Bohr']['Angstrom']
            pos = atoms.get_scaled_positions()
            self.atoms['lattice'] = lattice.copy()
            self.atoms['position'] = pos.copy()
            lattice = np.asarray(lattice).T/ LEN_CONV['Bohr']['Angstrom']
            cell = DirectCell(lattice)
            ions = Atom(Z = Z, pos=pos, cell=cell, basis = 'Crystal')
            ions.restart()
            if self.results is not None and self.config['MATH']['reuse'] :
                results = OptimizeDensityConf(self.config, ions = ions, rhoini = self.results['density'])
            else :
                results = OptimizeDensityConf(self.config, ions = ions)
            self.results = results
        return self.results['energypotential']['TOTAL'].energy * ENERGY_CONV['Hartree']['eV']

    def get_forces(self, atoms):
        if self.check_restart(atoms):
            self.get_potential_energy(atoms)
        return self.results['forces']['TOTAL'] * FORCE_CONV['Ha/Bohr']['eV/A']

    def get_stress(self, atoms):
        if self.check_restart(atoms):
            self.get_potential_energy(atoms)
        # return self.results['stress']['TOTAL'] * STRESS_CONV['Ha/Bohr3']['eV/A3']
        stress_voigt = np.zeros(6)
        if 'TOTAL' not in self.results['stress'] :
            print('!WARN : NOT calculate the stress, so return zeros')
            return stress_voigt
        for i in range(3):
            stress_voigt[i] = self.results['stress']['TOTAL'][i, i]
        stress_voigt[3] = self.results['stress']['TOTAL'][1, 2] #yz
        stress_voigt[4] = self.results['stress']['TOTAL'][0, 2] #xz
        stress_voigt[5] = self.results['stress']['TOTAL'][0, 1] #xy
        # stress_voigt  *= -1.0
        return stress_voigt * STRESS_CONV['Ha/Bohr3']['eV/A3']
