import numpy as np
from dftpy.constants import ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.interface import ConfigParser, OptimizeDensityConf
from dftpy.ions import Ions
from ase.calculators.calculator import Calculator, all_changes

class DFTpyCalculator(Calculator):
    """DFTpy calculator for ase"""
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, config = None, mp = None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.config = config
        self.mp = mp
        self.dftpy_results = {}

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if len(system_changes) > 0 :
            self.run(system_changes)

        energy = self.dftpy_results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        self.energy = self.dftpy_results["density"].grid.mp.asum(energy)
        self.results['energy'] = self.energy
        if 'forces' in properties:
            self.forces = self.dftpy_results["forces"]["TOTAL"] * FORCE_CONV["Ha/Bohr"]["eV/A"]
            self.results['forces'] = self.forces

        if 'stress' in properties:
            stress_voigt = np.zeros(6)
            if "TOTAL" in self.dftpy_results["stress"]:
                for i in range(3):
                    stress_voigt[i] = self.dftpy_results["stress"]["TOTAL"][i, i]
                    stress_voigt[3] = self.dftpy_results["stress"]["TOTAL"][1, 2]  # yz
                    stress_voigt[4] = self.dftpy_results["stress"]["TOTAL"][0, 2]  # xz
                    stress_voigt[5] = self.dftpy_results["stress"]["TOTAL"][0, 1]  # xy
            else:
                self.mp.sprint("!WARN : NOT calculate the stress, so return zeros")
            self.stress = stress_voigt * STRESS_CONV["Ha/Bohr3"]["eV/A3"]
            self.results['stress'] = self.stress

    def run(self, system_changes=all_changes):
        pseudo = self.dftpy_results.get('pseudo', None)
        rho = self.dftpy_results.get('density', None)

        grid = None
        if 'cell' not in system_changes:
            if rho is not None: grid = rho.grid
        if not self.config["MATH"]["reuse"]:
            rho = None

        ions = Ions.from_ase(self.atoms)

        config, others = ConfigParser(self.config, ions=ions, rhoini=rho, pseudo=pseudo, grid=grid, mp = self.mp)
        self.dftpy_results = OptimizeDensityConf(config, **others)
        if self.mp is None :
            self.mp = self.dftpy_results["density"].grid.mp
