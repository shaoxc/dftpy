import numpy as np
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.interface import ConfigParser, OptimizeDensityConf
from dftpy.ions import Ions


class DFTpyCalculator(object):
    """DFTpy calculator for ase"""

    def __init__(self, config=None, mp = None, **kwargs):
        self.config = config
        self.results = None
        self.atoms = {}
        self.mp = mp

    def check_restart(self, atoms=None):
        if (
            self.atoms
            and np.allclose(self.atoms["lattice"], atoms.cell)
            and np.allclose(self.atoms["positions"], atoms.positions)
            and self.results is not None
        ):
            return False
        else:
            return True

    def get_potential_energy(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            lattice = atoms.cell
            pos = atoms.positions
            if self.results is not None and len(self.atoms) > 0 :
                pseudo = self.results["pseudo"]
                if np.allclose(self.atoms["lattice"], atoms.cell[:]):
                    grid = self.results["density"].grid
                else :
                    grid = None
            else :
                pseudo = None
                grid = None

            # Save the information of structure
            self.atoms["lattice"] = lattice.copy()
            self.atoms["positions"] = pos.copy()
            #
            ions = Ions.from_ase(atoms)
            #
            if self.results is not None and self.config["MATH"]["reuse"]:
                config, others = ConfigParser(self.config, ions=ions, rhoini=self.results["density"], pseudo=pseudo, grid=grid, mp = self.mp)
                results = OptimizeDensityConf(config, others["ions"], others["field"], others["E_v_Evaluator"], others["nr2"])
            else:
                config, others = ConfigParser(self.config, ions=ions, pseudo=pseudo, grid=grid, mp = self.mp)
                results = OptimizeDensityConf(config, others["ions"], others["field"], others["E_v_Evaluator"], others["nr2"])
            self.results = results
        energy = self.results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        energy = self.results["density"].grid.mp.asum(energy)
        return energy

    def get_forces(self, atoms):
        if self.check_restart(atoms):
            # if 'Force' not in self.config['JOB']['calctype'] :
                # self.config['JOB']['calctype'] += ' Force'
            self.get_potential_energy(atoms)
        return self.results["forces"]["TOTAL"] * FORCE_CONV["Ha/Bohr"]["eV/A"]

    def get_stress(self, atoms):
        if self.check_restart(atoms):
            # if 'Stress' not in self.config['JOB']['calctype'] :
                # self.config['JOB']['calctype'] += ' Stress'
            self.get_potential_energy(atoms)
        # return self.results['stress']['TOTAL'] * STRESS_CONV['Ha/Bohr3']['eV/A3']
        stress_voigt = np.zeros(6)
        if "TOTAL" not in self.results["stress"]:
            # print("!WARN : NOT calculate the stress, so return zeros")
            return stress_voigt
        for i in range(3):
            stress_voigt[i] = self.results["stress"]["TOTAL"][i, i]
        stress_voigt[3] = self.results["stress"]["TOTAL"][1, 2]  # yz
        stress_voigt[4] = self.results["stress"]["TOTAL"][0, 2]  # xz
        stress_voigt[5] = self.results["stress"]["TOTAL"][0, 1]  # xy
        # stress_voigt  *= -1.0
        return stress_voigt * STRESS_CONV["Ha/Bohr3"]["eV/A3"]
