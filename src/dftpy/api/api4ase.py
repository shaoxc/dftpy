import numpy as np
from dftpy.mpi import sprint
from dftpy.constants import ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.interface import ConfigParser, OptimizeDensityConf, evaluator2results
from dftpy.ions import Ions
from dftpy.optimization import Optimization
from dftpy.field import DirectField
from dftpy.utils import field2distrib
from ase.calculators.calculator import Calculator, all_changes
from dftpy.td.real_time_runner import RealTimeRunner
from dftpy.functional import KEDF


class DFTpyCalculator(Calculator):
    """DFTpy calculator for ase"""
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, config = None, mp = None, optimizer=None, evaluator=None,
            rho = None, zero_stress=False, task='scf', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.config = config
        self.mp = mp
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.zero_stress = zero_stress
        self.dftpy_results = {'density': rho}
        self.rt_runner = None
        self.task = task
        #
        if self.config is not None:
            if config["TD"]["single_step"]: self.task = 'propagate'

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if len(system_changes) > 0 :
            if self.task == 'propagate':
                self.run_tddft(system_changes, properties=properties)
            else:
                self.run(system_changes, properties=properties)

        energy = self.dftpy_results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        if self.task == 'propagate':
            kedf = KEDF(name="vw")
            kedff = self.mp.vsum(kedf(self.dftpy_results["density"]).energy * ENERGY_CONV["Hartree"]["eV"])
            energy += kedff
        #self.energy = self.mp.asum(energy)
        self.energy = energy
        self.results['energy'] = self.energy
 
        if 'forces' in properties:
            if "TOTAL" not in self.dftpy_results["forces"]: self.calc_forces()
            self.forces = self.dftpy_results["forces"]["TOTAL"] * FORCE_CONV["Ha/Bohr"]["eV/A"]
            self.results['forces'] = self.forces

        if 'stress' in properties:
            stress_voigt = np.zeros(6)
            if not self.zero_stress :
                if "TOTAL" not in self.dftpy_results["stress"]: self.calc_stress()
                for i in range(3):
                    stress_voigt[i] = self.dftpy_results["stress"]["TOTAL"][i, i]
                stress_voigt[3] = self.dftpy_results["stress"]["TOTAL"][1, 2]  # yz
                stress_voigt[4] = self.dftpy_results["stress"]["TOTAL"][0, 2]  # xz
                stress_voigt[5] = self.dftpy_results["stress"]["TOTAL"][0, 1]  # xy
            self.stress = stress_voigt * STRESS_CONV["Ha/Bohr3"]["eV/A3"]
            self.results['stress'] = self.stress

    def run(self, system_changes=all_changes, properties=['energy']):
        if self.config is not None:
            self.run_config(system_changes=system_changes, properties=properties)
        elif self.evaluator is not None:
            self.run_normal(system_changes=system_changes, properties=properties)
        else:
            raise ValueError("!!!Error: Please give a 'config' or 'evaluator' for the calculator")
        if self.mp is None :
            self.mp = self.dftpy_results["density"].grid.mp

    def parse_config(self, system_changes=all_changes, properties=['energy'], **kwargs):
        pseudo = self.dftpy_results.get('pseudo', None)
        rho = self.dftpy_results.get('density', None)
        ions = Ions.from_ase(self.atoms)

        grid = None
        if 'cell' not in system_changes:
            if rho is not None: grid = rho.grid

        if not self.config["MATH"]["reuse"]: rho = None

        return ConfigParser(self.config, ions=ions, rhoini=rho, pseudo=pseudo, grid=grid, mp = self.mp)

    def run_config(self, system_changes=all_changes, properties=['energy'], **kwargs):
        config, others = self.parse_config(system_changes=system_changes, properties=properties, **kwargs)
        self.dftpy_results = OptimizeDensityConf(config, **others)

    def run_normal(self, system_changes=all_changes, properties=['energy'], **kwargs):
        rho = self.dftpy_results.get('density', None)
        ions = Ions.from_ase(self.atoms)

        if rho is None:
            raise ValueError("!!!Error: Please give a initial density for the calculator")

        grid = rho.grid
        if 'cell' in system_changes:
            grid = grid.create(lattice = ions.cell)
            rho_old = rho
            rho = DirectField(grid=grid, rank=rho_old.rank, cplx=rho_old.cplx)

        for key, func in self.evaluator.funcDict.items():
            if func.type == "PSEUDO":
                func.restart(ions=ions, grid=grid)

        if 'cell' in system_changes:
            field2distrib(rho_old, rho)
            rho *= ions.get_ncharges() / np.sum(rho.integral())

        if self.optimizer is None:
            self.optimizer = Optimization(EnergyEvaluator=self.evaluator)
        rho = self.optimizer.optimize_rho(guess_rho=rho)

        calcType = ['E']
        # if 'energy' in properties: calcType.append('E')
        if 'forces' in properties: calcType.append('F')
        if 'potential' in properties: calcType.append('V')
        if 'stress' in properties: calcType.append('S')
        if self.zero_stress:
            if 'S' in calcType : calcType.remove('S')

        self.dftpy_results = evaluator2results(self.evaluator, rho=rho, calcType=calcType, ions=ions, **kwargs)

    def run_tddft(self, system_changes=all_changes, properties=['energy'], **kwargs):
        if self.rt_runner is None:
            config, others = self.parse_config(system_changes=system_changes, properties=properties, **kwargs)
            self.rt_runner = RealTimeRunner(others["field"], config, others["E_v_Evaluator"]) 
        sprint("tddft step")
        rho = self.dftpy_results.get('density', None)

        if self.dftpy_results is not None and len(self.atoms) > 0 : 
            pseudo = self.dftpy_results.get('pseudo', None)
            if 'cell' not in system_changes:
                grid = rho.grid
            else :
                grid = None 
        else :    
            pseudo = None  
            grid = None  

        ions = Ions.from_ase(self.atoms)

        self.rt_runner.ions = ions
        results = self.rt_runner.step()
        self.dftpy_results = results
        if self.mp is None :
            self.mp = self.dftpy_results["density"].grid.mp


    def calc_forces(self):
        evaluator = self.dftpy_results.get('evaluator', None)
        rho = self.dftpy_results.get('density', None)
        results = evaluator2results(evaluator, rho=rho, calcType=['F'])
        self.dftpy_results['forces'] = results.get('forces', None)

    def calc_stress(self):
        evaluator = self.dftpy_results.get('evaluator', None)
        rho = self.dftpy_results.get('density', None)
        results = evaluator2results(evaluator, rho=rho, calcType=['S'])
        self.dftpy_results['stress'] = results.get('stress', None)
