import numpy as np
from dftpy.mpi import sprint
from dftpy.constants import ENERGY_CONV, FORCE_CONV, STRESS_CONV, TIME_CONV
from dftpy.interface import ConfigParser, OptimizeDensityConf, evaluator2results
from dftpy.ions import Ions
from dftpy.optimization import Optimization
from dftpy.field import DirectField
from dftpy.utils import field2distrib
from ase.calculators.calculator import Calculator, all_changes
from ase.md.verlet import VelocityVerlet
from dftpy.td.real_time_runner import RealTimeRunner
from dftpy.td.propagator import Propagator
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.utils.utils import calc_rho, calc_j
from dftpy.td.utils import initial_kick
from dftpy.functional import KEDF, Functional, TotalFunctional


class DFTpyCalculator(Calculator):
    """DFTpy calculator for ase"""
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, config = None, mp = None, optimizer=None, evaluator=None,
            rho = None, zero_stress=False, task='scf', step=1.0,**kwargs):
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
            self.dt = step * TIME_CONV['fs']['au'] / 1000
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

#   def run_tddft(self, system_changes=all_changes, properties=['energy'], **kwargs):
#       sprint("tddft step")

#       if self.rt_runner is None:
#           config, others = self.parse_config(system_changes=system_changes, properties=properties, **kwargs)
#           self.rt_runner = RealTimeRunner( others["field"], config, others["E_v_Evaluator"]) 
#       rho = self.dftpy_results.get('density', None)
#      #grid = rho.grid

#       if self.dftpy_results is not None and len(self.atoms) > 0 : 
#           pseudo = self.dftpy_results.get('pseudo', None)
#           if 'cell' not in system_changes:
#               grid = rho.grid
#           else :
#               grid = None 
#       else :    
#           pseudo = None  
#           grid = None  

#       ions = Ions.from_ase(self.atoms)

#      #self.rt_runner.rho = self.dftpy_results.get('density', None)
#       self.rt_runner.ions = ions
#       results = self.rt_runner.step()
#       self.dftpy_results = results
#       if self.mp is None :
#           self.mp = self.dftpy_results["density"].grid.mp

    def run_tddft(self, system_changes=all_changes, properties=['energy'], **kwargs):
        sprint("tddft step")

        if self.rt_runner is None:
            config, others = self.parse_config(system_changes=system_changes, properties=properties, **kwargs)
            rho = self.dftpy_results.get('density', None)
            rho0 = self.optimizer.optimize_rho(guess_rho=rho)
            self.rt_runner = RealTimeRunner( rho0, config, others["E_v_Evaluator"])

#

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
        F_o =  self.evaluator.get_forces(rho,ions)
        M = ions.get_masses()[:, np.newaxis]
        x_o = ions.get_positions()

        p = ions.get_momenta()
        p += 0.5 * self.dt / 2 * F_o 
      # nf = np.array(np.shape(F_o))

      # a_o = np.zeros(nf)
      # for i in range(len(F_o)):
      #     a_o[i]= F_o[i] / M[i]


      # x_m = x_o + v_o * self.dt / 2 + 0.5 * a_o * self.dt / 2 * self.dt / 2
        ions.set_positions(x_o + self.dt * p / M / 2)
        p = (ions.get_positions() - x_o)* M / self.dt / 2
        ions.set_momenta(p,apply_constraint=False)
      # Fm = self.evaluator.get_forces(rho,ions)
      # am = np.zeros(np.array(np.shape(Fm)))
      # for i in range(len(Fm)):
      #     am[i] = Fm[i]/M[i]

      # v_m = v_o + am * self.dt/2
      # ions.set_velocities(v_m)

       #self.rt_runner.rho = self.dftpy_results.get('density', None)

        self.rt_runner.ions = ions
        results = self.rt_runner.step()
        self.dftpy_results = results
        #F_mm = results["forces"]["TOTAL"]

        #a_mm = np.zeros(np.array(np.shape(F_mm)))

       # for i in range(len(F_mm)):
         #   a_mm[i] = F_mm[i]/M[i]

        #x_n = x_m + v_m *self.dt/2 + 0.5 *a_mm * self.dt/2 * self.dt/2
       ##ions.set_positions(x_n)
       # Fn = totalfunctional.get_forces(rho,ions)
        self.dftpy_results["forces"]["TOTAL"] = 0.0  
        self.dftpy_results["forces"]["TOTAL"] = F_o # FORCE_CONV["Ha/Bohr"]["eV/A"]
       ##an = np.zeros(np.array(np.shape(Fn)))

       ##for i in range(len(Fn)):
       # #   an[i] = Fn[i]/M[i]

       ##vn = v_m + an*self.dt/2
       # ions.set_velocities(vn)


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
        
    def get_density(self):
        return self.self.dftpy_results.get('densityF',None)

class TDDFTpyCalculator(Calculator):        
    implemented_properties = ['energy', 'forces']
        
    def __init__(self, config = None, mp = None, optimizer=None, evaluator=None,
            rho = None, task='propagate', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.config = config
        self.mp = mp
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.dftpy_results = {'density': rho}
        self.rt_runner = None
        self.task = task


    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):

        Calculator.calculate(self, atoms, properties, system_changes) 
        ions = Ions.from_ase(self.atoms)
        rho_ini = self.dftpy_results.get('density', None)

        rho0 = self.optimizer.optimize_rho(guess_rho=rho_ini) 

        potential = self.evaluator(rho0, calcType=['V']).potential
        hamiltonian = Hamiltonian(v=potential)
        interval = 0.043 # time interval in a.u. Note this is a relatively large time step. In real calculations you typically want a smaller time step like 1e-1 or 1e-2.
        prop = Propagator(hamiltonian, interval, name='crank-nicholson')

        direction = 0 # 0, 1, 2 means x, y, z-direction, respectively
        k = 1.0e-3 # kick_strength in a.u.
        psi = initial_kick(k, direction, np.sqrt(rho0))
        psi, info = prop(psi)
        rho = calc_rho(psi)
    #j = calc_j(psi)
        potential = self.evaluator(rho, calcType=['V']).potential
        prop.hamiltonian.v = potential

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


    def calc_forces(self):
        evaluator = self.dftpy_results.get('evaluator', None)
        rho = self.dftpy_results.get('density', None)
        results = evaluator2results(evaluator, rho=rho, calcType=['F'])
        self.dftpy_results['forces'] = results.get('forces', None)        

class EherenfestCalculator(Calculator):

    implemented_properties = ['energy', 'forces']
        
    def __init__(self, config = None, mp = None, optimizer=None, evaluator=None,
            rho = None, interval=None, ptype=None, initial_kick=None, task='propagate', ke = None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.config = config
        self.mp = mp
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.interval = interval
        self.ptype=ptype
        self.dftpy_results = {'density': rho}
        self.rt_runner = None
        self.kick = initial_kick
        self.task = task
        self.ke = ke

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):

        Calculator.calculate(self, atoms, properties, system_changes) 
        ions = Ions.from_ase(self.atoms)
        sprint(ions)
        rho_ini = self.dftpy_results.get('density', None)


        rho0 = self.optimizer.optimize_rho(guess_rho=rho_ini)
        #evaluator = self.dftpy_results.get('evaluator', None)
        #results = evaluator2results(evaluator, rho=rho0, calcType=['F'])
        fbo = self.evaluator.get_forces(rho0,ions)
        sprint("Fbo:",fbo) 
        ebo = self.evaluator.Energy(rho0)     
        sprint("EBO:",ebo)

        self.ke.options.update({'y':0})
        direction = 0 # 0, 1, 2 means x, y, z-direction, respectively
         # kick_strength in a.u.
        psi = initial_kick(self.kick, direction, np.sqrt(rho_ini))
        j0 = calc_j(psi)
        potential = self.evaluator(rho_ini, current=j0, calcType=['V']).potential
        hamiltonian = Hamiltonian(v=potential)
       # interval = 43 # time interval in a.u. Note this is a relatively large time step. In real calculations you typically want a smaller time step like 1e-1 or 1e-2.
        prop = Propagator(hamiltonian,interval=self.interval,name=self.ptype)

        psi, info = prop(psi)
        rho = calc_rho(psi)
        j = calc_j(psi)
        potential = self.evaluator(rho,current=j, calcType=['V']).potential
        prop.hamiltonian.v = potential

        evaluator = self.dftpy_results.get('evaluator', None)
#       results = evaluator2results(evaluator, rho=rho, calcType=['F'])
        fef = self.evaluator.get_forces(rho,ions)
        sprint("FEF:",fef) 
        eef = self.evaluator.Energy(rho)
        sprint("EEF:",eef)
        self.dftpy_results["energy"] = eef-ebo
        self.results['energy'] = self.dftpy_results["energy"] * FORCE_CONV["Ha/Bohr"]["eV/A"]
        sprint("EEF-EBO:",eef - ebo)
        sprint("EEF-EBO (eV):",self.results['energy'])

       #if 'forces' in properties:
        self.dftpy_results["forces"] = fef-fbo
        sprint(fef-fbo)
        self.forces = self.dftpy_results["forces"] * FORCE_CONV["Ha/Bohr"]["eV/A"]
        self.results['forces'] = self.forces



