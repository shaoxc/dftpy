import os
import os.path

import numpy as np

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField
from dftpy.formats import npy
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.mpi import mp, sprint, MPIFile
from dftpy.optimize import Dynamics
from dftpy.system import System
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.predictor_corrector import PredictorCorrector
from dftpy.td.propagator import Propagator
from dftpy.utils.utils import calc_rho, calc_j
from dftpy.time_data import TimeData, timer


class RealTimeRunner(Dynamics):

    def __init__(self, system, config, functionals):
        self.outfile = config["TD"]["outfile"]
        Dynamics.__init__(self, system, self.outfile)
        self.int_t = config["TD"]["timestep"]
        self.t_max = config["TD"]["tmax"]
        self.max_pred_corr = config["TD"]["max_pc"]
        self.tol = config["TD"]["tol_pc"]
        self.atol = config["TD"]["atol_pc"]
        self.direc = config["TD"]["direc"]
        self.k = config["TD"]["strength"]
        self.max_runtime = config["TD"]["max_runtime"]
        self.restart = config["TD"]["restart"]
        self.restart_input = config["TD"]["restart_input"]
        self.save_interval = config["TD"]["save_interval"]
        self.correction = config["TD"]["correction"]
        self.vector_potential = config["TD"]['vector_potential']
        if self.vector_potential:
            self.propagate_vector_potential = config["TD"]['propagate_vector_potential']
        else:
            self.propagate_vector_potential = False
        self.max_steps = int(self.t_max / self.int_t)
        self.N0 = self.system.field.integral()
        self.read_kpoint(fname='./test113.npy')
        self.k_point_list = self.system.field.grid.get_reciprocal().calc_k_points(self.nk)
        #self.rho0 = self.system.field
        #self.psi_list[0] = np.sqrt(self.rho0)
        #self.psi_list[0] = self.psi_list[0].normalize()
        self.rho0 = calc_rho(self.psi_list, N=self.N0, nnk=self.nnk)
        self.functionals = functionals
        self.predictor_corrector = None
        self.delta_mu = None
        self.j_int = None
        self.E = None
        self.correct_potential = None
        self.timer = None



        if self.vector_potential:
            self.A_t = None
            self.A_tm1 = None
            self.Omega = config["TD"]["omega"]
            if self.Omega <= 0:
                self.Omega = self.system.field.grid.Volume
            else:
                self.Omega = self.system.field.grid.Volume * self.Omega

        if self.correction:
            correct_potential_dict = dict()
            if config['KEDF2']['active']:
                correct_kedf = Functional(type='KEDF', name=config['KEDF2']['kedf'], **config['KEDF2'])
                correct_potential_dict.update({'Nonlocal': correct_kedf})
            if config['NONADIABATIC2']['active']:
                correct_dynamic = Functional(type='DYNAMIC', name=config['NONADIABATIC2']['nonadiabatic'],
                                             **config['NONADIABATIC2'])
                correct_potential_dict.update({'Dynamic': correct_dynamic})
            self.correct_functionals = TotalFunctional(**correct_potential_dict)

        self.attach(self.calc_obeservables, before_log=True)
        if self.max_runtime > 0:
            self.attach(self.safe_quit)
        self.attach(self.save, interval=self.save_interval)
        if self.max_steps % self.save_interval != 0:
            self.attach(self.save, interval=-self.max_steps)
        #self.attach(self.debug)

        hamiltonian = Hamiltonian()
        self.propagator = Propagator(hamiltonian, self.int_t, name = config["PROPAGATOR"]["propagator"], **config["PROPAGATOR"])

        if self.restart:
            self.load()
        else:
            self.apply_initial_field()

        self.rho = calc_rho(self.psi_list, nnk=self.nnk)
        self.j = calc_j(self.psi_list, nnk=self.nnk)
        self.update_hamiltonian()

    def initialize(self):
        sprint("{:20s}{:30s}{:24s}".format('Iter', 'Num. of Predictor-corrector', 'Total Cost(s)'))

    def step(self):
        system = System(cell=self.system.cell)
        self.predictor_corrector = PredictorCorrector(system, **self.predictor_corrector_arguments())
        converged = self.predictor_corrector.run()
        if not converged:
            self.predictor_corrector.print_not_converged_info()
        self.update_hamiltonian()
        # if self.correction:
        #     self.predictor_corrector.psi_pred = self.predictor_corrector.psi_pred - 1.0j * self.int_t * self.correct_potential * self.psi
        #     self.predictor_corrector.psi_pred.normalize(N=self.N0)
        #     self.predictor_corrector.rho_pred = calc_rho(self.predictor_corrector.psi_pred)
        #     self.predictor_corrector.j_pred = calc_j(self.predictor_corrector.psi_pred)

        self.psi_list = self.predictor_corrector.psi_pred_list
        self.rho = self.predictor_corrector.rho_pred
        self.j = self.predictor_corrector.j_pred
        if self.propagate_vector_potential:
            self.A_tm1 = self.A_t
            self.A_t = self.predictor_corrector.A_t_pred

        self.timer = TimeData.Time('Real-time propagation')
        sprint("{:<20d}{:<30d}{:<24.4f}".format(self.nsteps + 1, self.predictor_corrector.nsteps, self.timer))

    @timer('Real-time propagation')
    def run(self):
        return Dynamics.run(self)

    def log(self):
        if mp.is_root:
            if self.nsteps == 0:
                real_time_runner_print_title(self.logfile, self.vector_potential)
            real_time_runner_print_data(self.logfile, self.nsteps * self.int_t, self.E, self.delta_mu, self.j_int,
                                        self.A_t if self.vector_potential else None)

    def predictor_corrector_arguments(self):
        arguments = {
            'propagator': self.propagator,
            'tol': self.tol,
            'atol': self.atol,
            'max_steps': self.max_pred_corr,
            'propagate_vector_potential': self.propagate_vector_potential,
            'int_t': self.int_t,
            'functionals': self.functionals,
            'nk': self.nk,
            'psi_list': self.psi_list,
            'N0': self.N0
        }
        if self.propagate_vector_potential:
            arguments.update({
                'Omega': self.Omega,
                'A_t': self.A_t,
                'A_tm1': self.A_tm1
            })
        return arguments

    def apply_initial_field(self):
        if self.vector_potential:
            self.A_t = np.zeros(3)
            self.A_t[self.direc] = self.k * SPEED_OF_LIGHT
            self.A_tm1 = self.A_t.copy()
            for psi in self.psi_list:
                psi.cplx = True
        else:
            x = self.system.field.grid.r[self.direc]
            for i in range(len(self.psi_list)):
                self.psi_list[i] = self.psi_list[i] * np.exp(1j * self.k * x)
                self.psi_list[i].cplx = True



    def update_hamiltonian(self):
        func = self.functionals(self.rho, calcType=["V"], current=self.j)
        self.propagator.hamiltonian.potential = func.potential
        if self.vector_potential:
            self.propagator.hamiltonian.vector_potential = self.A_t
        if self.correction:
            self.correct_potential = self.correct_functionals(self.rho, calcType=['V'], current=self.j).potential

    def calc_obeservables(self):
        delta_rho = self.rho - self.rho0
        self.delta_mu = (delta_rho * delta_rho.grid.r).integral()
        self.j_int = self.j.integral()
        self.calc_energy()

    def calc_energy(self):
        self.E = 0
        for i_k, k_point in enumerate(self.k_point_list):
            if self.correction:
                self.E = np.real(np.conj(self.psi_list[i_k]) * (self.propagator.hamiltonian(self.psi_list[i_k], k_point=k_point) + self.correct_potential * self.psi_list[i_k])).integral() / self.psi_list[i_k].norm() ** 2.0 * self.N0
            else:
                self.E = np.real(np.conj(self.psi_list[i_k]) * (self.propagator.hamiltonian(self.psi_list[i_k], k_point=k_point))).integral() / self.psi_list[i_k].norm() ** 2.0 * self.N0
        self.E /= self.nnk
        if self.propagate_vector_potential:
            self.E += self.Omega / 8.0 / np.pi / SPEED_OF_LIGHT ** 2 * (
                    np.dot((self.A_t - self.A_tm1), (self.A_t - self.A_tm1)) / self.int_t / self.int_t)

    def load(self):
        fname = ''.join(['./tmp/', self.restart_input])
        if mp.size > 1:
            f = MPIFile(fname, mp, amode=mp.MPI.MODE_RDONLY)
        else:
            f = open(fname, "rb")
        self.nsteps = npy.read(f, single=True) + 1
        self.nk = npy.read(f, single=True)
        self.nnk = self.nk[0] * self.nk[1] * self.nk[2]
        self.psi_list = [None] * self.nnk
        for i_psi in range(len(self.psi_list)):
            psi = npy.read(f, grid=self.system.field.grid)
            self.psi_list[i_psi] = DirectField(grid=self.system.field.grid, rank=1, griddata_3d=psi, cplx=True)
        if self.vector_potential:
            self.A_t = npy.read(f, single=True)
            self.A_tm1 = npy.read(f, single=True)
        f.close()

    def save(self):
        sprint('Save wavefunction data.')
        if not os.path.isdir('./tmp') and mp.is_root:
            os.mkdir('./tmp')
        fname = './tmp/{0:s}_{1:d}.npy'.format(self.outfile, self.nsteps)
        if mp.size > 1:
            f = MPIFile(fname, mp, amode=mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
        else:
            f = open(fname, "wb")
        if mp.is_root:
            npy.write(f, self.nsteps - 1, single=True)
            npy.write(f, self.nk, single=True)
        for psi in self.psi_list:
            npy.write(f, psi, grid=psi.grid)
        if self.vector_potential and mp.is_root:
            npy.write(f, self.A_t, single=True)
            npy.write(f, self.A_tm1, single=True)
        f.close()

    def safe_quit(self):
        self.timer = TimeData.Time('Real-time propagation')
        if self.timer > self.max_runtime:
            self.save()
            sprint('Maximum run time reached. Clean exiting.')
            self.stop_generator = True

    def read_kpoint(self, fname = './test.npy'):
        if mp.size > 1:
            f = MPIFile(fname, mp, amode=mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
        else:
            f = open(fname, "rb")
        if mp.is_root:
            self.nk = npy.read(f, single=True)
        self.nnk = self.nk[0] * self.nk[1] * self.nk[2]
        self.psi_list = [None] * self.nnk
        for i_psi in range(len(self.psi_list)):
            psi = npy.read(f, grid=self.system.field.grid)
            self.psi_list[i_psi] = DirectField(grid=self.system.field.grid, rank=1, griddata_3d=psi, cplx=True)

        f.close()

    def debug(self):
        #from dftpy.formats.xsf import XSF
        #i_k = 4
        #xsf = XSF(filexsf='./jy_{0:d}_{1:d}.xsf'.format(i_k, self.nsteps))
        #xsf.write(self.system, field=self.psi_list[i_k])
        #j = calc_j(self.psi_list[i_k])
        #sprint(np.sum(j[1]))
        #j_y = j[1]
        #j_z_inv = np.flip(j_z, axis=2)
        #xsf.write(self.system, field=j[1])
        #sprint(self.k_point_list[i_k], ':\n', self.psi_list[i_k])
        psi = self.psi_list[0]
        print(psi[0,0,1]-psi[0,1,0])
        for i_k, k_point in enumerate(self.k_point_list):
             j = calc_j(self.psi_list[i_k]).integral()
             sprint(k_point, ": ", j)

        sprint(self.j_int)



def real_time_runner_print_title(fileobj, vector_potential=False):
    sprint("{0:^17s} {1:^17s} {2:^17s} {3:^17s} {4:^17s} {5:^17s} {6:^17s} {7:^17s}".format('t', 'E', 'mu_x', 'mu_y',
                                                                                            'mu_z',
                                                                                            'j_x', 'j_y', 'j_z'),
           end='',
           fileobj=fileobj)
    if vector_potential:
        sprint(' {0:^17s} {1:^17s} {2:^17s}'.format('A_x', 'A_y', 'A_z'), fileobj=fileobj)
    else:
        sprint('', fileobj=fileobj)


def real_time_runner_print_data(fileobj, t, E, mu, j, A=None):
    sprint("{0:17.10e}".format(t), end='', fileobj=fileobj)
    sprint(" {0:17.10e}".format(E), end='', fileobj=fileobj)
    sprint(" {0:17.10e} {1:17.10e} {2:17.10e}".format(mu[0], mu[1], mu[2]), end='', fileobj=fileobj)
    sprint(" {0:17.10e} {1:17.10e} {2:17.10e}".format(j[0], j[1], j[2]), end='',
           fileobj=fileobj)
    if A is not None:
        sprint(" {0:17.10e} {1:17.10e} {2:17.10e}".format(A[0], A[1], A[2]), fileobj=fileobj)
    else:
        sprint('', fileobj=fileobj)
