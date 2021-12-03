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
        self.functionals = functionals
        self.N0 = self.system.field.integral()
        self.psi = None
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

        hamiltonian = Hamiltonian()
        self.propagator = Propagator(hamiltonian, self.int_t, name=config["PROPAGATOR"]["propagator"], **config["PROPAGATOR"])

        if self.restart:
            self.load()
        else:
            self.apply_initial_field()

        self.rho = calc_rho(self.psi)
        self.j = calc_j(self.psi)
        self.update_hamiltonian()

    def initialize(self):
        sprint("{:20s}{:30s}{:24s}".format('Iter', 'Num. of Predictor-corrector', 'Total Cost(s)'))

    def step(self):
        system = System(field=self.psi)
        self.predictor_corrector = PredictorCorrector(system, **self.predictor_corrector_arguments())
        converged = self.predictor_corrector.run()
        if not converged:
            self.predictor_corrector.print_not_converged_info()
        self.update_hamiltonian()
        if self.correction:
            self.predictor_corrector.psi_pred = self.predictor_corrector.psi_pred - 1.0j * self.int_t * self.correct_potential * self.psi
            self.predictor_corrector.psi_pred.normalize(N=self.N0)
            self.predictor_corrector.rho_pred = calc_rho(self.predictor_corrector.psi_pred)
            self.predictor_corrector.j_pred = calc_j(self.predictor_corrector.psi_pred)

        self.psi = self.predictor_corrector.psi_pred
        self.rho = self.predictor_corrector.rho_pred
        self.j = self.predictor_corrector.j_pred

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
        }
        if self.propagate_vector_potential:
            arguments.update({
                'Omega': self.Omega,
                'A_t': self.A_t,
                'A_tm1': self.A_tm1,
                'N0': self.N0
            })
        return arguments

    def apply_initial_field(self):
        if self.vector_potential:
            self.psi = np.complex128(np.sqrt(self.system.field))
            self.A_t = np.zeros(3)
            self.A_t[self.direc] = self.k * SPEED_OF_LIGHT
            self.A_tm1 = self.A_t.copy()
        else:
            x = self.system.field.grid.r[self.direc]
            self.psi = np.sqrt(self.system.field) * np.exp(1j * self.k * x)
        self.psi.cplx = True

    def update_hamiltonian(self):
        func = self.functionals(self.rho, calcType=["V"], current=self.j)
        self.propagator.hamiltonian.v = func.potential
        if self.vector_potential:
            self.propagator.hamiltonian.A = self.A_t
        if self.correction:
            self.correct_potential = self.correct_functionals(self.rho, calcType=['V'], current=self.j).potential

    def calc_obeservables(self):
        delta_rho = self.rho - self.system.field
        self.delta_mu = (delta_rho * delta_rho.grid.r).integral()
        self.j_int = self.j.integral()
        self.calc_energy()

    def calc_energy(self):
        if self.correction:
            self.E = np.real(
                np.conj(self.psi) * (
                            self.propagator.hamiltonian(self.psi) + self.correct_potential * self.psi)).integral()
        else:
            self.E = np.real(np.conj(self.psi) * self.propagator.hamiltonian(self.psi)).integral()
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
        psi = npy.read(f, grid=self.system.field.grid)
        self.psi = DirectField(grid=self.system.field.grid, rank=1, griddata_3d=psi, cplx=True)
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
        npy.write(f, self.psi, grid=self.psi.grid)
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
