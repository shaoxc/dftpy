import numpy as np
from dftpy.mpi import sprint
from dftpy.time_data import timer
from dftpy.functional import ExternalPotential, Hartree


class OESCF:
    """
    """
    def __init__(self, optimization = None, evaluator_emb = None, guess_rho=None, mixer = None,
            maxiter = 0):
        self.optimization = optimization
        self.evaluator_emb = evaluator_emb
        self.rho = guess_rho
        self.mixer = mixer
        self.maxiter = maxiter
        #
        self.iter = 0
        self.residual_norm_prev = 1.0
        if self.maxiter < 1 : self.maxiter = self.options['maxiter']

    @property
    def comm(self):
        return self.rho.mp.comm

    @property
    def mp(self):
        return self.rho.mp

    @property
    def options(self):
        return self.optimization.optimization_options

    def optimize_rho(self, **kwargs):
        return self.run(**kwargs)

    def run(self, guess_rho = None, **kwargs):
        #-----------------------------------------------------------------------
        if guess_rho is None and self.rho is None:
            raise AttributeError("Must provide a guess density")
        elif guess_rho is not None :
            self.rho = guess_rho
        self.econv = self.options['econv'] / 100.0
        #-----------------------------------------------------------------------
        for i in range(self.maxiter):
            self.get_density()
            self.mix()
            sprint(f" OESCF--> iter={self.iter:<5d} conv={self.options['econv']:<.3E}  econv={self.dp_norm:<.3E}", comm=self.comm, level = 2)
            if self.dp_norm < self.econv :
                sprint("##### OESCF Density Optimization Converged #####", comm=self.comm)
                break
        else :
            sprint("!WARN: Not converged, but reached max iterations", comm=self.comm)
        return self.rho

    @timer('OESCF')
    def get_density(self, res_max = None, **kwargs):
        self.iter += 1
        #-----------------------------------------------------------------------
        if res_max is None :
            res_max = self.residual_norm_prev

        if self.iter == 1 :
            self.options['econv0'] = self.options['econv'] * 1E4
            self.options['econv'] = self.options['econv0']
            res_max = 1.0

        if self.comm.size > 1 :
            res_max = self.comm.bcast(res_max, root = 0)

        norm = res_max
        if self.iter < 3 :
            norm = max(0.1, res_max)

        econv = self.options['econv0'] * norm
        if econv < self.options['econv'] :
            self.options['econv'] = econv
        if norm < 1E-7 and self.iter > 3 :
            self.options['maxiter'] = 4
        self.iterative(**kwargs)

    def iterative(self, **kwargs):
        #
        extpot = self.evaluator_emb(self.rho, calcType = ('V')).potential
        self.rho_prev = self.rho.copy()
        #
        self.optimization.EnergyEvaluator.funcDict['EMB'] = ExternalPotential(v = extpot)
        self.optimization_method = 'CG-HS'
        #
        self.optimization.optimize_rho(guess_rho=self.rho)
        self.rho = self.optimization.rho
        #
        del self.optimization.EnergyEvaluator.funcDict['EMB']

    def mix(self, **kwargs):
        r = self.rho - self.rho_prev
        self.dp_norm = Hartree.compute(r, calcType=('E')).energy
        self.residual_norm = np.sqrt((r*r).amean())
        if self.iter == 1 :
            self.dp_norm_prev = self.dp_norm
            self.residual_norm_prev = self.residual_norm
        else : # Sometime the density not fully converged.
            self.dp_norm, self.dp_norm_prev = self.dp_norm_prev, self.dp_norm
            self.residual_norm, self.residual_norm_prev = self.residual_norm_prev, self.residual_norm
            self.dp_norm = (self.dp_norm + self.dp_norm_prev)/2
            self.residual_norm = (self.residual_norm + self.residual_norm_prev)/2
        #
        if self.mixer :
            self.rho = self.mixer(self.rho_prev, self.rho, **kwargs)
