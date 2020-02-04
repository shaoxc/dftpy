import numpy as np
from scipy.optimize import minimize, line_search
from scipy import optimize as sopt
from functools import partial
from dftpy.field import DirectField
from dftpy.math_utils import LineSearchDcsrch, LineSearchDcsrch2, Brent, TimeData
from dftpy.math_utils import LBFGS
from abc import ABC, abstractmethod


class AbstractOptimization(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, optimization_options={}):
        pass

    @abstractmethod
    def get_direction(self):
        pass

    @abstractmethod
    def optimize_rho(self):
        pass


class Optimization(AbstractOptimization):
    """
    Class handling electron density optimization.
    minimizer based on scipy.minimize

    Attributes
    ---------
    optimization_method: string

    optimization_options: dict
            kwargs for the minim. method

    EnergyEvaluator: TotalEnergyAndPotential class   
            

    guess_rho: DirectField, optional
            an initial guess for the electron density

     Example
     -------
     EE = TotalEnergyAndPotential(...)
     opt = Optimization(EnergyEvaluator=EE)
     new_rho = opt(guess_rho)
    """

    def __init__(self, optimization_method="CG-HS", optimization_options=None, EnergyEvaluator=None, guess_rho=None):

        self.rho = guess_rho

        if optimization_options is None:
            self.optimization_options = {}
        else:
            self.optimization_options = optimization_options

        default_options = {
            "maxcor": 6,
            "ftol": 1.0e-7,
            "xtol": 1.0e-12,
            "maxfun": 50,
            "maxiter": 200,
            "maxls": 30,
            "econv": 1.0e-5,
            "c1": 1e-4,
            "c2": 0.2,
            "algorithm": "EMM",
            "vector": "Orthogonalization",
        }
        for key in default_options:
            if key not in self.optimization_options:
                self.optimization_options[key] = default_options[key]

        if EnergyEvaluator is None:
            raise AttributeError("Must provide an energy evaluator")
        else:
            self.EnergyEvaluator = EnergyEvaluator

        self.optimization_method = optimization_method

    def get_direction_CG(self, resA, dirA=None, method="CG-HS", **kwargs):
        # https ://en.wikipedia.org/wiki/Conjugate_gradient_method
        number = 1
        # HS->DY-CD
        if len(resA) == 1:
            beta = 0.0
        elif method == "CG-HS" and len(dirA) > 0:  # Maybe the best of the CG.
            beta = np.einsum("ijk->", resA[-1] * (resA[-1] - resA[-2])) / np.einsum(
                "ijk->", dirA[-1] * (resA[-1] - resA[-2])
            )
            # print('beta', beta)
        elif method == "CG-FR":
            beta = np.einsum("ijk->", resA[-1] ** 2) / np.einsum("ijk->", resA[-2] ** 2)
        elif method == "CG-PR":
            beta = np.einsum("ijk->", resA[-1] * (resA[-1] - resA[-2])) / np.einsum("ijk->", resA[-2] ** 2)
            beta = max(beta, 0.0)
        elif method == "CG-DY" and len(dirA) > 0:
            beta = np.einsum("ijk->", resA[-1] ** 2) / np.einsum("ijk->", dirA[-1] * (resA[-1] - resA[-2]))
        elif method == "CG-CD" and len(dirA) > 0:
            beta = -np.einsum("ijk->", resA[-1] ** 2) / np.einsum("ijk->", dirA[-1] * resA[-2])
        elif method == "CG-LS" and len(dirA) > 0:
            beta = np.einsum("ijk->", resA[-1] * (resA[-1] - resA[-2])) / np.einsum("ijk->", dirA[-1] * resA[-2])
        else:
            beta = np.einsum("ijk->", resA[-1] ** 2) / np.einsum("ijk->", resA[-2] ** 2)

        if len(dirA) > 0:
            direction = -resA[-1] + beta * dirA[-1]
        else:
            direction = -resA[-1]

        return direction, number

    def get_direction_TN(self, resA, phi=None, mu=None, **kwargs):
        direction = np.zeros_like(resA[-1])
        epsi = 1.0e-9
        rho = phi * phi
        if mu is None:
            func = self.EnergyEvaluator(rho, calcType="Potential")
            mu = (func.potential * rho).integral() / rho.N
        res = -resA[-1]
        p = res.copy()
        r0Norm = np.einsum("ijk, ijk->", res, res)
        r1Norm = r0Norm
        rConv = r0Norm * 0.1
        stat = "NOTCONV"
        Best = direction
        rLists = [1e10]
        for it in range(self.optimization_options["maxfun"]):
            phi1 = phi + epsi * p
            rho1 = phi1 * phi1
            func = self.EnergyEvaluator(rho1, calcType="Potential")
            # munew = (func.potential * rho1).integral() / rho.N
            Ap = ((func.potential - mu) * phi1 - resA[-1]) / epsi
            pAp = np.einsum("ijk, ijk->", p, Ap)
            if pAp < 0.0:
                if it == 0:
                    direction = r0Norm / pAp * p
                    stat = "WARN"
                else:
                    stat = "FAILED"
                    print("!WARN : pAp small than zero :iter = ", it)
                break
            alpha = r0Norm / pAp
            direction += alpha * p
            res -= alpha * Ap
            r1Norm = np.einsum("ijk, ijk->", res, res)
            # print('it', it, rConv, r1Norm)
            if r1Norm < min(rLists):
                Best = direction.copy()
            rLists.append(r1Norm)
            if r1Norm < rConv:
                stat = "CONV"
                break
            elif r1Norm > 1000 * min(rLists[:-1]):
                stat = "WARN : Not reduce"
                direction = Best
                break
            elif it > 10 and abs(r0Norm - r1Norm) < 0.1 * r0Norm:
                stat = "WARN : Change too small"
                direction = Best
                break
            beta = r1Norm / r0Norm
            r0Norm = r1Norm
            p = res + beta * p
        number = it + 1

        return direction, number

    def get_direction_LBFGS(self, resA, dirA=None, phi=None, method="CG-HS", lbfgs=None, mu=None, **kwargs):
        number = 1
        direction = np.zeros_like(resA[-1])
        rho = phi * phi
        if mu is None:
            func = self.EnergyEvaluator(rho, calcType="Potential")
            mu = (func.potential * rho).integral() / rho.N
        q = -resA[-1]
        alphaList = np.zeros(len(lbfgs.s))
        for i in range(len(lbfgs.s) - 1, 0, -1):
            alpha = lbfgs.rho[i] * np.einsum("ijk->", lbfgs.s[i] * q)
            alphaList[i] = alpha
            q -= alpha * lbfgs.y[i]

        if not lbfgs.H0:
            if len(lbfgs.s) < 1:
                gamma = 1.0
            else:
                gamma = np.einsum("ijk->", lbfgs.s[-1] * lbfgs.y[-1]) / np.einsum("ijk->", lbfgs.y[-1] * lbfgs.y[-1])
            direction = gamma * q
        else:
            direction = lbfgs.H0 * q

        for i in range(len(lbfgs.s)):
            beta = lbfgs.rho[i] * np.einsum("ijk->", lbfgs.y[i] * direction)
            direction += lbfgs.s[i] * (alphaList[i] - beta)

        return direction, number

    def get_direction_DIIS(self, resA, **kwargs):
        number = 1
        direction = -resA[-1]
        return direction, number

    def get_direction(self, resA, dirA=None, phi=None, method="CG-HS", lbfgs=None, mu=None):
        if method[0:2] == "CG":
            return self.get_direction_CG(resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu)
        elif method == "TN":
            return self.get_direction_TN(resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu)
        elif method == "LBFGS":
            return self.get_direction_LBFGS(resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu)
        elif method == "DIIS":
            return self.get_direction_DIIS(resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu)
        else:
            raise AttributeError("The %s direction method not implemented." % method)

    def OrthogonalNormalization(self, p, phi, Ne=None, vector="Orthogonalization"):
        if vector == "Orthogonalization":
            if Ne is None:
                Ne = (phi * phi).integral()
            N = Ne
            # ptest = p + phi ; # N = (ptest * ptest).integral()
            p -= (p * phi).integral() / Ne * phi
            pNorm = (p * p).integral()
            theta = np.sqrt(pNorm / N)
            p *= np.sqrt(Ne / pNorm)
        else:
            theta = 0.01
        return p, theta

    def ValueAndDerivative(self, phi, p, theta, Ne=None, algorithm="EMM", vector="Orthogonalization", func=None):
        if Ne is None:
            Ne = (phi * phi).integral()
        if vector == "Orthogonalization":
            newphi = phi * np.cos(theta) + p * np.sin(theta)
            newrho = newphi * newphi
        else:  # Scaling
            newphi = phi + p * theta
            newrho = newphi * newphi
            norm = Ne / newrho.integral()
            newrho *= norm
            newphi *= np.sqrt(norm)
        if func is not None:
            f = func
        if algorithm == "EMM":
            if func is None:
                f = self.EnergyEvaluator(newrho, calcType="Both")
            value = f.energy
        else:  # RMM
            if func is None:
                f = self.EnergyEvaluator(newrho, calcType="Both")
                # f = self.EnergyEvaluator(newrho, calcType = 'Potential')
            mu = (f.potential * newrho).integral() / Ne
            residual = (f.potential - mu) * newphi
            resN = np.einsum("ijk, ijk->", residual, residual) * phi.grid.dV
            value = resN
        if vector == "Orthogonalization":
            grad = 2.0 * np.einsum("ijk, ijk, ijk->", f.potential, newphi, (p * np.cos(theta) - phi * np.sin(theta)))
        else:
            grad = 2.0 * np.einsum("ijk, ijk, ijk->", f.potential, phi, p)
        # print('theta', theta, value, grad)
        return [value, grad, newphi, f]

    def optimize_rho(self, guess_rho=None):
        TimeData.Begin("Optimize")
        if guess_rho is None and self.rho is None:
            raise AttributeError("Must provide a guess density")
        else:
            rho = guess_rho
            self.old_rho = rho
        # -----------------------------------------------------------------------
        xtol = self.optimization_options["xtol"]
        maxls = self.optimization_options["maxls"]
        c1 = self.optimization_options["c1"]
        c2 = self.optimization_options["c2"]
        theta = 0.5
        # -----------------------------------------------------------------------
        EnergyHistory = []
        phi = np.sqrt(rho)
        func = self.EnergyEvaluator(rho)
        mu = (func.potential * rho).integral() / rho.N
        residual = (func.potential - mu) * phi
        residualA = []
        residualA.append(residual)
        directionA = []
        energy = func.energy
        EnergyHistory.append(energy)

        CostTime = TimeData.Time("Optimize")

        fmt = "{:8s}{:24s}{:16s}{:16s}{:8s}{:8s}{:16s}".format(
            "Step", "Energy(a.u.)", "dE", "dP", "Nd", "Nls", "Time(s)"
        )
        print(fmt)
        dE = energy
        resN = np.einsum("ijk, ijk->", residual, residual) * rho.grid.dV
        fmt = "{:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<8d}{:<16.6E}".format(0, energy, dE, resN, 1, 1, CostTime)
        print(fmt)
        Bound = self.optimization_options["maxcor"]

        if self.optimization_method == "LBFGS":
            lbfgs = LBFGS(H0=self.optimization_options["h0"], Bound=Bound)
        else:
            lbfgs = None

        if self.optimization_method == "DIIS":
            self.optimization_options["vector"] = "Scaling"

        for it in range(1, self.optimization_options["maxiter"]):
            p, NumDirectrion = self.get_direction(
                residualA, directionA, phi=phi, method=self.optimization_method, lbfgs=lbfgs, mu=mu
            )
            p, theta0 = self.OrthogonalNormalization(p, phi, vector=self.optimization_options["vector"])

            thetaDeriv0 = np.einsum("ijk, ijk, ijk->", func.potential, phi, p) * 2.0
            if thetaDeriv0 > 0:
                print("!WARN: Change to steepest decent")
                p = -residualA[-1]
                p, theta0 = self.OrthogonalNormalization(p, phi, vector=self.optimization_options["vector"])

            theta = min(theta0, theta)
            fun_value_deriv = partial(
                self.ValueAndDerivative,
                phi,
                p,
                algorithm=self.optimization_options["algorithm"],
                vector=self.optimization_options["vector"],
            )

            lsfun = "dcsrch"
            if lsfun == "dcsrch":
                func0 = fun_value_deriv(0.0, func=func)
                theta, _, _, task, NumLineSearch, valuederiv = LineSearchDcsrch2(
                    fun_value_deriv,
                    alpha0=theta,
                    func0=func0,
                    c1=c1,
                    c2=c2,
                    amax=np.pi,
                    amin=0.0,
                    xtol=xtol,
                    maxiter=maxls,
                )
            elif lsfun == "brent":
                theta, _, _, task, NumLineSearch, valuederiv = Brent(
                    fun_value_deriv, theta, brack=(0.0, theta), tol=1e-8, full_output=1
                )
            else:
                # p = -residual
                theta = 0.1
                newphi = phi + p * theta
                newrho = newphi * newphi
                norm = rho.N / newrho.integral()
                newrho *= norm
                newphi *= np.sqrt(norm)
                newfunc = self.EnergyEvaluator(newrho, calcType="Potential")
                NumLineSearch = 1

            if theta is None:
                print("!!!ERROR : Line-Search Failed!!!")
                print("!!!ERROR : Density Optimization NOT Converged  !!!")
                break
                # print('!WARN: Line-search failed and change to steepest decent')
                # theta = 0.001

            newphi = valuederiv[2]
            newfunc = valuederiv[3]
            old_phi, phi = phi, newphi

            rho = phi * phi
            func = newfunc
            # if self.optimization_options["algorithm"] == 'RMM' :
            # f = self.EnergyEvaluator(rho, calcType = 'Energy')
            # func.energy = f.energy
            # func = self.EnergyEvaluator(rho, calcType = 'Both')
            mu = (func.potential * rho).integral() / rho.N
            residual = (func.potential - mu) * phi
            # -----------------------------------------------------------------------
            if self.optimization_method == "DIIS":
                p = -residual
                phi = phi + p * theta
                rho = phi * phi
                norm = rho.N / rho.integral()
                rho *= norm
                phi *= np.sqrt(norm)
                func = self.EnergyEvaluator(rho, calcType="Both")
                mu = (func.potential * rho).integral() / rho.N
                residual = (func.potential - mu) * phi

            residualA.append(residual)

            if self.optimization_method == "LBFGS":
                lbfgs.update(phi - old_phi, residualA[-1] - residualA[-2])

            energy = func.energy
            EnergyHistory.append(energy)
            CostTime = TimeData.Time("Optimize")
            dE = EnergyHistory[-1] - EnergyHistory[-2]
            resN = np.einsum("ijk, ijk->", residual, residual) * rho.grid.dV
            fmt = "{:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<8d}{:<16.6E}".format(
                it, energy, dE, resN, NumDirectrion, NumLineSearch, CostTime
            )
            print(fmt)
            if abs(dE) < self.optimization_options["econv"]:
                # if True :
                if (
                    len(EnergyHistory) > 2
                    and abs(EnergyHistory[-1] - EnergyHistory[-3]) < self.optimization_options["econv"]
                ):
                    print("#### Density Optimization Converged ####")
                    break

            directionA.append(p)
            if len(residualA) > 2:
                residualA.pop(0)
            if len(directionA) > 2:
                directionA.pop(0)

        TimeData.End("Optimize")
        mu = (func.potential * rho).integral() / rho.N
        print('Chemical potential :', mu)
        return rho

    def __call__(self, guess_rho=None, calcType="Both"):
        return self.optimize_rho(guess_rho=guess_rho)
