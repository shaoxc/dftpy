import numpy as np
from scipy.optimize import minimize, line_search
from scipy import optimize as sopt
from functools import partial
from dftpy.field import DirectField
from dftpy.math_utils import LineSearchDcsrch,LineSearchDcsrch2, TimeData

class LBFGS(object):

    def __init__(self, H0 = 1.0, Bound = 5):
        self.Bound = Bound
        self.H0 = H0
        self.s = []
        self.y = []
        self.rho = []

    def update(self, dx, dg):
        if len(self.s) > self.Bound :
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)
        self.s.append(dx)
        self.y.append(dg)
        rho = 1.0/np.einsum('ijkl->', dg * dx)
        self.rho.append(rho)

class Optimization(object):
    '''
    Class handling electron density optimization.
    minimizer based on scipy.minimize

    Attributes
    ---------
    optimization_method: string
            See scipy.minimize for available methods
            default: L-BFGS-B

    optimization_options: dict
            kwargs for the minim. method

    EnergyEvaluator: TotalEnergyAndPotential class   
            

    guess_rho: DirectField, optional
            an initial guess for the electron density

     Example
     -------
     EE = TotalEnergyAndPotential(...)
     opt = Optimization(EnergyEvaluator=EE)
     new_rho = Optimization.get_optimal_rho(guess_rho)
    ''' 

    def __init__(self, 
                 optimization_method='CG-HS', 
                 optimization_options=None,
                 EnergyEvaluator=None,
                 guess_rho=None):
                    
        self.rho = guess_rho

        if optimization_options is None:
            self.optimization_options={}
        else:
            self.optimization_options = optimization_options

        if not 'disp' in self.optimization_options.keys():
            self.optimization_options["disp"] = None
        if not 'maxcor' in self.optimization_options.keys():
            self.optimization_options["maxcor"] = 5
        if not 'ftol' in self.optimization_options.keys():
            self.optimization_options["ftol"] = 1.0e-7
        if not 'xtol' in self.optimization_options.keys():
            self.optimization_options["xtol"] = 1.0e-12
        if not 'maxfun' in self.optimization_options.keys():
            self.optimization_options["maxfun"] = 50
        if not 'maxiter' in self.optimization_options.keys():
            self.optimization_options["maxiter"] = 100
        if not 'maxls' in self.optimization_options.keys():
            self.optimization_options["maxls"] = 30
        if not 'econv' in self.optimization_options.keys():
            self.optimization_options["econv"] = 1.0e-5
        if not 'vector' in self.optimization_options.keys():
            self.optimization_options["vector"] = 'Orthogonalization'
        if not 'c1' in self.optimization_options.keys():
            self.optimization_options["c1"] = 1E-4
        if not 'c2' in self.optimization_options.keys():
            self.optimization_options["c2"] = 0.2
        if not 'algorithm' in self.optimization_options.keys():
            self.optimization_options["algorithm"] = 'EMM'
        
        if EnergyEvaluator is None:
            raise AttributeError('Must provide an energy evaluator')
        else:
            self.EnergyEvaluator = EnergyEvaluator
        
        self.optimization_method = optimization_method
        
    def get_optimal_rho(self,guess_rho=None):
        if guess_rho is None and self.rho is None:
            raise AttributeError('Must provide a guess density')
        else:
            rho = guess_rho
            self.old_rho = rho
        phi = np.sqrt(rho).ravel()
        res = minimize(fun=self.EnergyEvaluator,
                       jac=True,x0=phi,
                       method=self.optimization_method,
                       options=self.optimization_options)
        print(res.message)
        rho = DirectField(rho.grid,griddata_3d=np.reshape(res.x**2,np.shape(rho)),rank=1)
        self.rho = rho
        return rho

    def get_direction(self, resA, dirA, phi=None, method='CG-HS', lbfgs=None, mu=None):
        #https ://en.wikipedia.org/wiki/Conjugate_gradient_method
        number = 1
        if method[0:2] == 'CG' :
            #HS->DY-CD
            if len(resA) == 1 :
                beta = 0.0
            elif method == 'CG-HS' and len(dirA) > 0 : #Maybe the best of the CG.
                beta = np.einsum('ijkl->',resA[-1] *(resA[-1]-resA[-2]) ) / np.einsum('ijkl->',dirA[-1]*(resA[-1]-resA[-2]))
                # print('beta', beta)
            elif  method == 'CG-FR':
                beta = np.einsum('ijkl->',resA[-1] ** 2) / np.einsum('ijkl->',resA[-2] ** 2) 
            elif method == 'CG-PR' :
                beta = np.einsum('ijkl->',resA[-1] *(resA[-1]-resA[-2]) ) / np.einsum('ijkl->',resA[-2] ** 2) 
            elif method == 'CG-DY' and len(dirA) > 0 :
                beta = np.einsum('ijkl->',resA[-1] **2 ) / np.einsum('ijkl->',dirA[-1]*(resA[-1]-resA[-2]))
            elif method == 'CG-CD' and len(dirA) > 0 :
                beta = -np.einsum('ijkl->',resA[-1] **2 ) / np.einsum('ijkl->',dirA[-1]*resA[-2])
            elif method == 'CG-LS' and len(dirA) > 0 :
                beta = np.einsum('ijkl->',resA[-1] *(resA[-1]-resA[-2]) ) / np.einsum('ijkl->',dirA[-1]*resA[-2])
            else :
                beta = np.einsum('ijkl->',resA[-1] ** 2) / np.einsum('ijkl->',resA[-2] ** 2) 

            if len(dirA) > 0 :
                direction = -resA[-1] + beta * dirA[-1]
            else :
                direction = -resA[-1]

        elif method == 'TN' :
            direction = np.zeros_like(resA[-1])
            epsi = 1.0E-9
            rho = phi * phi
            if mu is None :
                func = self.EnergyEvaluator.ComputeEnergyPotential(rho, calcType = 'Potential')
                mu = (func.potential * rho).integral() / self.EnergyEvaluator.N
            res = -resA[-1]
            p = res.copy()
            r0Norm = np.einsum('ijkl, ijkl->', res, res)
            r1Norm = r0Norm
            rConv = r0Norm * 0.1
            stat = 'NOTCONV'
            Best = direction
            rLists = [1E10]
            for it in range(self.optimization_options["maxfun"]):
                phi1 = phi + epsi * p
                rho1 = phi1 * phi1
                func = self.EnergyEvaluator.ComputeEnergyPotential(rho1, calcType = 'Potential')
                # munew = (func.potential * rho1).integral() / self.EnergyEvaluator.N
                Ap = ((func.potential - mu) * phi1 - resA[-1]) / epsi
                pAp = np.einsum('ijkl, ijkl->', p, Ap)
                if pAp < 0.0 :
                    if it == 0 :
                        direction = r0Norm / pAp * p
                        stat = 'WARN'
                    else :
                        stat = 'FAILED'
                        print('!WARN : pAp small than zero :iter = ', it)
                    break
                alpha = r0Norm / pAp
                direction += alpha * p
                res -= alpha * Ap
                r1Norm = np.einsum('ijkl, ijkl->', res, res)
                # print('it', it, rConv, r1Norm)
                if r1Norm < min(rLists):
                    Best = direction.copy()
                rLists.append(r1Norm)
                if r1Norm < rConv :
                    stat = 'CONV'
                    break
                elif r1Norm > 10 * min(rLists[:-1]):
                    stat = 'WARN : Not reduce'
                    direction = Best
                    break
                elif it > 10 and abs(r0Norm - r1Norm) < 0.1 * r0Norm :
                    stat = 'WARN : Change too small'
                    direction = Best
                    break
                beta = r1Norm / r0Norm
                r0Norm = r1Norm
                p = res + beta * p 
            number = it + 1

        elif method == 'LBFGS' :
            direction = np.zeros_like(resA[-1])
            rho = phi * phi
            if mu is None :
                func = self.EnergyEvaluator.ComputeEnergyPotential(rho, calcType = 'Potential')
                mu = (func.potential * rho).integral() / self.EnergyEvaluator.N
            q = -resA[-1]
            alphaList = np.zeros(len(lbfgs.s))
            for i in range(len(lbfgs.s)-1, 0, -1):
                alpha = lbfgs.rho[i] * np.einsum('ijkl->', lbfgs.s[i] * q)
                alphaList[i] = alpha
                q -= alpha * lbfgs.y[i]

            if not lbfgs.H0 :
                if len(lbfgs.s) < 1 :
                    gamma = 1.0
                else :
                    gamma = np.einsum('ijkl->', lbfgs.s[-1] * lbfgs.y[-1]) / np.einsum('ijkl->', lbfgs.y[-1] * lbfgs.y[-1])
                direction = gamma * q
            else :
                direction = lbfgs.H0 * q

            for i in range(len(lbfgs.s)):
                beta = lbfgs.rho[i] * np.einsum('ijkl->', lbfgs.y[i] * direction)
                direction += lbfgs.s[i] * (alphaList[i]-beta)

        elif method == 'DIIS' :
            direction = -resA[-1]

        return direction, number


    def OrthogonalNormalization(self, p, phi, vector = 'Orthogonalization'):
        if vector == 'Orthogonalization' :
            N = self.EnergyEvaluator.N
            # ptest = p + phi ; # N = (ptest * ptest).integral()
            p -= ((p * phi).integral() / self.EnergyEvaluator.N * phi)
            pNorm = (p * p).integral()
            theta = np.sqrt( pNorm / N)
            p *= np.sqrt(self.EnergyEvaluator.N / pNorm)
        else :
            theta = 0.01
        return p, theta
    #-----------------------------------------------------------------------
    def ValueAndDerivative(self, phi, p, theta, algorithm = 'EMM', vector = 'Orthogonalization', func = None):
        if vector == 'Orthogonalization' :
            newphi = phi * np.cos(theta) + p * np.sin(theta)
            newrho = newphi * newphi
        else : # Scaling
            newphi = phi + p * theta
            newrho = newphi * newphi
            norm = self.EnergyEvaluator.N / newrho.integral() 
            newrho *= norm
            newphi *= np.sqrt(norm)
        if func is not None :
            f = func
        if algorithm == 'EMM' : 
            if func is None :
                f = self.EnergyEvaluator.ComputeEnergyPotential(newrho, calcType = 'Both')
            value = f.energy
        else : #RMM
            if func is None :
                f = self.EnergyEvaluator.ComputeEnergyPotential(newrho, calcType = 'Potential')
            mu = (f.potential * newrho).integral() / self.EnergyEvaluator.N
            residual = (f.potential - mu) * newphi
            resN = np.einsum('ijkl, ijkl->', residual, residual)*phi.grid.dV
            value = resN
        if vector == 'Orthogonalization' :
            grad = 2.0 * np.einsum('ijkl, ijkl, ijkl->', f.potential, newphi, (p * np.cos(theta) - phi * np.sin(theta)))
        else :
            grad = 2.0 * np.einsum('ijkl, ijkl, ijkl->', f.potential, phi, p)
        # print('theta', theta, value, grad)
        return [value, grad, newphi, f]
    #-----------------------------------------------------------------------

    def optimize_rho(self, guess_rho = None):
        TimeData.Begin('Optimize')
        if guess_rho is None and self.rho is None:
            raise AttributeError('Must provide a guess density')
        else:
            rho = guess_rho
            self.old_rho = rho
        #-----------------------------------------------------------------------
        xtol = self.optimization_options["xtol"]
        maxls = self.optimization_options["maxls"]
        c1 = self.optimization_options["c1"]
        c2 = self.optimization_options["c2"]
        theta = 0.5
        #-----------------------------------------------------------------------
        EnergyHistory = []
        phi = np.sqrt(rho)
        func = self.EnergyEvaluator.ComputeEnergyPotential(rho)
        mu = (func.potential * rho).integral() / self.EnergyEvaluator.N
        residual = (func.potential - mu)* phi
        residualA = []
        residualA.append(residual)
        directionA = []
        energy = func.energy
        EnergyHistory.append(energy)

        CostTime = TimeData.Time('Optimize')

        fmt = "{:8s}{:24s}{:16s}{:16s}{:8s}{:8s}{:16s}".format(\
                'Step','Energy(a.u.)', 'dE', 'dP', 'Nd', 'Nls', 'Time(s)')
        print(fmt)
        dE = energy
        resN = np.einsum('ijkl, ijkl->', residual,residual) * rho.grid.dV
        fmt = "{:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<8d}{:<16.6E}".format(\
                0, energy, dE, resN, 1, 1, CostTime)
        print(fmt)
        Bound = self.optimization_options["maxcor"]

        if self.optimization_method=='LBFGS' :
            lbfgs = LBFGS(H0 = self.optimization_options["h0"], Bound = Bound)
        else :
            lbfgs = None

        if self.optimization_method=='DIIS' :
            self.optimization_options["vector"] = 'Scaling'

        for it in range(1, self.optimization_options["maxiter"]):
            p, NumDirectrion = self.get_direction(residualA, directionA, phi=phi, method=self.optimization_method, lbfgs=lbfgs, mu=mu)
            p, theta0 = self.OrthogonalNormalization(p, phi, self.optimization_options["vector"])

            thetaDeriv0 = np.einsum('ijkl, ijkl, ijkl->', func.potential, phi, p) * 2.0
            if thetaDeriv0 > 0 :
                print('!WARN: Change to steepest decent')
                p = -residualA[-1]
                p, theta0 = self.OrthogonalNormalization(p, phi, self.optimization_options["vector"])

            theta = min(theta0, theta)
            fun_value_deriv = partial(self.ValueAndDerivative, phi, p, \
                    algorithm = self.optimization_options["algorithm"], \
                    vector = self.optimization_options["vector"])

            lsfun = 'dcsrch'
            if lsfun == 'dcsrch' :
                func0 = fun_value_deriv(0.0, func = func)
                theta,_, _, task, NumLineSearch, valuederiv =  LineSearchDcsrch2(fun_value_deriv, alpha0 = theta,
                       func0 = func0, c1=c1, c2=c2, amax=np.pi, amin=0.0, xtol=xtol, maxiter = maxls)
            elif lsfun == 'brent' :
                theta, ene, _, NumLineSearch = sopt.brent(thetaEnergy, theta, brack=(0.0, theta), tol=1E-8, full_output=1)
            else : 
                # p = -residual
                theta = 0.1
                newphi = phi + p * theta
                newrho = newphi * newphi
                norm = self.EnergyEvaluator.N / newrho.integral() 
                newrho *= norm
                newphi *= np.sqrt(norm)
                newfunc = self.EnergyEvaluator.ComputeEnergyPotential(newrho, calcType = 'Potential')
                NumLineSearch = 1

            if theta is None :
                print('!!!ERROR : Line-Search Failed!!!')
                print('!!!ERROR : Density Optimization NOT Converged  !!!')
                break
                # print('!WARN: Line-search failed and change to steepest decent')
                # theta = 0.001

            # [value, grad, newphi, f]
            newphi = valuederiv[2]
            newfunc = valuederiv[3]
            old_phi, phi = phi, newphi

            rho = phi * phi
            func = newfunc
            if self.optimization_options["algorithm"] == 'RMM' :
                f = self.EnergyEvaluator.ComputeEnergyPotential(rho, calcType = 'Energy')
                func.energy = f.energy
                # func = self.EnergyEvaluator.ComputeEnergyPotential(rho, calcType = 'Both')
            mu = (func.potential * rho).integral() / self.EnergyEvaluator.N
            residual = (func.potential - mu)* phi
            #-----------------------------------------------------------------------
            if self.optimization_method=='DIIS' :
                p = -residual
                phi = phi + p * theta
                rho = phi * phi
                norm = self.EnergyEvaluator.N / rho.integral() 
                rho *= norm
                phi *= np.sqrt(norm)
                func = self.EnergyEvaluator.ComputeEnergyPotential(rho, calcType = 'Both')
                mu = (func.potential * rho).integral() / self.EnergyEvaluator.N
                residual = (func.potential - mu)* phi

            residualA.append(residual)

            if self.optimization_method=='LBFGS' :
                lbfgs.update(phi-old_phi, residualA[-1]-residualA[-2])

            energy = func.energy
            EnergyHistory.append(energy)
            CostTime = TimeData.Time('Optimize')
            dE = EnergyHistory[-1]-EnergyHistory[-2]
            resN = np.einsum('ijkl, ijkl->', residual,residual) * rho.grid.dV
            fmt = "{:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<8d}{:<16.6E}".format(\
                    it, energy, dE, resN, NumDirectrion, NumLineSearch, CostTime)
            print(fmt)
            if abs(dE) < self.optimization_options["econv"] :
                # if True :
                if len(EnergyHistory) > 2 and abs(EnergyHistory[-1]-EnergyHistory[-3]) < self.optimization_options["econv"] :
                    print('#### Density Optimization Converged ####')
                    break

            directionA.append(p)
            if len(residualA) > 2 :
                residualA.pop(0)
            if len(directionA) > 2 :
                directionA.pop(0)

        TimeData.End('Optimize')
        return rho
