# Collection of local and semilocal functionals

import numpy as np
from dftpy.field import DirectField,ReciprocalField
from dftpy.functional_output import Functional
from dftpy.math_utils import TimeData, PowerInt
from dftpy.kedf.tf import TF

__all__  =  ['GGA']

def GGAFs(s, functional = 'LKT', calcType = 'Both', GGAargs = None, **kwargs):
    '''
    Ref:
	1. Kinetic energy density study of some representative semilocal kinetic energy functionals
	2. Performance of Kinetic Energy Functionals for Interaction Energies in a Subsystem Formulation of Density Functional Theory
	3. Single-point kinetic energy density functionals : A pointwise kinetic energy density analysis and numerical convergence investigation
	4. Nonempirical generalized gradient approximation free-energy functional for orbital-free simulations
	5. A simple generalized gradient approximation for the noninteracting kinetic energy density functional
	6. Semilocal Pauli–Gaussian Kinetic Functionals for Orbital-Free Density Functional Theory Calculations of Solids
        7. Conjoint gradient correction to the Hartree-Fock kinetic- and exchange-energy density functionals
    '''
    cTF = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)
    tkf0 = 2.0 * (3.0 * np.pi**2)**(1.0/3.0)
    F = np.empty_like(s)
    dFds2 = np.empty_like(s) # Actually, it's 1/s*dF/ds
    # Introducing PROFESS 3.0: An advanced program for orbital-free density functional theory molecular dynamics simulations
    if functional == 'LKT' : # Ref. 5
        if not GGAargs :
            GGAargs = [1.3]
        ss = s/tkf0
        s2 = ss * ss
        mask1 = ss > 100.0
        mask2 = ss < 1E-5
        mask = np.invert(np.logical_or(mask1, mask2))
        F[mask] = 1.0/np.cosh(GGAargs[0] * ss[mask]) + 5.0/3.0 * (s2[mask])
        F[mask1] = 5.0/3.0 * (s2[mask1])
        F[mask2] = 1.0 + (5.0/3.0-0.5 * GGAargs[0] ** 2) * s2[mask2]+ 5.0/24.0 * GGAargs[0] ** 4 * s2[mask2] ** 2
        if calcType != 'Energy' :
            dFds2[mask] = 10.0/3.0 - GGAargs[0] * np.sinh(GGAargs[0] * ss[mask]) / np.cosh(GGAargs[0] * ss[mask]) ** 2 / ss[mask]
            dFds2[mask1] = 10.0/3.0
            dFds2[mask2] = 10.0/3.0 -  GGAargs[0] ** 2 + 5.0/6.0 * GGAargs[0] ** 4 * s2[mask2] - 61.0/120.0 * GGAargs[0] ** 6 * s2[mask2] ** 2
            dFds2 /=(tkf0 ** 2)
    elif functional == 'DK' : # Ref. 1 (8)
        if not GGAargs :
            GGAargs = [0.95,14.28111,-19.5762,-0.05,9.99802,2.96085]
        x = s*s/(72*cTF)
        Fa = (9.0 * GGAargs[5] * x ** 4 +GGAargs[2] * x ** 3 +GGAargs[1] * x ** 2 +GGAargs[0] * x + 1.0)
        Fb = (GGAargs[5] * x ** 3 +GGAargs[4] * x ** 2 +GGAargs[3] * x + 1.0)
        F = Fa / Fb
        if calcType != 'Energy' :
            dFds2 = (36.0 * GGAargs[5] * x ** 3 +3 * GGAargs[2] * x ** 2 + 2 * GGAargs[1] * x +GGAargs[0]) / Fb - \
                    Fa/(Fb * Fb) * (3.0 * GGAargs[5] * x ** 2 +2.0 * GGAargs[4] * x + GGAargs[3])
            dFds2 /= (36.0 * cTF)
    elif functional == 'LLP' or functional == 'LLP91' : # Ref. 1 (9), but have a mistake, arcsin should arcsinh. Ref. 7. or Ref. 2 (LLP91)
        if not GGAargs :
            GGAargs = [0.0044188, 0.0253]
        # symf = 2 ** (1.0/3.0) * GGAargs[0] #!!! I'm not sure if this is correct.
        symf = 2.0 ** (1.0/3.0) 
        ss = symf * s
        s2 = ss * ss
        Fa = GGAargs[0] * s2
        Fb = 1.0 + GGAargs[1] * ss * np.arcsinh(ss)
        F = 1.0 + Fa / Fb
        if calcType != 'Energy' :
            dFds2 = 2.0 * GGAargs[0] / Fb - (GGAargs[0] * GGAargs[1]* ss * np.arcsinh(ss) + Fa * GGAargs[1]/np.sqrt(1.0+s2))/(Fb * Fb)
            dFds2 *= (symf * symf)

    return F, dFds2

def GGA(rho, functional = 'LLP', calcType = 'Both', split = False, **kwargs):
# def GGA(rho, functional = 'LKT', calcType = 'Both', split = False, **kwargs):
    
    rhom = rho.copy()
    tol = 1E-16
    rhom[rhom < tol] = tol

    rho23 = rhom ** (2.0/3.0)
    rho53 = rho23 * rhom
    cTF = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)
    tf = cTF * rho53
    rho43 = rho23 * rho23
    rho83 = rho43 * rho43
    g=rho.grid.get_reciprocal().g

    rhoG = rho.fft()
    rhoGrad = []
    for i in range(3):
        item = (1j * g[..., i][..., np.newaxis] * rhoG).ifft(force_real = True)
        rhoGrad.append(item)
    s = np.sqrt(rhoGrad[0] ** 2 + rhoGrad[1] ** 2 + rhoGrad[2] ** 2) / rho43
    ene = 0.0
    F, dFds2 = GGAFs(s, functional = functional, calcType = calcType,  **kwargs)
    if calcType == 'Energy' :
        ene = np.einsum('ijkl, ijkl -> ', tf, F) * rhom.grid.dV
        pot = np.empty_like(rho)
    else :
        pot = 5.0/3.0 * cTF * rho23 * F
        pot += (-4.0/3.0 * tf * dFds2 * s * s / rhom)

        p3 = []
        for i in range(3):
            item = tf * dFds2 * rhoGrad[i] / rho83 
            p3.append(item.fft())
        pot3G = g[..., 0][..., None] * p3[0] +  g[..., 1][..., None] * p3[1] + g[..., 2][..., None] * p3[2]
        pot -= (1j * pot3G).ifft(force_real = True)

        if calcType == 'Both' :
            ene = np.einsum('ijkl, ijkl -> ', tf, F) * rhom.grid.dV

    OutFunctional = Functional(name='GGA-'+str(functional))
    OutFunctional.potential = pot
    OutFunctional.energy= ene

    if split :
        return {'GGA': OutFunctional}
    else :
        return OutFunctional
