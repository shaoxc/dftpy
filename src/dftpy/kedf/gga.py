# Collection of local and semilocal functionals

import numpy as np
from dftpy.field import DirectField,ReciprocalField
from dftpy.functional_output import Functional
from dftpy.math_utils import TimeData, PowerInt
from dftpy.kedf.tf import TF

__all__  =  ['GGA', 'GGA_KEDF_list', 'GGAStress']

GGA_KEDF_list = [
        'LKT', 
        'DK', 
        'LLP', 
        'LLP91', 
        'OL1', 
        'OL2', 
        'T92', 
        'THAK', 
        'B86A', 
        'B86B', 
        'DK87', 
        'PW86', 
        'PW91O', 
        'PW91', 
        'PW91k', # same as `PW91`
        'LG94', 
        'E00', 
        'P92', 
        'PBE2', 
        'PBE3', 
        'PBE4', 
        'P82', 
        'TW02', 
        'APBE', 
        'APBEK', # same as `APBE`
        'REVAPBE', 
        'REVAPBEK',  # same as `REVAPBE`
        'VJKS00', 
        'LC94', 
        'VT84F',
        'SMP19',
        ]

def GGAStress(rho, functional = 'LKT', energy=None, potential=None, **kwargs):
    '''
    Not finished.
    '''
    rhom = rho.copy()
    tol = 1E-16
    rhom[rhom < tol] = tol

    rho23 = rhom ** (2.0/3.0)
    rho53 = rho23 * rhom
    rho43 = rho23 * rho23
    rho83 = rho43 * rho43
    cTF = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)
    ckf2 = (3.0 * np.pi**2)**(2.0/3.0)
    tf = cTF * rho53
    vkin2 = tf * ckf2 * dFds2 / rho83
    dRho_ij = []
    for i in range(3):
        dRho_ij.append((1j * g[..., i][..., np.newaxis] * rhoG).ifft(force_real = True))
    stress = np.zeros((3, 3))

    if potential is None :
        gga = GGA(rho, functional =functional, calcType = 'Both', **kwargs)
        energy = gga.energy
        potential = gga.potential

    rhoP = np.einsum('ijkl, ijkl', rho, potential)
    for i in range(3):
        for j in range(i, 3):
            stress[i, j] = np.einsum('ijk, ijk', vkin2, dRho_ij[i]*dRho_ij[j])
            if i == j :
                stress[i, j] += (energy - rhoP)
            stress[j, i] = stress[i, j]

def GGAFs(s, functional = 'LKT', calcType = 'Both', parms = None, **kwargs):
    '''
    ckf = (3\pi^2)^{1/3}
    cTF = (3/10) * (3\pi^2)^{2/3} = (3/10) * ckf^2
    bb = 2^{4/3} * ckf = 2^{1/3} * tkf0
    x = (5/27) * ss * ss 

    In DFTpy, default we use following definitions :
    tkf0 = 2 * ckf
    ss = s/tkf0
    x = (5/27) * s * s / (tkf0^2) = (5/27) * s * s / (4 * ckf^2) = (5 * 3)/(27 * 10 * 4) * s * s / cTF
    x = (5/27) * ss * ss = s*s / (72*cTF)
    bs = bb * ss = 2^{1/3} * tkf0  * ss  = 2^{1/3} * s
    b = 2^{1/3}

    Some KEDF have passed the test compared with `PROFESS3.0`.

    I hope someone can write all these equations...

    Ref:
        @article{garcia2007kinetic,
          title={Kinetic energy density study of some representative semilocal kinetic energy functionals}}
        @article{gotz2009performance,
          title={Performance of kinetic energy functionals for interaction energies in a subsystem formulation of density functional theory}}
        @article{lacks1994tests, 
          title = {Tests of nonlocal kinetic energy functionals}}
        @misc{hfofke,
          url = {http://www.qtp.ufl.edu/ofdft/research/KE_refdata_27ii18/Explanatory_Post_HF_OFKE.pdf}}
        @article{xia2015single,
          title={Single-point kinetic energy density functionals: A pointwise kinetic energy density analysis and numerical convergence investigation}}
        @article{luo2018simple,
          title={A simple generalized gradient approximation for the noninteracting kinetic energy density functional},
    '''
    cTF = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)
    tkf0 = 2.0 * (3.0 * np.pi**2)**(1.0/3.0)
    b = 2 ** (1.0/3.0)
    tol2 = 1E-8 # It's a very small value for safe deal with 1/s
    F = np.empty_like(s)
    dFds2 = np.empty_like(s) # Actually, it's 1/s*dF/ds
    if functional == 'LKT' : # \cite{luo2018simple}
        if not parms : parms = [1.3]
        ss = s/tkf0
        s2 = ss * ss
        mask1 = ss > 100.0
        mask2 = ss < 1E-5
        mask = np.invert(np.logical_or(mask1, mask2))
        F[mask] = 1.0/np.cosh(parms[0] * ss[mask]) + 5.0/3.0 * (s2[mask])
        F[mask1] = 5.0/3.0 * (s2[mask1])
        F[mask2] = 1.0 + (5.0/3.0-0.5 * parms[0] ** 2) * s2[mask2]+ 5.0/24.0 * parms[0] ** 4 * s2[mask2] ** 2 #- 61.0/720.0 * parms[0] ** 6 * s2[mask2] ** 3
        if calcType != 'Energy' :
            dFds2[mask] = 10.0/3.0 - parms[0] * np.sinh(parms[0] * ss[mask]) / np.cosh(parms[0] * ss[mask]) ** 2 / ss[mask]
            dFds2[mask1] = 10.0/3.0
            dFds2[mask2] = 10.0/3.0 -  parms[0] ** 2 + 5.0/6.0 * parms[0] ** 4 * s2[mask2] - 61.0/120.0 * parms[0] ** 6 * s2[mask2] ** 2
            dFds2 /=(tkf0 ** 2)

    elif functional == 'DK' : # \cite{garcia2007kinetic} (8)
        if not parms : parms = [0.95,14.28111,-19.5762,-0.05,9.99802,2.96085]
        x = s*s/(72*cTF)
        Fa = (9.0 * parms[5] * x ** 4 +parms[2] * x ** 3 +parms[1] * x ** 2 +parms[0] * x + 1.0)
        Fb = (parms[5] * x ** 3 +parms[4] * x ** 2 +parms[3] * x + 1.0)
        F = Fa / Fb
        if calcType != 'Energy' :
            dFds2 = (36.0 * parms[5] * x ** 3 +3 * parms[2] * x ** 2 + 2 * parms[1] * x +parms[0]) / Fb - \
                    Fa/(Fb * Fb) * (3.0 * parms[5] * x ** 2 +2.0 * parms[4] * x + parms[3])
            dFds2 /= (36.0 * cTF)

    elif functional == 'LLP' or functional == 'LLP91' : # \cite{garcia2007kinetic} (9)[!x] \cite{gotz2009performance} (18)
        if not parms : parms = [0.0044188, 0.0253]
        bs = b * s
        bs2 = bs * bs
        Fa = parms[0] * bs2
        Fb = 1.0 + parms[1] * bs * np.arcsinh(bs)
        F = 1.0 + Fa / Fb
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[0] / Fb - (parms[0] * parms[1]* bs * np.arcsinh(bs) + Fa * parms[1]/np.sqrt(1.0+bs2))/(Fb * Fb)
            dFds2 *= (b * b)

    elif functional == 'OL1' or functional == 'OL' : # \cite{gotz2009performance} (16)
        if not parms : parms = [0.00677]
        F = 1.0 + s * s/72.0/cTF + parms[0]/cTF * s
        if calcType != 'Energy' :
            mask = s > tol2
            dFds2[:] = 1.0/36.0/cTF
            dFds2[mask] += parms[0]/cTF/s[mask]

    elif functional == 'OL2' : # \cite{gotz2009performance} (17)
        if not parms : parms = [0.0887]
        F = 1.0 + s * s/72.0/cTF + parms[0]/cTF * s/(1 + 4 * s)
        if calcType != 'Energy' :
            mask = s > tol2
            dFds2[:] = 1.0/36.0/cTF
            dFds2[mask] += parms[0]/cTF/(1 + 4 * s[mask]) ** 2/s[mask]

    elif functional == 'T92' or functional == 'THAK' : # \cite{garcia2007kinetic} (12),\cite{gotz2009performance} (22), \cite{hfofke} (15)[!x] 
        if not parms : parms = [0.0055, 0.0253, 0.072]
        bs = b * s
        bs2 = bs * bs
        F = 1.0 + parms[0] * bs2 / (1.0 + parms[1] * bs * np.arcsinh(bs)) - parms[2] * bs / (1.0+2 ** (5.0/3.0) * bs)
        # F = 1.0 + parms[0] * bs2 / (1.0 + parms[1] * bs * np.arcsinh(bs)) - parms[2] * bs / (1.0+ 2**2 * bs)
        if calcType != 'Energy' :
            mask = s > tol2
            Fb = (1.0 + parms[1] * bs * np.arcsinh(bs)) ** 2
            dFds2 =(-(parms[0] * parms[1] * bs2)/np.sqrt(1 + bs2) + \
                    (parms[0] * parms[1] * bs * np.arcsinh(bs)) + 2.0 * parms[0]) / Fb
            dFds2[mask] -= parms[2]/(1.0+2 ** (5.0/3.0) * bs[mask]) ** 2/bs[mask]
            # dFds2[mask] -= parms[2]/(1.0+4* bs[mask]) ** 2/bs[mask]
            dFds2 *= (b * b)

    elif functional == 'B86A' or functional == 'B86' : # \cite{garcia2007kinetic} (13)
        if not parms : parms = [0.0039, 0.004]
        bs = b * s
        bs2 = bs * bs
        Fa = parms[0] * bs2
        Fb = 1.0 + parms[1] * bs2
        F = 1.0 + Fa / Fb
        if calcType != 'Energy' :
            dFds2 = 2 * parms[0] / (Fb * Fb) * (b * b)

    elif functional == 'B86B' : # \cite{garcia2007kinetic} (14)
        if not parms : parms = [0.00403, 0.007]
        bs = b * s
        bs2 = bs * bs
        Fa = parms[0] * bs2
        Fb = (1.0 + parms[1] * bs2) ** (4.0/5.0)
        F = 1.0 + Fa / Fb
        if calcType != 'Energy' :
            dFds2 = (2 * parms[0] * (parms[1] * bs2 + 5.0)) /(5 * (1.0 + parms[1] * bs2) * Fb) * (b * b)

    elif functional == 'DK87' : # \cite{garcia2007kinetic} (15)
        if not parms : parms = [7.0/324.0/(18.0*np.pi**4)**(1.0/3.0), 0.861504, 0.044286]
        bs = b * s
        bs2 = bs * bs
        Fa = parms[0] * bs2 * (1 + parms[1] * bs)
        Fb = (1.0 + parms[2] * bs2)
        F = 1.0 + Fa / Fb
        if calcType != 'Energy' :
            dFds2 = parms[0] * (2.0+ 3.0 * parms[1] * bs + parms[1] * parms[2] * bs2 * bs)/(Fb * Fb) * (b * b)

    elif functional == 'PW86' : # \cite{gotz2009performance} (19)
        if not parms : parms = [1.296, 14.0, 0.2]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s2 * s4
        Fa = (1.0 + parms[0] * s2 + parms[1] * s4 + parms[2] * s6)
        F = Fa ** (1.0/15.0)
        if calcType != 'Energy' :
            dFds2 = 2.0/15.0 * (parms[0] + 2 * parms[1] * s2 + 3 * parms[2] * s4)/Fa ** (14.0/15.0)
            dFds2 /= (tkf0 * tkf0)

    elif functional == 'PW91O' : # \cite{gotz2009performance} (20)
        if not parms : parms = [0.093907,0.26608,0.0809615,100.0,76.320,0.57767E-4]#(A1, A2, A3, A4, A, B1)
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = (parms[1]-parms[2] * np.exp(-parms[3] * s2)) * s2 - parms[5] * s4
        Fb = 1.0 + parms[0] * ss * np.arcsinh(parms[4] * ss) + parms[5] * s4
        F = 1.0+ Fa/Fb
        if calcType != 'Energy' :
            Fa_s2 = (parms[1]-parms[2] * np.exp(-parms[3] * s2))- parms[5] * s2

            dFds2 = 2.0 * (parms[1]+ (parms[3] * s2-1) * parms[2] * np.exp(-parms[3] * s2)- 4.0 * parms[5] * s2) / Fb - \
                    Fa_s2 * ss /(Fb * Fb) * \
                    ((parms[0] * parms[4] * ss)/(parms[4] ** 2 * s2 + 1)+ parms[0] * np.arcsinh(parms[4]*ss) + 4.0*parms[5]*s2)
            dFds2 /= (tkf0 * tkf0)

    elif functional == 'PW91' or functional == 'PW91k' : # \cite{lacks1994tests} (16) and \cite{garcia2007kinetic} (17)[!x]
        if not parms : parms = [0.19645, 0.2747, 0.1508, 100.0, 7.7956, 0.004] #(A1, A2, A3, A4, A, B1)
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = 1.0 + parms[0] * ss * np.arcsinh(parms[4] * ss) + (parms[1]-parms[2] * np.exp(-parms[3] * s2)) * s2
        Fb = 1.0 + parms[0] * ss * np.arcsinh(parms[4] * ss) + parms[5] * s4
        F = Fa/Fb
        if calcType != 'Energy' :
            Fa_s = parms[0] * parms[4] * ss/np.sqrt(parms[4] ** 2 * s2 + 1)  + parms[0] * np.arcsinh(parms[4] * ss) + 2.0 * ss * (parms[1] + 2.0 * parms[2] * parms[3] * s2 * np.exp(-parms[3] * s2) -2.0 * parms[2] * np.exp(-parms[3] * s2))

            dFds2 = Fa_s/Fb - Fa/(Fb * Fb) * ( parms[0] * parms[4] * ss/np.sqrt(parms[4] ** 2 * s2 + 1) + \
		    (parms[0] * np.arcsinh(parms[4] * ss) + 4.0*parms[5]*s2*ss))

            mask = s > tol2
            dFds2[mask] /= ss[mask]
            dFds2 /= (tkf0 * tkf0)

    elif functional == 'LG94' : # \cite{garcia2007kinetic} (18)
        if not parms : parms = [(1E-8+0.1234)/0.024974, 29.790, 22.417, 12.119, 1570.1, 55.944, 0.024974] # a2, a4, a6, a8, a10, a12, b
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        s8 = s4 * s4
        s10= s4 * s6
        s12= s6 * s6
        Fa = 1.0 + parms[0] * s2 + parms[1] * s4 + parms[2] * s6 + parms[3] * s8 + parms[4] * s10 + parms[5] * s12
        Fb = 1.0 + 1E-8 * s2
        F = (Fa/Fb) ** parms[6]
        if calcType != 'Energy' :
            dFds2 = -2 * parms[6] * F * (1E-8 * Fa - Fb * (parms[0] + 2 * parms[1] * s2 + 3 * parms[2] * s4 + 4 * parms[3] * s6 + 5 * parms[4] * s8 + 6 * parms[5] * s10)) / (Fa * Fb)
            dFds2 /= (tkf0 * tkf0)

    elif functional == 'E00' : # \cite{gotz2009performance} (14)
        if not parms : parms = [135.0, 28.0, 5.0, 3.0]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = parms[0] + parms[1] * s2 + parms[2] * s4
        Fb = parms[0] + parms[3] * s2
        F = Fa/Fb
        if calcType != 'Energy' :
            dFds2 = (2.0*parms[1] + 4.0*parms[2] * s2)/Fb - (2.0* parms[3]*Fa)/(Fb*Fb)
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'P92' : # \cite{gotz2009performance} (15)
        if not parms : parms = [1.0,88.3960,16.3683,88.2108]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = parms[0] + parms[1] * s2 + parms[2] * s4
        Fb = parms[0] + parms[3] * s2
        F = Fa/Fb
        if calcType != 'Energy' :
            dFds2 = (2.0*parms[1] + 4.0*parms[2] * s2)/Fb - (2.0* parms[3]*Fa)/(Fb*Fb)
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'PBE2' : # \cite{gotz2009performance} (23)
        if not parms : parms = [0.2942, 2.0309]
        ss = s/tkf0
        s2 = ss * ss
        Fa = parms[1] * s2 
        Fb = 1.0 + parms[0] * s2
        F = 1.0 + Fa/Fb
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[1] / (Fb * Fb)
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'PBE3' : # \cite{gotz2009performance} (23)
        if not parms : parms = [4.1355, -3.7425, 50.258]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fb = 1.0 + parms[0] * s2
        Fb2 = Fb * Fb
        F = 1.0 + parms[1] * s2 / Fb + parms[2] * s4 / Fb2
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[1] / Fb2 + 4 * parms[2] * s2 / (Fb2 * Fb)
            dFds2 /= (tkf0 * tkf0)

    elif functional == 'PBE4' : # \cite{gotz2009performance} (23)
        if not parms : parms = [1.7107,-7.2333,61.645,-93.683]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        Fb = 1.0 + parms[0] * s2
        Fb2 = Fb * Fb
        Fb3 = Fb * Fb * Fb
        F = 1.0 + parms[1] * s2 / Fb + parms[2] * s4 / Fb2 + parms[3] * s6 / Fb3
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[1] / Fb2 + 4 * parms[2] * s2 / (Fb3) + 4 * parms[3] * s4 / (Fb3 * Fb)
            dFds2 /= (tkf0 * tkf0)

    elif functional == 'P82' : # \cite{hfofke} (9)
        if not parms : parms = [5.0/27.0]
        ss = s/tkf0
        s2 = ss * ss
        s6 = s2 * s2 * s2
        Fb = 1 + s6
        F = 1.0 + parms[0] * s2/Fb
        if calcType != 'Energy' :
            dFds2 = parms[0] * (2.0/Fb - 6.0 * s6/(Fb * Fb))
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'TW02' : # \cite{hfofke} (20)
        if not parms : parms = [0.8438, 0.27482816]
        ss = s/tkf0
        s2 = ss * ss
        Fa = parms[1] * s2 
        Fb = 1.0 + parms[1] * s2
        F = 1.0 + parms[0]-parms[0]/Fb
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[0]* parms[1] / (Fb * Fb)
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'APBEK' : # \cite{hfofke} (32)
        if not parms : parms = [0.23889, 0.804]
        ss = s/tkf0
        s2 = ss * ss
        Fa = parms[0] * s2 
        Fb = 1.0 + parms[0]/parms[1] * s2
        F = 1.0 + Fa/Fb
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[0] / (Fb * Fb)
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'REVAPBEK' or  functional == 'revAPBEK' or  functional == 'REVAPBE' : # \cite{hfofke} (33)
        if not parms : parms = [0.23889, 1.245]
        ss = s/tkf0
        s2 = ss * ss
        Fa = parms[0] * s2 
        Fb = 1.0 + parms[0]/parms[1] * s2
        F = 1.0 + Fa/Fb
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[0] / (Fb * Fb)
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'VJKS00' : # \cite{hfofke} (18) !something wrong
        if not parms : parms = [0.8944,0.6511,0.0431]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        # Fa = 1.0 + parms[0] * s2 - parms[2] * s6
        Fa = 1.0 + parms[0] * s2 
        Fb = 1.0 + parms[1] * s2 + parms[2] * s4
        F = Fa/Fb
        if calcType != 'Energy' :
            # dFds2 = (2.0 * parms[0] - 6.0 * parms[2] * s4)/ Fb - (2.0 * parms[1] + 4.0 * parms[2] * s2)*Fa/(Fb*Fb)
            dFds2 = (2.0 * parms[0])/ Fb - (2.0 * parms[1] + 4.0 * parms[2] * s2)*Fa/(Fb*Fb)
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'LC94' : # \cite{hfofke} (16) # same as PW91
        if not parms : parms = [0.093907,0.26608, 0.0809615, 100.0, 76.32, 0.000057767]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = 1.0 + parms[0] * ss * np.arcsinh(parms[4] * ss) + (parms[1]-parms[2] * np.exp(-parms[3] * s2)) * s2
        Fb = 1.0 + parms[0] * ss * np.arcsinh(parms[4] * ss) + parms[5] * s4
        F = Fa/Fb
        if calcType != 'Energy' :
            Fa_s = parms[0] * parms[4] * ss/np.sqrt(parms[4] ** 2 * s2 + 1)  + parms[0] * np.arcsinh(parms[4] * ss) + 2.0 * ss * (parms[1] + 2.0 * parms[2] * parms[3] * s2 * np.exp(-parms[3] * s2) -2.0 * parms[2] * np.exp(-parms[3] * s2))

            dFds2 = Fa_s/Fb - Fa/(Fb * Fb) * ( parms[0] * parms[4] * ss/np.sqrt(parms[4] ** 2 * s2 + 1) + \
		    (parms[0] * np.arcsinh(parms[4] * ss) + 4.0*parms[5]*s2*ss))

            mask = s > tol2
            dFds2[mask] /= ss[mask]
            dFds2 /= (tkf0 * tkf0)
    
    elif functional == 'VT84F' : # \cite{hfofke} (33)
        if not parms : parms = [2.777028126, 2.777028126-40.0/27.0]
        ss = s/tkf0
        s2 = ss * ss
        s2[s2 < tol2] = tol2
        s4 = s2 * s2
        F = 1.0 + 5.0/3.0 * s2 + parms[0] * s2 * np.exp(-parms[1] * s2)/(1 + parms[0] * s2)+\
                (1 - np.exp(-parms[1] * s4)) * (1.0/s2 - 1.0)
        if calcType != 'Energy' :
            dFds2 = 10.0/3.0 + 2.0*parms[0] * np.exp(-parms[1] * s2)*(1.0-parms[1] * s2)/(1 + parms[0] * s2)-\
                    2.0*parms[0] *s2 * np.exp(-parms[1] * s2) / (1 + parms[0] * s2)**2 + \
                    4.0*parms[1]*s2*(1.0/s2 - 1.0)*np.exp(-parms[1] * s4)-\
                    2.0*(1 - np.exp(-parms[1] * s4))/(s4)

            dFds2 /= (tkf0 * tkf0)

    elif functional == 'SMP19' : 
        if not parms : parms = [33873.81423879, -1044.95674204, 33885.39245476, 27374.01470582, 3523.63738304]
        ss = s/tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = parms[0] + parms[1] * s2 
        Fb = parms[2] + parms[3] * s2 + parms[4] * s4
        F = Fa/Fb + 5.0/3.0 * s2
        if calcType != 'Energy' :
            dFds2 = 2.0 * parms[1]/Fb - (parms[0]+parms[1] * s2) * (2 * parms[3]  + 4 * parms[4] * s2)/(Fb * Fb) + 10.0/3.0
            dFds2 /= (tkf0 * tkf0)

    return F, dFds2

def GGA(rho, functional = 'LKT', calcType = 'Both', split = False, **kwargs):
    rhom = rho.copy()
    tol = 1E-16
    rhom[rhom < tol] = tol

    rho23 = rhom ** (2.0/3.0)
    rho53 = rho23 * rhom
    cTF = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)
    tkf0 = 2.0 * (3.0 * np.pi**2)**(1.0/3.0)
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
    return OutFunctional
