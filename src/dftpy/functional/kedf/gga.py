# Collection of semilocal functionals

import numpy as np
import scipy.special as sp

from dftpy.functional.functional_output import FunctionalOutput

__all__ = ["GGA", "GGAFs", "GGA_KEDF_list", "GGAStress"]

GGA_KEDF_list = {
    "LKT"           : [1.3, 1.0],  # \cite{luo2018simple}
    "DK"            : [0.95, 14.28111, -19.5762, -0.05, 9.99802, 2.96085],  # \cite{garcia2007kinetic} (8)
    "LLP"           : [0.0044188, 0.0253], # \cite{garcia2007kinetic} (9)[!x] \cite{gotz2009performance} (18)
    "LLP91"         : [0.0044188, 0.0253], # \cite{garcia2007kinetic} (9)[!x] \cite{gotz2009performance} (18)
    "OL1"           : [0.00677],  # \cite{gotz2009performance} (16)
    "OL"            : [0.00677],  # \cite{gotz2009performance} (16)
    "OL2"           : [0.0887],  # \cite{gotz2009performance} (17)
    "T92"           : [0.0055, 0.0253, 0.072], # \cite{garcia2007kinetic} (12),\cite{gotz2009performance} (22), \cite{hfofke} (15)[!x]
    "THAK"          : [0.0055, 0.0253, 0.072], # \cite{garcia2007kinetic} (12),\cite{gotz2009performance} (22), \cite{hfofke} (15)[!x]
    "B86A"          : [0.0039, 0.004],  # \cite{garcia2007kinetic} (13)
    "B86"           : [0.0039, 0.004],  # \cite{garcia2007kinetic} (13)
    "B86B"          : [0.00403, 0.007],  # \cite{garcia2007kinetic} (14)
    "DK87"          : [7.0 / 324.0 / (18.0 * np.pi ** 4) ** (1.0 / 3.0), 0.861504, 0.044286],  # \cite{garcia2007kinetic} (15)
    "PW86"          : [1.296, 14.0, 0.2],  # \cite{gotz2009performance} (19)
    "PW91O"         : [0.093907, 0.26608, 0.0809615, 100.0, 76.320, 0.57767e-4],  # (A1, A2, A3, A4, A, B1)  # \cite{gotz2009performance} (20)
    "PW91"          : [0.19645, 0.2747, 0.1508, 100.0, 7.7956, 0.004],  # (A1, A2, A3, A4, A, B1) # \cite{lacks1994tests} (16) and \cite{garcia2007kinetic} (17)[!x]
    "PW91k"         : [0.19645, 0.2747, 0.1508, 100.0, 7.7956, 0.004],  # (A1, A2, A3, A4, A, B1) # \cite{lacks1994tests} (16) and \cite{garcia2007kinetic} (17)[!x]
    "LG94"          : [(1e-8 + 0.1234) / 0.024974, 29.790, 22.417, 12.119, 1570.1, 55.944, 0.024974], # a2, a4, a6, a8, a10, a12, b # \cite{garcia2007kinetic} (18)
    "E00"           : [135.0, 28.0, 5.0, 3.0],  # \cite{gotz2009performance} (14)
    "P92"           : [1.0, 88.3960, 16.3683, 88.2108],  # \cite{gotz2009performance} (15)
    "PBE2"          : [0.2942, 2.0309],  # \cite{gotz2009performance} (23)
    "PBE3"          : [4.1355, -3.7425, 50.258],  # \cite{gotz2009performance} (23)
    "PBE4"          : [1.7107, -7.2333, 61.645, -93.683],  # \cite{gotz2009performance} (23)
    "P82"           : [5.0 / 27.0],  # \cite{hfofke} (9)
    "TW02"          : [0.8438, 0.27482816],  # \cite{hfofke} (20)
    "APBE"          : [0.23889, 0.804],  # \cite{hfofke} (32)
    "APBEK"         : [0.23889, 0.804],  # \cite{hfofke} (32)
    "REVAPBEK"      : [0.23889, 1.245], # \cite{hfofke} (33)
    "REVAPBE"       : [0.23889, 1.245],  # \cite{hfofke} (33)
    "VJKS00"        : [0.8944, 0.6511, 0.0431],  # \cite{hfofke} (18) !something wrong
    "LC94"          : [0.093907, 0.26608, 0.0809615, 100.0, 76.32, 0.000057767],  # \cite{hfofke} (16) # same as PW91
    "VT84F"         : [2.777028126, 2.777028126 - 40.0 / 27.0],  # \cite{hfofke} (33)
    "LKT-PADE46"    : [1.3],
    "LKT-PADE46-S"  : [1.3, 0.01],
    "SMP"           : [1.0],  # test functional
    "TF"            : [1.0],
    "VW"            : [1.0],
    "X_TF_Y_VW"     : [1.0, 1.0],
    "TFVW"          : [1.0, 1.0],
    "STV"           : [1.0, 1.0, 0.01, 1.0],
    "PBE2M"         : [1.0, 0.2942, 2.0309],
    "TEST-TF-APBEK" : [1.3, 0.23889, 1.245],
    }


def GGAStress(rho, functional="LKT", energy=None, potential=None, dFds2=None, **kwargs):
    """
    Not finished.
    """
    rhom = rho.copy()
    tol = 1e-16
    rhom[rhom < tol] = tol

    rho23 = rhom ** (2.0 / 3.0)
    rho53 = rho23 * rhom
    rho43 = rho23 * rho23
    rho83 = rho43 * rho43
    cTF = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    ckf2 = (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    tf = cTF * rho53
    vkin2 = tf * ckf2 * dFds2 / rho83
    dRho_ij = []
    g = rho.grid.get_reciprocal().g
    rhoG = rho.fft()
    for i in range(3):
        dRho_ij.append((1j * g[i] * rhoG).ifft(force_real=True))
    stress = np.zeros((3, 3))

    if potential is None:
        gga = GGA(rho, functional=functional, calcType={"E", "V"}, **kwargs)
        energy = gga.energy
        potential = gga.potential

    rhoP = np.einsum("ijk, ijk", rho, potential)
    for i in range(3):
        for j in range(i, 3):
            stress[i, j] = np.einsum("ijk, ijk", vkin2, dRho_ij[i] * dRho_ij[j])
            if i == j:
                stress[i, j] += energy - rhoP
            stress[j, i] = stress[i, j]


def GGAFs(s, functional="LKT", calcType={"E", "V"}, params=None, gga_remove_vw=None, **kwargs):
    r"""
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
          title={Performance of kinetic energy functional for interaction energies in a subsystem formulation of density functional theory}}
        @article{lacks1994tests,
          title = {Tests of nonlocal kinetic energy functionals}}
        @misc{hfofke,
          url = {http://www.qtp.ufl.edu/ofdft/research/KE_refdata_27ii18/Explanatory_Post_HF_OFKE.pdf}}
        @article{xia2015single,
          title={Single-point kinetic energy density functionals: A pointwise kinetic energy density analysis and numerical convergence investigation}}
        @article{luo2018simple,
          title={A simple generalized gradient approximation for the noninteracting kinetic energy density functional},
    """
    cTF = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    tkf0 = 2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    b = 2 ** (1.0 / 3.0)
    tol2 = 1e-8  # It's a very small value for safe deal with 1/s
    F = np.empty_like(s)
    dFds2 = np.empty_like(s)  # Actually, it's 1/s*dF/ds

    functional = functional.upper()
    if functional not in GGA_KEDF_list:
        raise AttributeError("%s GGA KEDF to be implemented" % functional)

    params0 = GGA_KEDF_list[functional]

    if params is None:
        pass
    elif isinstance(params, (float, int)):
        params0[0] = params
    else:
        l = min(len(params), len(params0))
        params0[:l] = params[:l]

    params = params0
    if functional == "LKT":  # \cite{luo2018simple}
        ss = s / tkf0
        s2 = ss * ss
        mask1 = ss > 100.0
        mask2 = ss < 1e-5
        mask = np.invert(np.logical_or(mask1, mask2))
        F[mask] = 1.0 / np.cosh(params[0] * ss[mask]) + 5.0 / 3.0 * (s2[mask]) * params[1]
        F[mask1] = 5.0 / 3.0 * (s2[mask1]) * params[1]
        F[mask2] = (
                1.0 + (5.0 / 3.0 * params[1] - 0.5 * params[0] ** 2) * s2[mask2] + 5.0 / 24.0 * params[0] ** 4 * s2[
            mask2] ** 2
        )  # - 61.0/720.0 * params[0] ** 6 * s2[mask2] ** 3
        if "V" in calcType:
            dFds2[mask] = (
                    10.0 / 3.0 * params[1] - params[0] * np.sinh(params[0] * ss[mask]) / np.cosh(
                params[0] * ss[mask]) ** 2 / ss[mask]
            )
            dFds2[mask1] = 10.0 / 3.0 * params[1]
            dFds2[mask2] = (
                    10.0 / 3.0 * params[1]
                    - params[0] ** 2
                    + 5.0 / 6.0 * params[0] ** 4 * s2[mask2]
                    - 61.0 / 120.0 * params[0] ** 6 * s2[mask2] ** 2
            )
            dFds2 /= tkf0 ** 2

    elif functional == "DK":  # \cite{garcia2007kinetic} (8)
        x = s * s / (72 * cTF)
        Fa = 9.0 * params[5] * x ** 4 + params[2] * x ** 3 + params[1] * x ** 2 + params[0] * x + 1.0
        Fb = params[5] * x ** 3 + params[4] * x ** 2 + params[3] * x + 1.0
        F = Fa / Fb
        if "V" in calcType:
            dFds2 = (36.0 * params[5] * x ** 3 + 3 * params[2] * x ** 2 + 2 * params[1] * x + params[0]) / Fb - Fa / (
                    Fb * Fb
            ) * (3.0 * params[5] * x ** 2 + 2.0 * params[4] * x + params[3])
            dFds2 /= 36.0 * cTF

    elif (
            functional == "LLP" or functional == "LLP91"
    ):  # \cite{garcia2007kinetic} (9)[!x] \cite{gotz2009performance} (18)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2
        Fb = 1.0 + params[1] * bs * np.arcsinh(bs)
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] / Fb - (
                    params[0] * params[1] * bs * np.arcsinh(bs) + Fa * params[1] / np.sqrt(1.0 + bs2)
            ) / (Fb * Fb)
            dFds2 *= b * b

    elif functional == "OL1" or functional == "OL":  # \cite{gotz2009performance} (16)
        F = 1.0 + s * s / 72.0 / cTF + params[0] / cTF * s
        if "V" in calcType:
            mask = s > tol2
            dFds2[:] = 1.0 / 36.0 / cTF
            dFds2[mask] += params[0] / cTF / s[mask]

    elif functional == "OL2":  # \cite{gotz2009performance} (17)
        F = 1.0 + s * s / 72.0 / cTF + params[0] / cTF * s / (1 + 4 * s)
        if "V" in calcType:
            mask = s > tol2
            dFds2[:] = 1.0 / 36.0 / cTF
            dFds2[mask] += params[0] / cTF / (1 + 4 * s[mask]) ** 2 / s[mask]

    elif (
            functional == "T92" or functional == "THAK"
    ):  # \cite{garcia2007kinetic} (12),\cite{gotz2009performance} (22), \cite{hfofke} (15)[!x]
        bs = b * s
        bs2 = bs * bs
        F = (
                1.0
                + params[0] * bs2 / (1.0 + params[1] * bs * np.arcsinh(bs))
                - params[2] * bs / (1.0 + 2 ** (5.0 / 3.0) * bs)
        )
        # F = 1.0 + params[0] * bs2 / (1.0 + params[1] * bs * np.arcsinh(bs)) - params[2] * bs / (1.0+ 2**2 * bs)
        if "V" in calcType:
            mask = s > tol2
            Fb = (1.0 + params[1] * bs * np.arcsinh(bs)) ** 2
            dFds2 = (
                            -(params[0] * params[1] * bs2) / np.sqrt(1 + bs2)
                            + (params[0] * params[1] * bs * np.arcsinh(bs))
                            + 2.0 * params[0]
                    ) / Fb
            dFds2[mask] -= params[2] / (1.0 + 2 ** (5.0 / 3.0) * bs[mask]) ** 2 / bs[mask]
            # dFds2[mask] -= params[2]/(1.0+4* bs[mask]) ** 2/bs[mask]
            dFds2 *= b * b

    elif functional == "B86A" or functional == "B86":  # \cite{garcia2007kinetic} (13)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2
        Fb = 1.0 + params[1] * bs2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2 * params[0] / (Fb * Fb) * (b * b)

    elif functional == "B86B":  # \cite{garcia2007kinetic} (14)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2
        Fb = (1.0 + params[1] * bs2) ** (4.0 / 5.0)
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = (2 * params[0] * (params[1] * bs2 + 5.0)) / (5 * (1.0 + params[1] * bs2) * Fb) * (b * b)

    elif functional == "DK87":  # \cite{garcia2007kinetic} (15)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2 * (1 + params[1] * bs)
        Fb = 1.0 + params[2] * bs2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = params[0] * (2.0 + 3.0 * params[1] * bs + params[1] * params[2] * bs2 * bs) / (Fb * Fb) * (b * b)

    elif functional == "PW86":  # \cite{gotz2009performance} (19)
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s2 * s4
        Fa = 1.0 + params[0] * s2 + params[1] * s4 + params[2] * s6
        F = Fa ** (1.0 / 15.0)
        if "V" in calcType:
            dFds2 = 2.0 / 15.0 * (params[0] + 2 * params[1] * s2 + 3 * params[2] * s4) / Fa ** (14.0 / 15.0)
            dFds2 /= tkf0 * tkf0

    elif functional == "PW91O":  # \cite{gotz2009performance} (20)
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = (params[1] - params[2] * np.exp(-params[3] * s2)) * s2 - params[5] * s4
        Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            Fa_s2 = (params[1] - params[2] * np.exp(-params[3] * s2)) - params[5] * s2

            dFds2 = 2.0 * (
                    params[1] + (params[3] * s2 - 1) * params[2] * np.exp(-params[3] * s2) - 4.0 * params[5] * s2
            ) / Fb - Fa_s2 * ss / (Fb * Fb) * (
                            (params[0] * params[4] * ss) / (params[4] ** 2 * s2 + 1)
                            + params[0] * np.arcsinh(params[4] * ss)
                            + 4.0 * params[5] * s2
                    )
            dFds2 /= tkf0 * tkf0

    elif (
            functional == "PW91" or functional == "PW91k"
    ):  # \cite{lacks1994tests} (16) and \cite{garcia2007kinetic} (17)[!x]
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + (params[1] - params[2] * np.exp(-params[3] * s2)) * s2
        Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
        F = Fa / Fb
        if "V" in calcType:
            Fa_s = (
                    params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                    + params[0] * np.arcsinh(params[4] * ss)
                    + 2.0
                    * ss
                    * (
                            params[1]
                            + 2.0 * params[2] * params[3] * s2 * np.exp(-params[3] * s2)
                            - 2.0 * params[2] * np.exp(-params[3] * s2)
                    )
            )

            dFds2 = Fa_s / Fb - Fa / (Fb * Fb) * (
                    params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                    + (params[0] * np.arcsinh(params[4] * ss) + 4.0 * params[5] * s2 * ss)
            )

            mask = s > tol2
            dFds2[mask] /= ss[mask]
            dFds2 /= tkf0 * tkf0

    elif functional == "LG94":  # \cite{garcia2007kinetic} (18)
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        s8 = s4 * s4
        s10 = s4 * s6
        s12 = s6 * s6
        Fa = 1.0 + params[0] * s2 + params[1] * s4 + params[2] * s6 + params[3] * s8 + params[4] * s10 + params[5] * s12
        Fb = 1.0 + 1e-8 * s2
        F = (Fa / Fb) ** params[6]
        if "V" in calcType:
            dFds2 = (
                    -2
                    * params[6]
                    * F
                    * (
                            1e-8 * Fa
                            - Fb
                            * (
                                    params[0]
                                    + 2 * params[1] * s2
                                    + 3 * params[2] * s4
                                    + 4 * params[3] * s6
                                    + 5 * params[4] * s8
                                    + 6 * params[5] * s10
                            )
                    )
                    / (Fa * Fb)
            )
            dFds2 /= tkf0 * tkf0

    elif functional == "E00":  # \cite{gotz2009performance} (14)
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = params[0] + params[1] * s2 + params[2] * s4
        Fb = params[0] + params[3] * s2
        F = Fa / Fb
        if "V" in calcType:
            dFds2 = (2.0 * params[1] + 4.0 * params[2] * s2) / Fb - (2.0 * params[3] * Fa) / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "P92":  # \cite{gotz2009performance} (15)
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = params[0] + params[1] * s2 + params[2] * s4
        Fb = params[0] + params[3] * s2
        F = Fa / Fb
        if "V" in calcType:
            dFds2 = (2.0 * params[1] + 4.0 * params[2] * s2) / Fb - (2.0 * params[3] * Fa) / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "PBE2":  # \cite{gotz2009performance} (23)
        ss = s / tkf0
        s2 = ss * ss
        Fa = params[1] * s2
        Fb = 1.0 + params[0] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[1] / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "PBE3":  # \cite{gotz2009performance} (23)
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fb = 1.0 + params[0] * s2
        Fb2 = Fb * Fb
        F = 1.0 + params[1] * s2 / Fb + params[2] * s4 / Fb2
        if "V" in calcType:
            dFds2 = 2.0 * params[1] / Fb2 + 4 * params[2] * s2 / (Fb2 * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "PBE4":  # \cite{gotz2009performance} (23)
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        Fb = 1.0 + params[0] * s2
        Fb2 = Fb * Fb
        Fb3 = Fb * Fb * Fb
        F = 1.0 + params[1] * s2 / Fb + params[2] * s4 / Fb2 + params[3] * s6 / Fb3
        if "V" in calcType:
            dFds2 = 2.0 * params[1] / Fb2 + 4 * params[2] * s2 / (Fb3) + 6 * params[3] * s4 / (Fb3 * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "P82":  # \cite{hfofke} (9)
        ss = s / tkf0
        s2 = ss * ss
        s6 = s2 * s2 * s2
        Fb = 1 + s6
        F = 1.0 + params[0] * s2 / Fb
        if "V" in calcType:
            dFds2 = params[0] * (2.0 / Fb - 6.0 * s6 / (Fb * Fb))
            dFds2 /= tkf0 * tkf0

    elif functional == "TW02":  # \cite{hfofke} (20)
        ss = s / tkf0
        s2 = ss * ss
        Fa = params[1] * s2
        Fb = 1.0 + params[1] * s2
        F = 1.0 + params[0] - params[0] / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] * params[1] / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "APBE" or functional == "APBEK":  # \cite{hfofke} (32)
        ss = s / tkf0
        s2 = ss * ss
        Fa = params[0] * s2
        Fb = 1.0 + params[0] / params[1] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "REVAPBEK" or functional == "REVAPBE":  # \cite{hfofke} (33)
        ss = s / tkf0
        s2 = ss * ss
        Fa = params[0] * s2
        Fb = 1.0 + params[0] / params[1] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "VJKS00":  # \cite{hfofke} (18) !something wrong
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        # Fa = 1.0 + params[0] * s2 - params[2] * s6
        Fa = 1.0 + params[0] * s2
        Fb = 1.0 + params[1] * s2 + params[2] * s4
        F = Fa / Fb
        if "V" in calcType:
            # dFds2 = (2.0 * params[0] - 6.0 * params[2] * s4)/ Fb - (2.0 * params[1] + 4.0 * params[2] * s2)*Fa/(Fb*Fb)
            dFds2 = (2.0 * params[0]) / Fb - (2.0 * params[1] + 4.0 * params[2] * s2) * Fa / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "LC94":  # \cite{hfofke} (16) # same as PW91
        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + (params[1] - params[2] * np.exp(-params[3] * s2)) * s2
        Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
        F = Fa / Fb
        if "V" in calcType:
            Fa_s = (
                    params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                    + params[0] * np.arcsinh(params[4] * ss)
                    + 2.0
                    * ss
                    * (
                            params[1]
                            + 2.0 * params[2] * params[3] * s2 * np.exp(-params[3] * s2)
                            - 2.0 * params[2] * np.exp(-params[3] * s2)
                    )
            )

            dFds2 = Fa_s / Fb - Fa / (Fb * Fb) * (
                    params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                    + (params[0] * np.arcsinh(params[4] * ss) + 4.0 * params[5] * s2 * ss)
            )

            mask = s > tol2
            dFds2[mask] /= ss[mask]
            dFds2 /= tkf0 * tkf0

    elif functional == "VT84F":  # \cite{hfofke} (33)
        ss = s / tkf0
        s2 = ss * ss
        s2[s2 < tol2] = tol2
        s4 = s2 * s2
        F = (
                1.0
                + 5.0 / 3.0 * s2
                + params[0] * s2 * np.exp(-params[1] * s2) / (1 + params[0] * s2)
                + (1 - np.exp(-params[1] * s4)) * (1.0 / s2 - 1.0)
        )
        if "V" in calcType:
            dFds2 = (
                    10.0 / 3.0
                    + 2.0 * params[0] * np.exp(-params[1] * s2) * (1.0 - params[1] * s2) / (1 + params[0] * s2)
                    - 2.0 * params[0] * s2 * np.exp(-params[1] * s2) / (1 + params[0] * s2) ** 2
                    + 4.0 * params[1] * s2 * (1.0 / s2 - 1.0) * np.exp(-params[1] * s4)
                    - 2.0 * (1 - np.exp(-params[1] * s4)) / (s4)
            )

            dFds2 /= tkf0 * tkf0

    elif functional == "LKT-PADE46":
        coef = [131040, 3360, 34, 62160, 3814, 59]
        # (131040 - 3360 *x**2 + 34 *x**4)/(131040 + 62160 *x**2 + 3814 *x**4 + 59 *x**6)
        coef[1] *= params[0] ** 2
        coef[2] *= params[0] ** 4
        coef[3] *= params[0] ** 2
        coef[4] *= params[0] ** 4
        coef[5] *= params[0] ** 6

        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        Fa = coef[0] + coef[1] * s2 + coef[2] * s4
        Fb = coef[0] + coef[3] * s2 + coef[4] * s4 + coef[5] * s6
        F = Fa / Fb + 5.0 / 3.0 * s2
        if "V" in calcType:
            dFds2 = (
                    (2.0 * coef[1] + 4.0 * coef[2] * s2) / Fb
                    - Fa * (2 * coef[3] + 4 * coef[4] * s2 + 6 * coef[5] * s4) / (Fb * Fb)
                    + 10.0 / 3.0
            )
            dFds2 /= tkf0 * tkf0

    elif functional == "LKT-PADE46-S":
        coef = [131040, 3360, 34, 62160, 3814, 59]
        coef[1] *= params[0] ** 2
        coef[2] *= params[0] ** 4
        coef[3] *= params[0] ** 2
        coef[4] *= params[0] ** 4
        coef[5] *= params[0] ** 6
        alpha = params[1]

        ss = s / tkf0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        ms = 1.0 / (1.0 + alpha * s2)
        Fa = coef[0] + coef[1] * s2 + coef[2] * s4
        Fb = coef[0] + coef[3] * s2 + coef[4] * s4 + coef[5] * s6
        F = Fa / Fb + 5.0 / 3.0 * s2 * ms
        if "V" in calcType:
            dFds2 = (
                    (2.0 * coef[1] + 4.0 * coef[2] * s2) / Fb
                    - Fa * (2 * coef[3] + 4 * coef[4] * s2 + 6 * coef[5] * s4) / (Fb * Fb)
                    + 10.0 / 3.0 * (ms - alpha * s2 * ms * ms)
            )
            dFds2 /= tkf0 * tkf0

    elif functional == "SMP":  # test functional
        ss = s / tkf0
        s2 = ss * ss
        F = 5.0 / 3.0 * s2 + sp.erfc(s2 / params[0])
        mask = ss > 1e-5
        # mask1 = np.invert(mask)
        if "V" in calcType:
            dFds2[:] = 10.0 / 3.0
            dFds2[mask] += 2 / np.sqrt(np.pi) * np.exp(-(s2[mask] / params[0]) ** 2) / ss[mask]
            dFds2 /= tkf0 ** 2

    elif functional == "TF":
        F = np.ones_like(s)
        dFds2 = np.zeros_like(F)

    elif functional == "VW":
        ss = s / tkf0
        s2 = ss * ss
        F = 5.0 / 3.0 * params[0] * s2
        if "V" in calcType:
            dFds2[:] = 10.0 / 3.0 * params[0]
            dFds2 /= tkf0 ** 2

    elif functional == "X_TF_Y_VW" or functional == "TFVW":
        ss = s / tkf0
        s2 = ss * ss
        F = params[0] + 5.0 / 3.0 * params[1] * s2
        if "V" in calcType:
            dFds2[:] = 10.0 / 3.0 * params[1]
            dFds2 /= tkf0 ** 2

    elif functional == "STV":
        ss = s / tkf0
        s2 = ss * ss
        Fb = params[3] + params[2] * s2
        F = params[0] + 5.0 / 3.0 * params[1] * s2 / Fb
        if "V" in calcType:
            dFds2[:] = 5.0 / 3.0 * params[1] * (2.0 / Fb - 2.0 * s2 * params[2] / (Fb * Fb))
            dFds2 /= tkf0 ** 2

    elif functional == "PBE2M":  #
        ss = s / tkf0
        s2 = ss * ss
        Fa = params[2] * s2
        Fb = params[0] + params[1] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[2] / (Fb * Fb)
            dFds2 /= tkf0 * tkf0

    elif functional == "TEST-TF-APBEK":
        ss = s / tkf0
        s2 = ss * ss

        Fx, dFds2 = _GGAFx(ss, s2, calcType=calcType, params=params, **kwargs)

        Fa = params[1] * s2
        Fb = 1.0 + params[1] / params[2] * s2
        F = 1.0 + Fa / Fb * (1 - Fx)
        if "V" in calcType:
            dFds2_rest = 2.0 * params[1] / (Fb * Fb)

            dFds2 = (1.0 - Fx) * dFds2_rest - dFds2 * Fa / Fb

            dFds2 /= tkf0 ** 2
    # -----------------------------------------------------------------------
    if gga_remove_vw is not None and gga_remove_vw:
        if isinstance(gga_remove_vw, (int, float)):
            pa = float(gga_remove_vw)
        else:
            pa = 1.0
        ss = s / tkf0
        s2 = ss * ss
        F -= 5.0 / 3.0 * s2 * pa
        if "V" in calcType:
            dFds2 -= 10.0 / 3.0 / tkf0 ** 2 * pa

    return F, dFds2


def _GGAFx(ss, s2, functional="LKT", calcType={"E", "V"}, params=None, **kwargs):
    if not params:
        params = [1.3]
    mask1 = ss > 100.0
    mask2 = ss < 1e-5
    mask = np.invert(np.logical_or(mask1, mask2))

    Fx = np.empty_like(ss)  # Interpolating function
    Fx[mask] = 1.0 / np.cosh(params[0] * ss[mask])
    Fx[mask1] = 0.0
    Fx[mask2] = 1.0 - 0.5 * params[0] ** 2 * s2[mask2] + 5.0 / 24.0 * params[0] ** 4 * s2[mask2] ** 2

    if "V" in calcType:
        dFds2 = np.empty_like(ss)  # Interpolating function
        dFds2[mask] = - params[0] * np.sinh(params[0] * ss[mask]) / np.cosh(params[0] * ss[mask]) ** 2 / ss[mask]
        dFds2[mask1] = 0.0
        dFds2[mask2] = (- params[0] ** 2 + 5.0 / 6.0 * params[0] ** 4 * s2[mask2]
                        - 61.0 / 120.0 * params[0] ** 6 * s2[mask2] ** 2)
    else:
        dFds2 = None

    return Fx, dFds2


def GGA(rho, functional="LKT", calcType={"E", "V"}, split=False, params=None, **kwargs):
    """
    Interface to compute GGAs internally to DFTpy.
    This is the default way, even though DFTpy can generate some of the GGAs with LibXC.
    Nota Bene: gradient and divergence is done brute force here and in a non-smooth way.
               while the LibXC implementation can be numerically smoothed by changing
               flag='standard' to flag='smooth'. The results with smooth math are
               slightly different.
    """
    sigma = kwargs.get('sigma', None)
    rhom = rho.copy()
    tol = 1e-16
    rhom[rhom < tol] = tol

    rho23 = rhom ** (2.0 / 3.0)
    rho53 = rho23 * rhom
    cTF = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    # tkf0 = 2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    tf = cTF * rho53
    rho43 = rho23 * rho23
    rho83 = rho43 * rho43
    q = rho.grid.get_reciprocal().q
    g = rho.grid.get_reciprocal().g

    rhoG = rho.fft()
    rhoGrad = []
    for i in range(3):
        # item = (1j * g[i] * rhoG).ifft(force_real=True)
        if sigma is None:
            grhoG = g[i] * rhoG * 1j
        else:
            grhoG = g[i] * rhoG * np.exp(-q * (sigma) ** 2 / 4.0) * 1j
        item = (grhoG).ifft(force_real=True)
        rhoGrad.append(item)
    s = np.sqrt(rhoGrad[0] ** 2 + rhoGrad[1] ** 2 + rhoGrad[2] ** 2) / rho43
    F, dFds2 = GGAFs(s, functional=functional, calcType=calcType, params=params, **kwargs)
    OutFunctional = FunctionalOutput(name="GGA-" + str(functional))

    if 'E' in calcType or 'D' in calcType:
        energydensity = tf * F
        if 'D' in calcType:
            OutFunctional.energydensity = energydensity
        OutFunctional.energy = energydensity.sum() * rhom.grid.dV

    if 'V' in calcType:
        pot = 5.0 / 3.0 * cTF * rho23 * F
        pot += -4.0 / 3.0 * tf * dFds2 * s * s / rhom

        p3 = []
        for i in range(3):
            item = tf * dFds2 * rhoGrad[i] / rho83
            p3.append(item.fft())
        pot3G = g[0] * p3[0] + g[1] * p3[1] + g[2] * p3[2]
        pot -= (1j * pot3G).ifft(force_real=True)
        OutFunctional.potential = pot
    # np.savetxt('gga.dat', np.c_[rho.ravel(), pot.ravel(), s.ravel(), F.ravel(), dFds2.ravel()])

    return OutFunctional

    # def laplacian(self, check_real = False, force_real = False, sigma = 0.025):


def get_gga_p(rho, calcType=["E", "V"], params=None, **kwargs):
    sigma = kwargs.get('sigma', None)
    rho53 = rho ** (5.0 / 3.0)
    tkf0 = 2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    p = rho.laplacian(sigma=sigma) / (tkf0 ** 2 * rho53)
    return p