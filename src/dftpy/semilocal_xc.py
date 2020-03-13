# Drivers for LibXC

import numpy as np
from dftpy.field import DirectField
from dftpy.functional_output import Functional
from dftpy.constants import MATHLIB
from dftpy.math_utils import TimeData


def CheckLibXC():
    import importlib

    islibxc = importlib.util.find_spec("pylibxc")
    found = islibxc is not None
    if not found:
        raise ModuleNotFoundError("Install LibXC and pylibxc to use this functionality")
    return found


def Get_LibXC_Input(density, do_sigma=True):
    inp = {}
    if density.rank > 1 :
        rhoT = density.reshape((2, -1)).T
        inp["rho"] = rhoT.ravel()
    else :
        inp["rho"] = density.ravel()
    if do_sigma:
        # sigma = density.sigma()
        sigma = density.sigma("standard")
        if density.rank > 1 :
            sigma = sigma.reshape((3, -1)).T
        inp["sigma"] = sigma.ravel()
    return inp


def Get_LibXC_Output(out, density):
    if not isinstance(out, (dict)):
        raise TypeError("LibXC output must be a dictionary")

    OutFunctional = Functional(name="LibXC")

    for key in ["vrho", "v2rho2", "v3rho3", "v4rho4"]:
        if key in out.keys():
            if density.rank > 1 :
                v = out[key].reshape((-1, 2)).T
                v = DirectField(density.grid, rank=density.rank, griddata_3d=v)
            else :
                v = DirectField(density.grid, rank=density.rank, griddata_3d=out[key])
            if key == "vrho":
                OutFunctional.potential = v
            else:
                setattr(OutFunctional, key, v)

    if "vsigma" in out.keys():
        if density.rank > 1 :
            vsigma = out["vsigma"].reshape((-1, 3)).T
            vsigma = DirectField(density.grid, rank=3, griddata_3d=vsigma)
        else :
            vsigma = DirectField(density.grid, griddata_3d=out["vsigma"].reshape(np.shape(density)))
        do_sigma = True
    else :
        do_sigma = False

    if hasattr(OutFunctional, 'v2rho2'):
        raise Exception('2nd and higher order derivative for GGA functionals has not implemented yet.')

    if do_sigma and hasattr(OutFunctional, 'potential'):
        if density.rank > 1 :
            grhoU = density[0].gradient(flag="standard")
            grhoD = density[1].gradient(flag="standard")
            prodotto = vsigma[0] * grhoU
            v00=prodotto.divergence(flag="standard")
            prodotto = vsigma[1] * grhoD
            v01=prodotto.divergence(flag="standard")
            prodotto = vsigma[1] * grhoU
            v10=prodotto.divergence(flag="standard")
            prodotto = vsigma[2] * grhoD
            v11=prodotto.divergence(flag="standard")
            OutFunctional.potential[0] -= 2 * v00+v01
            OutFunctional.potential[1] -= 2 * v11+v10
        else :
            grho = density.gradient(flag="standard")
            prodotto = vsigma * grho
            vsigma_last = prodotto.divergence(flag="standard")
            OutFunctional.potential -= 2 * vsigma_last

    if "zk" in out.keys():
        if density.rank > 1 :
            rho = np.sum(density, axis = 0)
        else :
            rho = density
        edens = out["zk"].reshape(np.shape(rho))
        ene = np.einsum("ijk, ijk->", edens, rho) * density.grid.dV
        OutFunctional.energy = ene

    return OutFunctional


def XC(density, x_str='lda_x', c_str='lda_c_pz', polarization='unpolarized', do_sigma=False, calcType=["E","V"], **kwargs):
    TimeData.Begin("XC")
    if CheckLibXC():
        from pylibxc.functional import LibXCFunctional
    """
     Output: 
        - Functional_XC: a XC functional evaluated with LibXC
     Input:
        - density: a DirectField (rank=1)
        - x_str,c_str: strings like "gga_x_pbe" and "gga_c_pbe"
        - polarization: string like "polarized" or "unpolarized"
    """
    if not isinstance(x_str, str):
        raise AttributeError(
            "x_str and c_str must be LibXC functionals. Check pylibxc.util.xc_available_functional_names()"
        )
    if not isinstance(c_str, str):
        raise AttributeError(
            "x_str and c_str must be LibXC functionals. Check pylibxc.util.xc_available_functional_names()"
        )
    if not isinstance(polarization, str):
        raise AttributeError("polarization must be a ``polarized`` or ``unpolarized``")
    if not isinstance(density, (DirectField)):
        raise AttributeError("density must be a rank-1 PBCpy DirectField")
    func_x = LibXCFunctional(x_str, polarization)
    func_c = LibXCFunctional(c_str, polarization)
    inp = Get_LibXC_Input(density, do_sigma=do_sigma)
    kargs = {'do_exc': False, 'do_vxc': False}
    if 'E' in calcType:
        kargs.update({'do_exc': True})
    if 'V' in calcType:
        kargs.update({'do_vxc': True})
    if 'V2' in calcType:
        kargs.update({'do_fxc': True})
    if 'V3' in calcType:
        kargs.update({'do_kxc': True})
    if 'V4' in calcType:
        kargs.update({'do_lxc': True})
    out_x = func_x.compute(inp, **kargs)
    out_c = func_c.compute(inp, **kargs)
    Functional_X = Get_LibXC_Output(out_x, density)
    Functional_C = Get_LibXC_Output(out_c, density)
    Functional_XC = Functional_X.sum(Functional_C)
    name = x_str[6:] + "_" + c_str[6:]
    Functional_XC.name = name.upper()
    TimeData.End("XC")
    return Functional_XC


def PBE(density, polarization="unpolarized", calcType=["E","V"]):
    if density.rank > 1 :
        polarization = 'polarized'
    return XC(
        density=density,
        x_str="gga_x_pbe",
        c_str="gga_c_pbe",
        polarization=polarization,
        do_sigma=True,
        calcType=calcType,
    )


def LDA_XC(density, polarization="unpolarized", calcType=["E","V"]):
    return XC(
        density=density, x_str="lda_x", c_str="lda_c_pz", polarization=polarization, do_sigma=False, calcType=calcType
    )


def LDA(rho, polarization="unpolarized", calcType=["E","V"], **kwargs):
    if rho.rank > 1 :
        polarization = 'polarized'
    if polarization != 'unpolarized' :
        return LDA_XC(rho,polarization, calcType)
    TimeData.Begin("LDA")
    OutFunctional = Functional(name="XC")
    a = (0.0311, 0.01555)
    b = (-0.048, -0.0269)
    c = (0.0020, 0.0007)
    d = (-0.0116, -0.0048)
    gamma = (-0.1423, -0.0843)
    beta1 = (1.0529, 1.3981)
    beta2 = (0.3334, 0.2611)

    rho_cbrt = np.cbrt(rho)
    Rs = np.cbrt(3.0 / (4.0 * np.pi)) / rho_cbrt
    rs1 = Rs < 1
    rs2 = Rs >= 1
    Rs2sqrt = np.sqrt(Rs[rs2])

    if "E" in calcType:
        ExRho = -3.0 / 4.0 * np.cbrt(3.0 / np.pi) * rho_cbrt
        ExRho[rs1] += a[0] * np.log(Rs[rs1]) + b[0] + c[0] * Rs[rs1] * np.log(Rs[rs1]) + d[0] * Rs[rs1]
        ExRho[rs2] += gamma[0] / (1.0 + beta1[0] * Rs2sqrt + beta2[0] * Rs[rs2])
        ene = np.einsum("ijk, ijk->", ExRho, rho) * rho.grid.dV
        OutFunctional.energy = ene
    if "V" in calcType:
        pot = np.cbrt(-3.0 / np.pi) * rho_cbrt
        pot[rs1] += (
            np.log(Rs[rs1]) * (a[0] + 2.0 / 3 * c[0] * Rs[rs1])
            + b[0]
            - 1.0 / 3 * a[0]
            + 1.0 / 3 * (2 * d[0] - c[0]) * Rs[rs1]
        )
        pot[rs2] += (
            gamma[0] + (7.0 / 6.0 * gamma[0] * beta1[0]) * Rs2sqrt + (4.0 / 3.0 * gamma[0] * beta2[0] * Rs[rs2])
        ) / (1.0 + beta1[0] * Rs2sqrt + beta2[0] * Rs[rs2]) ** 2
        OutFunctional.potential = pot
    if "V2" in calcType:
        fx = - np.cbrt(3.0 / np.pi) / 3.0 * np.cbrt(rho) /rho

        fc = np.empty(np.shape(rho))
        fc[rs1] = -a[0] / 3.0 - (c[0] / 9.0 * (np.log(Rs[rs1]) * 2.0 + 1.0) + d[0] * 2.0 / 9.0) * Rs[rs1]
        tmpa = beta1[0] * Rs2sqrt
        tmpb = beta2[0] * Rs[rs2]
        deno = 1.0 + tmpa + tmpb
        fc[rs2] = gamma[0] / 36.0 * ( 5.0 * tmpa + 7.0 * tmpa * tmpa + 8.0 * tmpb + 16.0 * tmpb * tmpb + 21.0 * tmpa * tmpb) / deno / deno / deno
        fc /= rho

        OutFunctional.v2rho2 = fx + fc


    TimeData.End("LDA")
    return OutFunctional


def LDAStress(rho, polarization="unpolarized", energy=None):
    TimeData.Begin("LDA_Stress")
    if rho.rank > 1 :
        polarization = 'polarized'
    if energy is None:
        EnergyPotential = LDA(rho, polarization, calcType=["E","V"])
        energy = EnergyPotential.energy
        potential = EnergyPotential.potential
    else:
        potential = LDA(rho, polarization, calcType=["V"]).potential
    stress = np.zeros((3, 3))
    Etmp = energy - np.einsum("..., ...-> ", potential, rho, optimize = 'optimal') * rho.grid.dV
    for i in range(3):
        stress[i, i] = Etmp / rho.grid.volume
    TimeData.End("LDA_Stress")
    return stress


def LIBXC_KEDF(density, polarization="unpolarized", k_str="gga_k_lc94", calcType=["E","V"], **kwargs):
    if CheckLibXC():
        from pylibxc.functional import LibXCFunctional
    """
     Output: 
        - Functional_KEDF: a KEDF functional evaluated with LibXC
     Input:
        - density: a DirectField (rank=1)
        - k_str: strings like "gga_k_lc94"
        - polarization: string like "polarized" or "unpolarized"
    """
    if not isinstance(k_str, str):
        raise AttributeError("k_str must be a LibXC functional. Check pylibxc.util.xc_available_functional_names()")
    if not isinstance(polarization, str):
        raise AttributeError("polarization must be a ``polarized`` or ``unpolarized``")
    if not isinstance(density, (DirectField)):
        raise AttributeError("density must be a rank-1 PBCpy DirectField")
    func_k = LibXCFunctional(k_str, polarization)
    inp = Get_LibXC_Input(density)
    kargs = {'do_exc': False, 'do_vxc': False}
    if 'E' in calcType:
        kargs.update({'do_exc': True})
    if 'V' in calcType:
        kargs.update({'do_vxc': True})
    if 'V2' in calcType:
        kargs.update({'do_fxc': True})
    if 'V3' in calcType:
        kargs.update({'do_kxc': True})
    if 'V4' in calcType:
        kargs.update({'do_lxc': True})
    out_k = func_k.compute(inp, **kargs)
    Functional_KEDF = Get_LibXC_Output(out_k, density)
    name = k_str[6:]
    Functional_KEDF.name = name.upper()
    return Functional_KEDF
