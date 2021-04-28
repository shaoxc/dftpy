# Drivers for LibXC

import numpy as np
from dftpy.field import DirectField
from dftpy.functional_output import Functional
from dftpy.time_data import TimeData

class XC :
    def __init__(self, xc = 'LDA', core_density = None, libxc = True, **kwargs):
        self.options = {'xc': xc}
        self.options.update(kwargs)
        self._core_density = core_density
        if libxc :
            self.xcfun = LibXC
        else :
            self.xcfun = LDA

    @property
    def core_density(self):
        return self._core_density

    @core_density.setter
    def core_density(self, value):
        self._core_density = value

    def __call__(self, density, calcType={"E","V"}, **kwargs):
        return self.compute(density, calcType=calcType, **kwargs)

    def compute(self, density, calcType={"E","V"}, **kwargs):
        self.options.update(kwargs)
        core_density = self.core_density
        if core_density is None :
            new_density = density
        elif density.rank == core_density.rank :
            new_density = density + core_density
        elif density.rank == 2 and core_density.rank == 1 :
            new_density = density.copy()
            new_density[0] += 0.5 * core_density
            new_density[1] += 0.5 * core_density

        xc = self.options.get('xc', None)
        if xc == 'PBE' :
            xc_kwargs = {"x_str":"gga_x_pbe", "c_str":"gga_c_pbe"}
        elif xc == 'LDA' :
            xc_kwargs = {"x_str":"lda_x", "c_str":"lda_c_pz"}
        else :
            xc_kwargs = {}
        self.options.update(xc_kwargs)

        functional = self.xcfun(new_density, calcType = calcType, **self.options)
        return functional

def CheckLibXC():
    import importlib.util

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

    rank_dict = {
        "vrho": 2,
        "v2rho2": 3,
        "v3rho3": 4,
        "v4rho4": 5,
        "vsigma": 3,
        "v2rhosigma": 6,
        "v2sigma2": 6,
    }

    for key in ["vrho", "v2rho2", "v3rho3", "v4rho4"]:
        if key in out.keys():
            if density.rank > 1 :
                v = out[key].reshape((-1, rank_dict[key])).T
                v = DirectField(density.grid, rank=rank_dict[key], griddata_3d=v)
            else :
                v = DirectField(density.grid, rank=1, griddata_3d=out[key])
            if key == "vrho":
                OutFunctional.potential = v
            else:
                setattr(OutFunctional, key, v)

    vsigmas = {}
    for key in ["vsigma", "v2rhosigma", "v2sigma2"]:
        if key in out.keys():
            if density.rank > 1 :
                vsigmas[key] = out[key].reshape((-1, rank_dict[key])).T
                vsigmas[key] = DirectField(density.grid, rank=rank_dict[key], griddata_3d=vsigmas[key])
            else :
                vsigmas[key] = DirectField(density.grid, griddata_3d=out[key].reshape(np.shape(density)))

    if vsigmas:
        if density.rank > 1:
            grhoU = density[0].gradient(flag="standard")
            grhoD = density[1].gradient(flag="standard")
        else:
            grho = density.gradient(flag="standard")

        if hasattr(OutFunctional, 'potential'):
            if density.rank > 1 :
                prodotto = vsigmas['vsigma'][0] * grhoU
                v00=prodotto.divergence(flag="standard")
                prodotto = vsigmas['vsigma'][1] * grhoD
                v01=prodotto.divergence(flag="standard")
                prodotto = vsigmas['vsigma'][1] * grhoU
                v10=prodotto.divergence(flag="standard")
                prodotto = vsigmas['vsigma'][2] * grhoD
                v11=prodotto.divergence(flag="standard")
                OutFunctional.potential[0] -= 2 * v00+v01
                OutFunctional.potential[1] -= 2 * v11+v10
            else :
                prodotto = vsigmas['vsigma'] * grho
                vsigma_last = prodotto.divergence(flag="standard")
                OutFunctional.potential -= 2 * vsigma_last

        if hasattr(OutFunctional, 'v2rho2'):
            if density.rank > 1:
                prodotto = - vsigmas['v2rhosigma'][0] * grhoU
                v2rhosigma00 = prodotto.divergence(flag="standard")
                prodotto = - vsigmas['v2rhosigma'][1] * grhoD * 0.5
                v2rhosigma01 = prodotto.divergence(flag="standard")
                prodotto = - vsigmas['v2rhosigma'][1] * grhoU * 0.5
                v2rhosigma11 = prodotto.divergence(flag="standard")
                prodotto = - vsigmas['v2rhosigma'][2] * grhoD
                v2rhosigma12 = prodotto.divergence(flag="standard")
                prodotto = - vsigmas['v2rhosigma'][3] * grhoU
                v2rhosigma13 = prodotto.divergence(flag="standard")
                prodotto = - vsigmas['v2rhosigma'][4] * grhoD * 0.5
                v2rhosigma14 = prodotto.divergence(flag="standard")
                prodotto = - vsigmas['v2rhosigma'][4] * grhoU * 0.5
                v2rhosigma24 = prodotto.divergence(flag="standard")
                prodotto = - vsigmas['v2rhosigma'][5] * grhoD
                v2rhosigma25 = prodotto.divergence(flag="standard")
                prolapto = vsigmas['v2sigma2'][0] * grhoU.dot(grhoU) * 2.0
                v2sigma200 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][1] * grhoU.dot(grhoD)
                v2sigma201 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][1] * grhoU.dot(grhoU)
                v2sigma211 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][2] * grhoU.dot(grhoD) * 2.0
                v2sigma212 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][3] * grhoD.dot(grhoD) * 0.5
                v2sigma203 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][3] * grhoU.dot(grhoD)
                v2sigma213 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][3] * grhoU.dot(grhoU) * 0.5
                v2sigma223 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][4] * grhoD.dot(grhoD)
                v2sigma214 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][4] * grhoU.dot(grhoD)
                v2sigma224 = prolapto.laplacian(force_real = True)
                prolapto = vsigmas['v2sigma2'][5] * grhoD.dot(grhoD) * 2.0
                v2sigma225 = prolapto.laplacian(force_real = True)
                OutFunctional.v2rho2[0] = OutFunctional.v2rho2[0] + v2rhosigma00 + v2rhosigma01 + v2sigma200 + v2sigma201 + v2sigma203
                OutFunctional.v2rho2[1] = OutFunctional.v2rho2[1] + v2rhosigma11 + v2rhosigma12 + v2rhosigma13 + v2rhosigma14 + v2sigma211 + v2sigma212 + v2sigma213 + v2sigma214
                OutFunctional.v2rho2[2] = OutFunctional.v2rho2[2] + v2rhosigma24 + v2rhosigma25 + v2sigma223 + v2sigma224 + v2sigma225
            else:
                prodotto = - vsigmas['v2rhosigma'] * grho
                v2rhosigma = prodotto.divergence(flag="standard")
                prolapto = vsigmas['v2sigma2'] * grho.dot(grho) * 2.0
                v2sigma2 = prolapto.laplacian(force_real = True)
                OutFunctional.v2rho2 = OutFunctional.v2rho2 + v2rhosigma + v2sigma2

        if hasattr(OutFunctional, 'v3rho3') or hasattr(OutFunctional, 'v4rho4'):
            raise Exception('3rd and higher order derivative for GGA functionals has not implemented yet.')

    if "zk" in out.keys():
        if density.rank > 1 :
            rho = np.sum(density, axis = 0)
        else :
            rho = density
        edens = rho * out["zk"].reshape(np.shape(rho))
        ene = edens.sum() * density.grid.dV
        OutFunctional.energy = ene
        OutFunctional.energydensity = edens

    return OutFunctional


def LibXC(density, k_str=None, x_str=None, c_str=None, calcType={"E","V"}, **kwargs):
    """
     Output:
        - out_functional: a functional evaluated with LibXC
     Input:
        - density: a DirectField (rank=1)
        - k_str, x_str,c_str: strings like "gga_k_lc94", "gga_x_pbe" and "gga_c_pbe"
    """
    TimeData.Begin("LibXC")
    if CheckLibXC():
        from pylibxc.functional import LibXCFunctional

    args = locals()
    do_sigma = False
    func_str = {}
    for key, value in args.items():
        if key in ["k_str", "x_str", "c_str"] and value is not None:
            if not isinstance(value, str):
                raise AttributeError(
                    "{} must be LibXC functionals. Check pylibxc.util.xc_available_functional_names()".format(key)
                )
            if value.startswith('hyb') or value.startswith('mgga'):
                raise AttributeError('Hybrid and Meta-GGA functionals have not been implemented yet')
            if value.startswith('gga'):
                do_sigma = True
            func_str[key] = value
    if not func_str:
        raise AttributeError("At least one of the k_str, x_str, c_str must not be None.")

    if not isinstance(density, (DirectField)):
        raise TypeError("density must be a rank-1 or -2 PBCpy DirectField")
    if density.rank == 1:
        polarization = "unpolarized"
    elif density.rank == 2:
        polarization = "polarized"
    else:
        raise AttributeError("density must be a rank-1 or -2 PBCpy DirectField")

    inp = Get_LibXC_Input(density, do_sigma=do_sigma)
    kargs = {'do_exc': False, 'do_vxc': False}
    if 'E' in calcType or 'D' in calcType:
        kargs.update({'do_exc': True})
    if 'V' in calcType:
        kargs.update({'do_vxc': True})
    if 'V2' in calcType:
        kargs.update({'do_fxc': True})
    if 'V3' in calcType:
        kargs.update({'do_kxc': True})
    if 'V4' in calcType:
        kargs.update({'do_lxc': True})

    for key, value in func_str.items():
        func = LibXCFunctional(value, polarization)
        TimeData.Begin("libxc_eval")
        out = func.compute(inp, **kargs)
        TimeData.End("libxc_eval")
        if 'out_functional' in locals():
            out_functional += Get_LibXC_Output(out, density)
            out_functional.name += "_" + value
        else:
            out_functional = Get_LibXC_Output(out, density)
            out_functional.name = value
    TimeData.End("LibXC")
    return out_functional


def PBE(density, calcType={"E","V"}):
    return LibXC(
        density=density,
        x_str="gga_x_pbe",
        c_str="gga_c_pbe",
        calcType=calcType
    )


def LDA_XC(density, calcType={"E","V"}):
    return LibXC(
        density=density, x_str="lda_x", c_str="lda_c_pz", calcType=calcType
    )


def LDA(rho, calcType={"E","V"}, **kwargs):
    if rho.rank > 1 :
        return LDA_XC(rho, calcType)
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
    rho_cbrt[rho_cbrt < 1E-30] = 1E-30 # for safe
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


def LDAStress(rho, energy=None, potential=None, **kwargs):
    TimeData.Begin("LDA_Stress")
    if energy is None:
        EnergyPotential = LDA(rho, calcType={"E","V"})
        potential = EnergyPotential.potential
        energy = EnergyPotential.energy
    elif potential is None :
        potential = LDA(rho, calcType={"V"}).potential
    stress = np.zeros((3, 3))
    try:
        Etmp = energy - np.einsum("..., ...-> ", potential, rho, optimize = 'optimal') * rho.grid.dV
    except Exception :
        Etmp = energy - np.asum(potential * rho) * rho.grid.dV
    for i in range(3):
        stress[i, i] = Etmp / rho.grid.volume
    TimeData.End("LDA_Stress")
    return stress


def _LDAStress(density, xc_str='lda_x', energy=None, flag='standard', **kwargs):
    if CheckLibXC():
        from pylibxc.functional import LibXCFunctional
    if density.rank > 1 :
        polarization = 'polarized'
    else:
        polarization = 'unpolarized'

    nspin = density.rank
    func_xc = LibXCFunctional(xc_str, polarization)
    inp = {}
    if nspin > 1 :
        rho = np.sum(density, axis = 0)
        rhoT = density.reshape((2, -1)).T
        inp["rho"] = rhoT.ravel()
    else :
        rho = density
        inp["rho"] = density.ravel()

    kargs = {'do_exc': True, 'do_vxc': True}
    if energy is not None :
        kargs['do_exc'] = False
        energy *= 0.5
    out= func_xc.compute(inp, **kargs)

    if "zk" in out.keys():
        edens = out["zk"].reshape(np.shape(rho))
        energy = np.einsum("ijk, ijk->", edens, rho) * density.grid.dV

    if nspin > 1 :
        v = out['vrho'].reshape((-1, 2)).T
        v = DirectField(density.grid, rank=density.rank, griddata_3d=v)
    else :
        v = DirectField(density.grid, rank=density.rank, griddata_3d=out['vrho'])
    stress = np.zeros((3, 3))
    try :
        P = energy - np.einsum("..., ...-> ", v, rho, optimize = 'optimal') * rho.grid.dV
    except Exception :
        P = energy - np.asum(v*rho) * rho.grid.dV
    stress = np.eye(3)*P
    return stress/ rho.grid.volume

def _GGAStress(density, xc_str='gga_x_pbe', energy=None, flag='standard', **kwargs):
    if CheckLibXC():
        from pylibxc.functional import LibXCFunctional

    if density.rank > 1 :
        polarization = 'polarized'
    else:
        polarization = 'unpolarized'

    nspin = density.rank
    func_xc = LibXCFunctional(xc_str, polarization)
    inp = {}
    if nspin > 1 :
        rho = np.sum(density, axis = 0)
        rhoT = density.reshape((2, -1)).T
        inp["rho"] = rhoT.ravel()
        gradDen = []
        for i in range(0, nspin):
            gradrho = density[i].gradient(flag=flag)
            gradDen.append(gradrho)
        sigma = []
        for i in range(0, nspin):
            for j in range(i, nspin):
                s = np.einsum("lijk,lijk->ijk", gradDen[i], gradDen[j])
                sigma.append(s)
        rank = (nspin * (nspin + 1))//2
        sigmaL = DirectField(grid=density.grid, rank=rank, griddata_3d=sigma)
        sigma = sigmaL.reshape((3, -1)).T
    else :
        rho = density
        inp["rho"] = density.ravel()
        gradDen = density.gradient(flag=flag)
        sigma = np.einsum("lijk,lijk->ijk", gradDen, gradDen)
        sigma = DirectField(grid=density.grid, rank=1, griddata_3d=sigma)
    inp["sigma"] = sigma.ravel()

    kargs = {'do_exc': True, 'do_vxc': True}
    if energy is not None :
        kargs['do_exc'] = False
        energy *= 0.5
    out= func_xc.compute(inp, **kargs)

    if "zk" in out.keys():
        edens = out["zk"].reshape(np.shape(rho))
        energy = np.einsum("ijk, ijk->", edens, rho) * density.grid.dV

    if nspin > 1 :
        v = out['vrho'].reshape((-1, 2)).T
        v = DirectField(density.grid, rank=density.rank, griddata_3d=v)
        vsigma = out["vsigma"].reshape((-1, 3)).T
        vsigma = DirectField(density.grid, rank=3, griddata_3d=vsigma)
        sigma = sigmaL
    else :
        v = DirectField(density.grid, rank=density.rank, griddata_3d=out['vrho'])
        vsigma = DirectField(density.grid, griddata_3d=out["vsigma"].reshape(np.shape(density)))

    P = energy
    try :
        P -= np.einsum("..., ...-> ", v, density, optimize = 'optimal') * rho.grid.dV
        P -= 2.0*np.einsum("..., ...-> ", sigma, vsigma, optimize = 'optimal') * rho.grid.dV
    except Exception :
        P -= np.sum(v*density) * rho.grid.dV
        P -= 2.0*np.sum(sigma*vsigma) * rho.grid.dV
    stress = np.eye(3)*P
    for i in range(3):
        for j in range(3):
            if nspin > 1 :
                stress[i, j] -= 2.0*np.einsum("ijk, ijk, ijk -> ", gradDen[0][i], gradDen[0][j], vsigma[0]) * rho.grid.dV
                stress[i, j] -= 2.0*np.einsum("ijk, ijk, ijk -> ", gradDen[0][i], gradDen[1][j], vsigma[1]) * rho.grid.dV
                stress[i, j] -= 2.0*np.einsum("ijk, ijk, ijk -> ", gradDen[1][i], gradDen[1][j], vsigma[2]) * rho.grid.dV
            else :
                stress[i, j] -= 2.0*np.einsum("ijk, ijk, ijk -> ", gradDen[i], gradDen[j], vsigma) * rho.grid.dV
    return stress/ rho.grid.volume


def GGAStress(density, x_str='gga_x_pbe', c_str='gga_c_pbe', energy=None, flag='standard', **kwargs):
    stress=_GGAStress(density, xc_str=x_str, energy=energy, flag=flag, **kwargs)
    stress+=_GGAStress(density, xc_str=c_str, energy=energy, flag=flag, **kwargs)
    return stress


def PBEStress(density, energy=None, flag='standard', **kwargs):
    stress=GGAStress(density, x_str='gga_x_pbe', c_str='gga_c_pbe', energy=energy, flag=flag, **kwargs)
    return stress


def XCStress(density, name=None, x_str='gga_x_pbe', c_str='gga_c_pbe', energy=None, flag='standard', **kwargs):
    TimeData.Begin("XCStress")
    if name == 'LDA' :
        x_str = 'lda_x'
        c_str = 'lda_c_pz'
        stress=_LDAStress(density, xc_str=x_str, energy=energy, flag=flag, **kwargs)
        stress+=_LDAStress(density, xc_str=c_str, energy=energy, flag=flag, **kwargs)
    elif name == 'PBE' :
        x_str = 'gga_x_pbe'
        c_str = 'gga_c_pbe'
        stress=_GGAStress(density, xc_str=x_str, energy=energy, flag=flag, **kwargs)
        stress+=_GGAStress(density, xc_str=c_str, energy=energy, flag=flag, **kwargs)
    elif x_str[:3] == c_str[:3] == 'lda' :
        stress=_LDAStress(density, xc_str=x_str, energy=energy, flag=flag, **kwargs)
        stress+=_LDAStress(density, xc_str=c_str, energy=energy, flag=flag, **kwargs)
    elif x_str[:3] == c_str[:3] == 'gga' :
        stress=_LDAStress(density, xc_str=x_str, energy=energy, flag=flag, **kwargs)
        stress+=_LDAStress(density, xc_str=c_str, energy=energy, flag=flag, **kwargs)
    else :
        raise AttributeError("'x_str' %s and 'c_str' %s must be same type" %(x_str, c_str))

    TimeData.End("XCStress")
    return stress
