# Collection of Kinetic Energy Density Functionals
import numpy as np
import copy
from dftpy.mpi import sprint
from dftpy.field import DirectField
from dftpy.kedf.tf import *
from dftpy.kedf.vw import *
from dftpy.kedf.wt import *
from dftpy.kedf.lwt import *
from dftpy.kedf.fp import *
from dftpy.kedf.sm import *
from dftpy.kedf.mgp import *
from dftpy.kedf.gga import *
from dftpy.kedf.hc import *
from dftpy.functional_output import Functional
from dftpy.semilocal_xc import LibXC

__all__ = ["KEDF", "KEDFunctional", "KEDFStress"]

KEDF_Dict = {
    "TF": TF,
    "vW": vW,
    "GGA": GGA,
}

NLKEDF_Dict = {
    "WT": WT,
    "SM": SM,
    "FP": FP,
    "MGP": MGP,
    "MGPA": MGPA,
    "MGPG": MGPG,
    "LWT": LWT,
    "LMGP": LMGP,
    "LMGPA": LMGPA,
    "LMGPG": LMGPG,
    "HC" : HC,
}

KEDF_Stress_Dict = {
    "TF": ThomasFermiStress,
    "vW": vonWeizsackerStress,
    "WT": WTStress,
}

class KEDF:
    def __init__(self, name = "WT", **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.ke_kernel_saved = {
                "Kernel": None,
                "rho0": 0.0,
                "shape": None,
                "KernelTable": None,
                "etamax": None,
                "KernelDeriv": None,
                "MGPKernelE": None,
                "kfmin" :None,
                "kfmax" :None,
                }

    def __new__(cls, name = "WT", **kwargs):
        if name.startswith('STV+GGA+') :
            return kedf2nlgga(name, **kwargs)
        elif name.startswith('MIX_') :
            return kedf2mixkedf(name, **kwargs)
        else :
            return super(KEDF, cls).__new__(cls)

    def __call__(self, density, calcType=["E","V"], name=None, **kwargs):
        if name is None :
            name = self.name
        return self.compute(density, name=name, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], name=None, split = False, **kwargs):
        if name is None :
            name = self.name
        ke_kwargs = copy.deepcopy(self.kwargs)
        ke_kwargs.update(kwargs)
        functional = None
        for item in name.split('+'):
            out = KEDFunctional(density, name = item, calcType = calcType, split = split, ke_kernel_saved = self.ke_kernel_saved, **ke_kwargs)
            if functional is None :
                functional = out
            else :
                if split :
                    functional.update(out)
                else :
                    functional += out
        return functional

def KEDFunctional(rho, name="WT", calcType=["E","V"], split=False, nspin = 1, **kwargs):
    """
    KEDF interface
    """
    if rho.ndim > 3 :
        nspin = rho.rank
        ke = KEDFunctional(rho[0] * nspin, name, calcType, split, nspin = nspin, **kwargs)
        if split :
            for i in range(1, nspin):
                ke1 = KEDFunctional(rho[i] * nspin, name, calcType, split, nspin = nspin, **kwargs)
                for k1 in ke :
                    if 'E' in calcType :
                        ke[k1].energy = ke[k1].energy + ke1[k1].energy
                    if 'V' in calcType :
                        ke[k1].potential = np.vstack((ke[k1].potential, ke1[k1].potential))
                    if 'D' in calcType :
                        ke[k1].energydensity = np.vstack((ke[k1].energydensity, ke1[k1].energydensity))
            for k1 in ke :
                if 'V' in calcType :
                    ke[k1].potential = ke[k1].potential.reshape(rho.shape)
                if 'D' in calcType :
                    ke[k1].energydensity = ke[k1].energydensity.reshape(rho.shape)
        else :
            for i in range(1, nspin):
                ke1 = KEDFunctional(rho[i] * nspin, name, calcType, split, nspin = nspin, **kwargs)
                if 'E' in calcType :
                    ke.energy = ke.energy + ke1.energy
                if 'V' in calcType :
                    ke.potential = np.vstack((ke.potential, ke1.potential))
                if 'D' in calcType :
                    ke.energydensity = np.vstack((ke.energydensity, ke1.energydensity))
            if 'V' in calcType :
                ke.potential = DirectField(grid=rho.grid, griddata_3d=ke.potential, rank=nspin)
            if 'D' in calcType :
                ke.energydensity = DirectField(grid=rho.grid, griddata_3d=ke.energydensity, rank=nspin)
        return ke
    #-----------------------------------------------------------------------
    if name.upper() == "NONE" :
        OutFunctional = Functional(name="NONE", energy = 0.0)
        if 'D' in calcType:
            OutFunctional.energydensity = DirectField(grid=rho.grid, rank=nspin)
        if 'V' in calcType:
            OutFunctional.potential = DirectField(grid=rho.grid, rank=nspin)
        OutFunctionalDict = {"NONE": OutFunctional}
    elif name[:3] == "GGA":
        func = GGA
        k_str = kwargs["k_str"].upper()
        if k_str not in GGA_KEDF_list:
            raise AttributeError("%s GGA KEDF to be implemented" % k_str)
        else:
            OutFunctional = func(rho, functional=k_str, calcType=calcType, **kwargs)
            OutFunctionalDict = {"GGA": OutFunctional}
    elif name == "LIBXC_KEDF":
        k_str = kwargs.get("k_str", "gga_k_lc94").lower()
        OutFunctional = LibXC(rho, k_str=k_str, calcType=calcType)
        OutFunctionalDict = {"XC": OutFunctional}
    elif name in KEDF_Dict:
        func = KEDF_Dict[name]
        OutFunctional = func(rho, calcType=calcType, **kwargs)
        if nspin > 1 and 'E' in calcType :
            OutFunctional.energy /= nspin
        OutFunctionalDict = {name: OutFunctional}
    elif name == "x_TF_y_vW" or name == "TFvW" or name == "xTFyvW" :
        xTF = TF(rho, calcType=calcType, **kwargs)
        yvW = vW(rho, calcType=calcType, **kwargs)
        if nspin > 1 and 'E' in calcType :
            xTF.energy /= nspin
            yvW.energy /= nspin
        OutFunctional = xTF + yvW
        OutFunctional.name = name
        OutFunctionalDict = {"TF": xTF, "vW": yvW}
    elif name in NLKEDF_Dict:
        func = NLKEDF_Dict[name]
        NL = func(rho, calcType=calcType, **kwargs)
        x = kwargs.get("x", 1.0)
        y = kwargs.get("y", 1.0)
        alpha = kwargs.get("alpha", 5.0 / 6.0)
        beta = kwargs.get("beta", 5.0 / 6.0)
        if abs(x) > 1e-8 :
            if abs(alpha + beta - 5.0 / 3.0) < 1e-8:
                s = 1.0 + (x - 1) * 20 / (25 - (abs(beta - alpha) * 3) ** 2)  # only works when \alpha+\beta=5/3
            else:
                s = x
            xTF = TF(rho, x=s, calcType=calcType)
        else :
            xTF = Functional(name="ZERO", energy = 0.0)

        if abs(y) > 1e-8 :
            yvW = vW(rho, calcType=calcType, **kwargs)
        else :
            yvW = Functional(name="ZERO", energy = 0.0)
        if nspin > 1 and 'E' in calcType :
            xTF.energy /= nspin
            yvW.energy /= nspin
            NL.energy /= nspin
        if hasattr(NL, 'potential'):
            mask = (NL.potential > 0.0) & (rho < 1E-3)
            NL.potential[mask] = -1.0*xTF.potential[mask]
        OutFunctional = NL + xTF + yvW
        OutFunctional.name = name
        OutFunctionalDict = {"TF": xTF, "vW": yvW, "NL": NL}
    elif name[:5] == 'NLGGA' and name[6:] in NLKEDF_Dict:
        func = NLKEDF_Dict[name[6:]]
        NL = func(rho, calcType=calcType, **kwargs)
        k_str = kwargs["k_str"].upper()
        if k_str in GGA_KEDF_list:
            gga = GGA(rho, functional=k_str, calcType=calcType, **kwargs)
        else :
            raise AttributeError("%s GGA KEDF to be implemented" % k_str)
        if nspin > 1 and 'E' in calcType :
            gga.energy /= nspin
            NL.energy /= nspin
        OutFunctional = NL + gga
        OutFunctional.name = name
        OutFunctionalDict = {"GGA": gga, "NL": NL}
    else:
        raise AttributeError("%s KEDF to be implemented" % name)
    #-----------------------------------------------------------------------
    # rho_min = kwargs.get("rho_min", None)
    # if rho_min is not None and 'V' in calcType :
        # OutFunctional.potential[rho < rho_min] = 0.0
    #-----------------------------------------------------------------------

    if split:
        return OutFunctionalDict
    else:
        return OutFunctional

def KEDFStress(rho, name="WT", energy=None, **kwargs):
    """
    KEDF stress interface
    """
    if name in KEDF_Stress_Dict :
        func = KEDF_Stress_Dict[name]

    if rho.ndim > 3 :
        nspin = rho.rank
        stress = func(rho[0] * nspin, energy=energy, **kwargs)
        for i in range(1, nspin):
            stress += func(rho[i] * nspin, energy=energy, **kwargs)
        stress /= nspin
    else :
        stress = func(rho, energy=energy, **kwargs)

    return stress


class NLGGA:
    def __init__(self, stv = None, gga = None, nl = None, rhomax = None, name = 'STV+GGA+LMGPA') :
        self.stv = stv
        self.gga = gga
        self.nl = nl
        self.rhomax = rhomax
        self.level = 3 # if smaller than 3 only use gga to guess the rhomax
        self.name = name

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], **kwargs):
        calcType = ["E","V"]
        # T_{s}[n]=\int \epsilon[n](\br) d\br=\int W[n](\br)\left[\epsilon_{NL} [n] + \epsilon_{STV}(n)\right] + \bigg(1-W[n](\br)\bigg)\epsilon_{GGA} d\br
        if 'V' in calcType :
            calc = ['D', 'V']
        elif 'E' in calcType :
            calc = ['D']

        rhomax_w = density.amax()

        nmax = kwargs.get('rhomax', None) or self.rhomax
        if not nmax : nmax = rhomax_w
        kfmax = 2.0 * np.cbrt(3.0 * np.pi ** 2 * nmax)
        kwargs['kfmax'] = kfmax

        func_gga = self.gga(density, calcType = calc, **kwargs)

        # truncate the density higher than rhomax for NL
        if rhomax_w > nmax :
            sprint('!WARN : some density large than rhomax', rhomax_w, nmax, comm = density.mp.comm, level=1)

        mask = density > nmax
        if np.count_nonzero(mask) > 0 :
            density_trunc = density.copy()
            density_trunc[mask] = nmax
        else :
            density_trunc = density

        if self.level > 2 :
            func_stv = self.stv(density, calcType = calc, **kwargs)
            func_nl = self.nl(density_trunc, calcType = calc, **kwargs)
            kdd = kwargs.get('kdd', self.nl.kwargs.get('kdd', 1))
            if kdd < 3 :
                func_nl.energydensity = 3.0/8.0 * density_trunc * func_nl.potential

        wn = density_trunc / nmax

        obj = Functional(name = 'NLGGA')

        if 'V' in calcType :
            # V_{STV} = W[n]v_{STV} + \frac{\epsilon_{STV}}{n_{max}}
            # V_{GGA} = \left(1-W[n]\right)v_{GGA} -\frac{\epsilon_{GGA}}{n_{max}}
            # V_{NL} =W[n]v_{NL} +\frac{\epsilon_{NL}}{n_{max}}
            if self.level > 2 :
                pot = wn * func_stv.potential + func_stv.energydensity/nmax
                pot += (1 - wn)* func_gga.potential - func_gga.energydensity/nmax
                pot += wn * func_nl.potential + func_nl.energydensity/nmax
            else :
                pot = func_gga.potential
            obj.potential = pot

        if 'E' in calcType :
            # E_{STV} =\int  W[n(\br)] \epsilon_{STV}(\br) d\br
            # E_{GGA} =\int (1-W[n](\br))\epsilon_{GGA}(\br)
            # E_{NL} =\int \epsilon_{NL} W[n](\br) d\br .
            if self.level > 2 :
                energydensity = wn * func_stv.energydensity
                energydensity += (1 - wn)* func_gga.energydensity
                energydensity += wn * func_nl.energydensity
            else :
                energydensity = func_gga.energydensity
            energy = energydensity.sum() * density.grid.dV
            obj.energy = energy

        return obj

def kedf2nlgga(name = 'STV+GGA+LMGPA', **kwargs) :
    """
    Base on the input generate NLKEDF

    Args:
        name: the name of NLGGA

    Raises:
        AttributeError: Must contain three parts in the name

    """

    if not name.startswith('STV+GGA+') :
        raise AttributeError("The name of NLGGA is not correct : {}".format(name))
    names = name.split('+')

    # Only for STV
    params = kwargs.get("params", None)
    if params is not None : del kwargs["params"]

    # Only for GGA
    if "k_str" in kwargs :
        k_str=kwargs.pop("k_str")
    else :
        k_str="REVAPBEK"

    # Remove TF and vW from NL
    for key in ['x', 'y'] :
        kwargs[key] = 0.0

    sigma = kwargs.get("sigma", None)
    rhomax = kwargs.get("rhomax", None)

    if k_str.upper() in GGA_KEDF_list :
        names[1] = 'GGA'
        k_str = k_str.upper()
    else :
        names[1] = 'LIBXC_KEDF'
        k_str = k_str.lower()

    stv = KEDF(name = 'GGA', k_str = 'STV', params = params, sigma = sigma)
    gga = KEDF(name = names[1], k_str = k_str, sigma = sigma)
    #-----------------------------------------------------------------------
    if names[2] == 'HC' :
        kwargs['k_str'] = 'PBE2'
        kwargs['params'] = [0.1, 0.45]
        kwargs['delta'] = 0.3
    #-----------------------------------------------------------------------
    nl = KEDF(name = names[2], **kwargs)
    obj = NLGGA(stv, gga, nl, rhomax = rhomax, name = name)

    return obj

def kedf2mixkedf(name = 'MIX_TF+GGA', first_high = True, **kwargs) :
    """
    Base on the input generate MIXKEDF

    Args:
        name: the name of MIXKEDF

    Raises:
        AttributeError: The name must startswith 'MIX_'

    """

    if not name.startswith('MIX_') :
        raise AttributeError("The name of MIXKEDF is not correct : {}".format(name))
    names = name[4:].split('+')

    # Only for second(GGA)
    if "k_str" in kwargs :
        k_str=kwargs.pop("k_str")
    else :
        k_str="REVAPBEK"

    sigma = kwargs.get("sigma", None)
    rhomax = kwargs.get("rhomax", None)

    if k_str.upper() in GGA_KEDF_list :
        names[1] = 'GGA'
        k_str = k_str.upper()
    else :
        names[1] = 'LIBXC_KEDF'
        k_str = k_str.lower()

    stv = KEDF(name = names[0], **kwargs)
    gga = KEDF(name = names[1], k_str = k_str, sigma = sigma)
    obj = MIXGGAS(stv, gga, rhomax = rhomax, name = name)

    return obj

class MIXGGAS:
    def __init__(self, stv = None, gga = None, rhomax = None, name = 'MIX_TF+GGA') :
        self.stv = stv
        self.gga = gga
        self.rhomax = rhomax
        self.name = name

    def __call__(self, density, calcType=["E","V"], **kwargs):
        return self.compute(density, calcType=calcType, **kwargs)

    def compute(self, density, calcType=["E","V"], **kwargs):
        calcType = ["E","V"]
        if 'V' in calcType :
            calc = ['D', 'V']
        elif 'E' in calcType :
            calc = ['D']

        func_stv = self.stv(density, calcType = calc, **kwargs)

        func_gga = self.gga(density, calcType = calc, **kwargs)

        obj = Functional(name = 'MIXGGAS')

        self.interpfunc(density, calcType = calcType, **kwargs)
        interpolate_f = self.interpolate_f
        interpolate_df = self.interpolate_df

        if 'V' in calcType :
            pot = interpolate_f * func_stv.potential + interpolate_df * func_stv.energydensity
            pot += (1 - interpolate_f)* func_gga.potential - interpolate_df * func_gga.energydensity
            obj.potential = pot

        if 'E' in calcType :
            energydensity = interpolate_f * func_stv.energydensity
            energydensity += (1 - interpolate_f)* func_gga.energydensity
            energy = energydensity.sum() * density.grid.dV
            obj.energy = energy

        return obj

    def interpfunc(self, rho, calcType=["E","V"], func = 'tanh', **kwargs):
        if self.rhomax is None or self.rhomax < 1E-30 :
            self.interpolate_f = 1.0
            self.interpolate_df = 0.0
        elif self.rhomax > 100 :
            self.interpolate_f = 0.0
            self.interpolate_df = 0.0
        else :
            if func == 'tanh' :
                self.interp_tanh(rho, calcType=calcType, **kwargs)
            else :
                self.interp_linear(rho, calcType=calcType, **kwargs)
        return

    def interp_tanh(self, rho, calcType=["E","V"], **kwargs):
        # x = -2 * np.abs(rho/self.rhomax)
        # ex = np.exp(x)
        # self.interpolate_f = (1 - ex)/(1 + ex)
        x = np.abs(rho/self.rhomax)
        self.interpolate_f = np.tanh(x)
        self.interpolate_df = 1.0/(np.cosh(x)**2*self.rhomax)
        return

    def interp_linear(self, rho, calcType=["E","V"], **kwargs):
        self.interpolate_f = np.abs(rho/self.rhomax)
        mask = self.interpolate_f > 1.0
        self.interpolate_f[mask] = 1.0
        self.interpolate_df = np.ones_like(self.interpolate_f)/self.rhomax
        self.interpolate_df[mask] = 0.0
        return

    def _interp_lkt(self, rho, calcType=["E","V"], **kwargs):
        sigma = kwargs.get('sigma', None)
        rhom = rho.copy()
        tol = 1e-16
        rhom[rhom < tol] = tol

        rho23 = rhom ** (2.0 / 3.0)
        rho43 = rho23 * rho23
        rho83 = rho43 * rho43
        q = rho.grid.get_reciprocal().q
        g = rho.grid.get_reciprocal().g

        rhoG = rho.fft()
        rhoGrad = []
        for i in range(3):
            if sigma is None :
                grhoG = g[i] * rhoG * 1j
            else :
                grhoG = g[i] * rhoG * np.exp(-q*(sigma)**2/4.0) * 1j
            item = (grhoG).ifft(force_real=True)
            rhoGrad.append(item)
        s = np.sqrt(rhoGrad[0] ** 2 + rhoGrad[1] ** 2 + rhoGrad[2] ** 2) / rho43
        F, dFds2 = GGAFs(s, functional='LKT', calcType=calcType, gga_remove_vw = True, **kwargs)

        if 'V' in calcType:
            dFdn = -4.0 / 3.0 * dFds2 * s * s / rhom
            p3 = []
            for i in range(3):
                item = dFds2 * rhoGrad[i] / rho83
                p3.append(item.fft())
            pot3G = g[0] * p3[0] + g[1] * p3[1] + g[2] * p3[2]
            dFdn -= (1j * pot3G).ifft(force_real=True)

        self.interpolate_s = s
        self.interpolate_f = F
        self.interpolate_df = dFdn
