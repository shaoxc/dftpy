# Drivers for LibXC

import numpy as np
from dftpy.field import DirectField
from dftpy.functional_output import Functional
from dftpy.constants import MATHLIB
from dftpy.math_utils import TimeData

def CheckLibXC()
    import importlib
    islibxc = importlib.util.find_spec("pylibxc")
    found = islibxc is not None
    if found:
        from pylibxc.functional import LibXCFunctional
    else: 
        raise ModuleNotFoundError("Must install LibXC and pylibxc") 
    return found


def Get_LibXC_Input(density,do_sigma=True):
    if not isinstance(density,(DirectField)):
        raise TypeError("density must be a PBCpy DirectField")
    if density.rank != 1:
        raise AttributeError("Wrong rank")
    dim=np.shape(np.shape(density))[0]
    if dim > 4 or dim < 3:
        raise AttributeError("Wrong dimension of density input")
    rho=density.ravel()
    inp = {}
    inp["rho"]=rho
    if do_sigma:
        sigma = density.sigma().ravel()
        # sigma = density.sigma('standard').ravel()
        inp["sigma"]=sigma
    return inp

def Compute_LibXC(inp,func,spin):
    if not isinstance(inp,(dict)):
        raise AttributeError("LibXC Input must be a dictionary")
    if spin == 1:
        sspin="unpolarized"
    else:
        sspin="polarized"
    print("Computing "+func+" by LibXC")
    func=LibXCFunctional(func, sspin)
    return func.compute(inp)

def Get_LibXC_Output(out,density, calcType = 'Both'):
    if not isinstance(out,(dict)):
        raise TypeError("LibXC output must be a dictionary")
    if density.rank != 1:
        raise AttributeError("Wrong rank")
    dim=np.shape(np.shape(density))[0]
    if dim > 4 or dim < 3:
        raise AttributeError("Wrong dimension of density input")

    OutFunctional = Functional(name='LibXC')

    do_sigma = False
    if "vsigma" in out.keys():
        do_sigma = True

    if do_sigma:
        sigma = density.sigma().reshape(np.shape(density)[0]*np.shape(density)[1]*np.shape(density)[2])
    if "zk" in out.keys():
        edens = out["zk"].reshape(np.shape(density))

    if "vrho" in out.keys():
        vrho = DirectField(density.grid,rank=1,griddata_3d=out["vrho"].reshape(np.shape(density)))

    if "vsigma" in out.keys():
        vsigma = DirectField(density.grid,griddata_3d=out["vsigma"].reshape(np.shape(density)))

    if not do_sigma:
        ene = np.einsum('ijkl, ijkl->',edens,density) * density.grid.dV
        pot = DirectField(density.grid,rank=1,griddata_3d=vrho)
    else:
        # print('rhoLIB1', np.max(density), np.min(density))
        grho = density.gradient(flag='supersmooth')
        # print('rhoLIB2', np.max(density), np.min(density))
        prodotto=vsigma*grho 
        vsigma_last = prodotto.divergence()
        # print('rhoLIB3', np.max(density), np.min(density))
        v=vrho-2*vsigma_last

        ene = np.real(np.einsum('ijkl->',edens*density)) * density.grid.dV
        pot = DirectField(density.grid,rank=1,griddata_3d=np.real(v))

    OutFunctional.energy = ene
    OutFunctional.potential = pot

    return OutFunctional 

def XC(density,x_str,c_str,polarization, do_sigma = True, calcType = 'Both'):
    if CheckLibXC ():
        from pylibxc.functional import LibXCFunctional
    '''
     Output: 
        - Functional_XC: a XC functional evaluated with LibXC
     Input:
        - density: a DirectField (rank=1)
        - x_str,c_str: strings like "gga_x_pbe" and "gga_c_pbe"
        - polarization: string like "polarized" or "unpolarized"
    '''
    if not isinstance(x_str, str):
        raise AttributeError("x_str and c_str must be LibXC functionals. Check pylibxc.util.xc_available_functional_names()")
    if not isinstance(c_str, str):
        raise AttributeError("x_str and c_str must be LibXC functionals. Check pylibxc.util.xc_available_functional_names()")
    if not isinstance(polarization, str):
        raise AttributeError("polarization must be a ``polarized`` or ``unpolarized``")
    if not isinstance(density,(DirectField)):
        raise AttributeError("density must be a rank-1 PBCpy DirectField")
    func_x = LibXCFunctional(x_str, polarization)
    func_c = LibXCFunctional(c_str, polarization)
    # inp=Get_LibXC_Input(density, do_sigma = False)
    inp=Get_LibXC_Input(density, do_sigma = do_sigma)
    kargs = {'do_exc' :True, 'do_vxc' :True}
    if calcType == 'Energy' :
        kargs['do_vxc'] = False
    elif calcType == 'Potential' :
        kargs['do_exc'] = False
    out_x = func_x.compute(inp, **kargs)
    out_c = func_c.compute(inp, **kargs)
    Functional_X = Get_LibXC_Output(out_x,density, calcType = calcType)
    Functional_C = Get_LibXC_Output(out_c,density, calcType = calcType)
    Functional_XC = Functional_X.sum(Functional_C)
    name = x_str[6:]+"_"+c_str[6:]
    Functional_XC.name = name.upper()
    return Functional_XC

# def PBE_XC(density,polarization, calcType = 'Both'):
def PBE(density,polarization = 'unpolarized', calcType = 'Both', **kwargs):
    return XC(density=density,x_str='gga_x_pbe',c_str='gga_c_pbe',polarization=polarization, do_sigma=True, calcType=calcType)

def LDA_XC(density,polarization = 'unpolarized', calcType = 'Both',  **kwargs):
    return XC(density=density,x_str='lda_x',c_str='lda_c_pz',polarization=polarization, do_sigma=False, calcType=calcType)

def LDA(rho, polarization = 'unpolarized', calcType = 'Both',  **kwargs):
    TimeData.Begin('LDA')
    # return LDA_XC(rho,polarization, calcType)
    a=( 0.0311,  0.01555)
    b=(-0.048,  -0.0269)
    c=( 0.0020,  0.0007)
    d=(-0.0116, -0.0048)
    gamma=(-0.1423, -0.0843)
    beta1=( 1.0529,  1.3981)
    beta2=( 0.3334,  0.2611)

    rho_cbrt = np.cbrt(rho)
    Rs = np.cbrt(3.0/(4.0 *np.pi)) / rho_cbrt
    rs1 = Rs < 1
    rs2 = Rs >= 1
    Rs2sqrt = np.sqrt(Rs[rs2])
    ene = 0
    if calcType == 'Energy' or calcType == 'Both' :
        ExRho = -3.0/4.0 * np.cbrt(3.0/np.pi) * rho_cbrt
        ExRho[rs1] += a[0] * np.log(Rs[rs1]) + b[0] + c[0] * Rs[rs1] * np.log(Rs[rs1]) + d[0] * Rs[rs1]
        ExRho[rs2] += gamma[0] / (1.0+beta1[0] * Rs2sqrt + beta2[0] * Rs[rs2])
        ene = np.einsum('ijkl, ijkl->',ExRho, rho) * rho.grid.dV
        if calcType == 'Energy' :
            pot = np.empty_like(rho)
    if calcType == 'Potential' or calcType == 'Both' :
        pot = np.cbrt(-3.0/np.pi) * rho_cbrt
        pot[rs1] += np.log(Rs[rs1]) * (a[0]+2.0/3 * c[0] * Rs[rs1]) + b[0]-1.0/3 * a[0]+1.0/3 * (2 * d[0]-c[0]) * Rs[rs1]
        pot[rs2] += ( gamma[0]+(7.0/6.0 * gamma[0] * beta1[0]) * Rs2sqrt + (4.0/3.0 * gamma[0] * beta2[0] * Rs[rs2]))\
                /( 1.0+beta1[0] * Rs2sqrt + beta2[0] * Rs[rs2]) ** 2
        ##rs1
        # part1 = (a[0]+2.0/3 * c[0] * Rs[rs1]) * np.log(Rs[rs1]) 
        # part2 =  b[0]-1.0/3 * a[0]
        # part3 = 1.0/3 * (2 * d[0]-c[0]) * Rs[rs1]
        # pot[rs1] += part1 + part2 + part3
        # #rs2
        # part1 = (7.0/6.0 * gamma[0] * beta1[0]) * Rs2sqrt 
        # part2 = (4.0/3.0 * gamma[0] * beta2[0] * Rs[rs2])
        # part3 = 1.0+beta1[0] * Rs2sqrt + beta2[0] * Rs[rs2]
        # pot[rs2] += ( gamma[0]+ part1 + part2 ) / (part3 * part3)

        ##rs1
        # part1 = 2.0/3 * c[0] * Rs[rs1]
        # np.add(a[0], part1, out = part1)
        # np.multiply(part1, np.log(Rs[rs1]),out = part1)
        # np.add(part1, b[0]-1.0/3 * a[0], out = part1)
        # np.add(part1,1.0/3 * (2 * d[0]-c[0]) * Rs[rs1], out = part1)
        # pot[rs1] += part1
        # #rs2
        # part1 = (7.0/6.0 * gamma[0] * beta1[0]) * Rs2sqrt 
        # part2 = (4.0/3.0 * gamma[0] * beta2[0] * Rs[rs2])
        # np.add(part1, part2, out = part1)
        # np.add(part1, gamma[0], out = part1)
        # np.multiply(beta1[0], Rs2sqrt, out = part2) 
        # np.add(part2, beta2[0] * Rs[rs2], out = part2)
        # np.add(part2, 1.0, out = part2)
        # np.square(part2, out = part2)
        # np.divide(part1, part2, out = part1)
        # pot[rs2] += part1

        # Rs2sqrt = np.sqrt(Rs)
        # pot = np.cbrt(-3.0/np.pi) * rho_cbrt
        # pot1 = np.log(Rs) * (a[0]+2.0/3 * c[0] * Rs) + b[0]-1.0/3 * a[0]+1.0/3 * (2 * d[0]-c[0]) * Rs
        # pot2 = ( gamma[0]+(7.0/6.0 * gamma[0] * beta1[0]) * Rs2sqrt + (4.0/3.0 * gamma[0] * beta2[0] * Rs))\
                # /( 1.0+beta1[0] * Rs2sqrt + beta2[0] * Rs) ** 2
        # pot[rs1] += pot1[rs1]
        # pot[rs2] += pot2[rs2]


    OutFunctional = Functional(name='XC')
    OutFunctional.energy = ene
    OutFunctional.potential = pot
    TimeData.End('LDA')
    return OutFunctional

def LDAStress(rho, polarization='unpolarized', energy=None):
    TimeData.Begin('LDA_Stress')
    if energy is None :
        EnergyPotential = LDA(rho, polarization, calcType = 'Both')
        energy = EnergyPotential.energy
        potential = EnergyPotential.potential
    else :
        potential = LDA(rho, polarization, calcType = 'Potential').potential
    stress = np.zeros((3, 3))
    Etmp = energy - np.einsum('ijkl, ijkl -> ', potential, rho) * rho.grid.dV
    for i in range(3):
        stress[i, i]= Etmp / rho.grid.volume
    TimeData.End('LDA_Stress')
    return stress

def LIBXC_KEDF(density,polarization = 'unpolarized', kstr='gga_k_lc94', calcType = 'Both', split = False, **kwargs):
    if CheckLibXC ():
        from pylibxc.functional import LibXCFunctional
    '''
     Output: 
        - Functional_KEDF: a KEDF functional evaluated with LibXC
     Input:
        - density: a DirectField (rank=1)
        - kstr: strings like "gga_k_lc94"
        - polarization: string like "polarized" or "unpolarized"
    '''
    if not isinstance(kstr, str):
        raise AttributeError("kstr must be a LibXC functional. Check pylibxc.util.xc_available_functional_names()")
    if not isinstance(polarization, str):
        raise AttributeError("polarization must be a ``polarized`` or ``unpolarized``")
    if not isinstance(density,(DirectField)):
        raise AttributeError("density must be a rank-1 PBCpy DirectField")
    func_k = LibXCFunctional(kstr, polarization)
    inp=Get_LibXC_Input(density)
    out_k = func_k.compute(inp)
    Functional_KEDF = Get_LibXC_Output(out_k,density)
    name = kstr[6:]
    Functional_KEDF.name = name.upper()
    return Functional_KEDF
