#!/usr/bin/env python3

import time

import numpy as np
from scipy.optimize import minimize
from typing import Dict

from dftpy import __version__
from dftpy.ions import Ions
from dftpy.cui.main import GetConf
from dftpy.field import DirectField
from dftpy.formats.io import write
from dftpy.formats.qepp import PP
from dftpy.functional import Functional, ExternalPotential
from dftpy.functional.pseudo.layer_pseudo import LayerPseudo
from dftpy.functional.total_functional import TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.interface import ConfigParser
from dftpy.inverter import Inverter
from dftpy.math_utils import interpolation_3d
from dftpy.mpi import mp, sprint
from dftpy.optimization import Optimization
from dftpy.time_data import TimeData
from dftpy.mpi import sprint
from scipy.signal import quadratic
from typing import Optional, Tuple
from dftpy.td.real_time_runner import RealTimeRunner
from dftpy.formats.xsf import XSF


def bisec(func: callable, xstart: float, xend: float, tol: float = 1.0e-4,
          ystart: Optional[float] = None, yend: Optional[float] = None) -> Tuple[float, float]:

    if ystart is None:
        ystart = func(xstart)
    if yend is None:
        yend = func(xend)

    xmid = (xstart + xend) / 2.0
    ymid = func(xmid)

    if ystart < yend:
        xend, yend = xmid, ymid
    else:
        xstart, ystart = xmid, ymid

    if np.abs(ystart - yend) <= tol:
        return (xstart, ystart) if ystart <= yend else (xend, yend)

    sprint(xstart, xend, ystart, yend)
    return bisec(func, xstart, xend, tol, ystart, yend)


def set_ls(ls, a, r_cut):
    ls.vr = a[0] * quadratic(ls.r / r_cut * 3 - 1.5)
    ls.vr += a[1] * quadratic(ls.r / r_cut * 3 / 2 - 1.5 / 2)
    ls.vr += a[2] * quadratic(ls.r / r_cut * 3 / 2 + 1.5 / 2)
    # ls.vr = a[0] * quadratic(ls.r / r_cut * 3 - 1.5)
    # ls.vr += a[1] * quadratic(ls.r / r_cut * a[3])
    # #ls.vr += a[1] * quadratic(ls.r / r_cut * 3 * 2 - 1.5)
    # ls.vr += a[2] * quadratic((ls.r / r_cut - a[4]) * 1.5 / (1 - a[4]))


def optimize_layer_pseudo(config: Dict, rho_target: DirectField, ions: Ions, total_functional: TotalFunctional):
    grid = rho_target.grid
    r_cut = 1.8
    r = np.linspace(0, r_cut, 21)
    #a = np.asarray([-1.6721515528, 19.0441546583, -0.5170819236, 5.5136191345, 0.8])
    a = np.asarray([0, 0, 0])
    #a = np.asarray([-2.93032635, 1.67275963, -33.30823238])
    vr = np.zeros_like(r)

    dis = DirectField(grid, griddata_3d=np.zeros(grid.nr))
    ls = LayerPseudo(vr=vr, r=r, grid=grid, ions=ions)

    total_functional.UpdateFunctional(newFuncDict={'LS': ls})

    # print(rho_target.integral())
    rho_ini = np.ones_like(rho_target)
    rho_ini *= rho_target.integral() / rho_ini.integral()
    # print(rho_ini.integral())
    opt = Optimization(EnergyEvaluator=total_functional)

    # rho = opt.optimize_rho(guess_rho=rho_ini)
    # print(rho.integral())

    def delta_rho(a):
        #ls.vr = a[0] * (np.cos(ls.r / 1 * 2 * np.pi) - 1) + a[1] * (np.sin(ls.r / 1 * 2 * np.pi))
        set_ls(ls, a, r_cut)
        #ls.vr[ls.r > 2 * a[1]] = 0
        #ls.update_v()
        delta_rho.rho = opt.optimize_rho(guess_rho=rho_ini)
        return 0.5 * (np.abs(delta_rho.rho - rho_target)).integral()

    out = XSF('diff_den_pseudo.xsf')
    # print(delta_rho(a))
    # print(delta_rho.rho.integral())
    #out.write(system, field=rho_target-delta_rho.rho)

    # print(delta_rho(0.19))

    res = minimize(delta_rho, a, method='Powell', tol=1.0e-4)

    sprint(delta_rho(a))
    sprint(res.x)
    sprint(delta_rho(res.x))
    out.write(ions=ions, data=rho_target - delta_rho.rho)

    with open('params', 'w') as f:
        f.write('{0:.10f}\n'.format(r_cut))
        for a_p in res.x:
            f.write('{0:.10f}\n'.format(a_p))

    # res = bisec(delta_rho, -1.984375, -1.84375)
    # sprint(res)

    # return ls, a, r_cut

def runner(config, rho, functionals):
    if config['KEDF2']['kedf'] == 'KS':
        inv = Inverter()
        ext = inv(rho, functionals)
    else:
        vt0 = functionals.funcDict['KineticEnergyFunctional'](rho=rho, calcType = {'V'}).potential
        kedf0 = Functional(type='KEDF', name=config['KEDF2']['kedf'], **config['KEDF2'])
        vn0t0 = kedf0(rho=rho, calcType={'V'}).potential
        ext = ExternalPotential(v=vn0t0-vt0)
    functionals.UpdateFunctional(newFuncDict={'ext': ext})

    realtimerunner = RealTimeRunner(rho, config, functionals)
    realtimerunner()


def RunJob(args):
    sprint("DFTpy {} Begin on : {}".format(__version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if mp.is_mpi:
        info = 'Parallel version (MPI) on {0:>8d} processors'.format(mp.comm.size)
    else:
        info = 'Serial version on {0:>8d} processors'.format(mp.comm.size)
    sprint(info)
    if len(args.confs) == 0:
        args.confs.append(args.input)
    for fname in args.confs:
        config, others = ConfigParser(fname, mp=mp)
        sprint("#" * 80)
        TimeData.Begin("TOTAL")

        optimize_layer_pseudo(config, others['field'], others['ions'], others["E_v_Evaluator"])
        #others["E_v_Evaluator"].funcDict["KineticEnergyFunctional"].options.update({'y': 0})
        #realtimerunner = RealTimeRunner(others['field'], config, others["E_v_Evaluator"])
        #realtimerunner()
        #runner(config, others['field'], others["E_v_Evaluator"])
        TimeData.End("TOTAL")
        TimeData.output(config)
        sprint("-" * 80)
    sprint("#" * 80)
    sprint("DFTpy {} Finished on : {}".format(__version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


def main():
    import sys
    from dftpy.mpi.utils import sprint

    if sys.version_info < (3, 6):
        sprint('Please upgrade your python to version 3.6 or higher.')
        sys.exit()
    args = GetConf()
    RunJob(args)


if __name__ == '__main__':
    main()
