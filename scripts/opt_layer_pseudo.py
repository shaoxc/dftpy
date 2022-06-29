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
from dftpy.functional import Functional
from dftpy.functional.pseudo.layer_pseudo import LayerPseudo
from dftpy.functional.total_functional import TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.interface import ConfigParser
from dftpy.math_utils import interpolation_3d
from dftpy.mpi import mp, sprint
from dftpy.optimization import Optimization
from dftpy.time_data import TimeData
from dftpy.mpi import sprint



def optimize_layer_pseudo(config: Dict, system: System, total_functional: TotalFunctional):
    grid = system.field.grid
    r_cut = 3
    r = np.linspace(0, r_cut, 31)
    a = np.asarray([1, 1])
    vr = np.zeros_like(r)

    dis = DirectField(grid, griddata_3d=np.zeros(grid.nr))
    ls = LayerPseudo(vr=vr, r=r, grid=grid, ions=system.ions)

    total_functional.UpdateFunctional(newFuncDict={'LS': ls})

    rho_target = system.field
    # print(rho_target.integral())
    rho_ini = np.ones_like(rho_target)
    rho_ini *= rho_target.integral() / rho_ini.integral()
    # print(rho_ini.integral())
    opt = Optimization(EnergyEvaluator=total_functional)

    # rho = opt.optimize_rho(guess_rho=rho_ini)
    # print(rho.integral())

    def delta_rho(a):
        ls.vr = a[0] * (np.cos(ls.r / 1 * 2 * np.pi) - 1) + a[1] * (np.sin(ls.r / 1 * 2 * np.pi))
        #ls.vr[ls.r > 2 * a[1]] = 0
        #ls.update_v()
        rho = opt.optimize_rho(guess_rho=rho_ini)
        return 0.5 * (np.abs(rho - rho_target)).integral()

    # print(delta_rho(0.19))

    res = minimize(delta_rho, a)

    sprint(delta_rho(a))
    sprint(res.x)
    sprint(delta_rho(res.x))


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

        optimize_layer_pseudo(config, others['struct'], others["E_v_Evaluator"])

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