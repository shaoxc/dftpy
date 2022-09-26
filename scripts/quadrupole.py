from qepy.driver import Driver
from qepy.io import QEInput
from typing import Dict

import numpy as np
from dftpy.field import DirectField
from dftpy.optimization import Optimization
from dftpy.functional import TotalFunctional, Functional, ExternalPotential
from dftpy.ions import Ions
from collections import defaultdict
from dftpy.inverter import Inverter
from dftpy.mpi import mp, sprint
from dftpy.cui.main import GetConf
from dftpy import __version__
import time
from dftpy.time_data import TimeData
from dftpy.interface import ConfigParser
from dftpy.formats.npy import read, write
from os.path import exists



def saw(emaxpos, eopreg, x):
    sawout = np.zeros_like(x)
    z = x - emaxpos
    y = z - np.floor(z)
    mask = y <= eopreg
    mask2 = np.invert(mask)
    sawout[mask] = (0.5-y[mask]/eopreg)*(1.0-eopreg)
    sawout[mask2] = (-0.5+(y[mask2]-eopreg)/(1.0-eopreg))*(1.0-eopreg)
    return sawout


def dipole(drho: DirectField, direction: int) -> float:
    grid = drho.grid
    dip = (drho * grid.r[direction]).integral()
    return dip


def quadrupole(drho: DirectField, direction1: int, direction2: int) -> float:
    grid = drho.grid
    tmp = 3 * grid.r[direction1] * grid.r[direction2]
    if direction1 == direction2:
        tmp -= grid.rr
    quadp = (drho * tmp).integral()
    return quadp


def qe_optimize(atoms, options, emaxpos, eopreg, eamp, direction):
    options = QEInput.update_atoms(atoms, qe_options=options)
    sprint(options)
    QEInput().write_qe_input('qe.in', qe_options=options)
    driver = Driver(qe_options=options)
    driver.scf()
    rho_ks = driver.data2field(driver.get_density().copy())
    ext = np.empty_like(rho_ks)
    ext[:] = saw(emaxpos, eopreg, rho_ks.grid.s[direction]) * eamp * 2
    ext = ext.get_3dinterpolation(rho_ks.grid.nr)
    #sprint(ext[:,0,0])
    driver.set_external_potential(ext.ravel('F'))
    driver.scf()
    rho_ks2 = driver.data2field(driver.get_density().copy())
    drho_ks = rho_ks2 - rho_ks

    return rho_ks, drho_ks


def qe_options(config):
    options = {
        '&control': {
            'calculation': "'scf'",
            'prefix': "'Mg8'",
            'pseudo_dir': "'{0:s}'".format(config["PATH"]["pppath"]),
            'restart_mode': "'from_scratch'"},
        '&system': {
            'ibrav': 0,
            'nosym': ".true.",
            'degauss': 0.02,
            'ecutwfc': 50,
            'occupations': "'smearing'",
            'smearing': "'gaussian'"},
        '&electrons': {
            'mixing_beta': 0.5,
            'electron_maxstep': 200},
        'atomic_species': [],
        'k_points gamma': [],
    }
    for key, value in config["PP"].items():
        options['atomic_species'].append('{0:s} {1:f} {2:s}'.format(key, 1, value))
    return options


def den_diff(rho1: DirectField, rho2: DirectField) -> float:
    return np.abs(rho1 - rho2).integral() / 2


def runner(config: Dict, rho_ini: DirectField, ions: Ions, functionals: TotalFunctional):
    lmgp = functionals["KineticEnergyFunctional"]
    if lmgp.name != 'LMGP':
        lmgp = Functional(type='KEDF', name='LMGP')
    lkt = Functional(type='KEDF', name='LKT')
    vw = Functional(type='KEDF', name='VW')

    kes = {'LMGP': lmgp,
           'LKT': lkt}
    keds = list(kes.keys())
    keds.append('KS')

    emaxpos = 0.8
    eopreg = 0.4
    eamp = 0.01
    direction = config["TD"]["direc"]
    direct_txt = ['x', 'y', 'z']
    v = np.empty_like(rho_ini)
    grid = rho_ini.grid
    s = grid.s
    v[:] = saw(emaxpos, eopreg, s[direction]) * eamp
    ext_field = ExternalPotential(v=v)

    rhos = dict()
    drhos = defaultdict(dict)

    if exists('rho_KS.npy') and exists('drho_KS_KS.npy'):
        rhos['KS'] = read('rho_KS.npy', grid=grid)
        drhos['KS']['KS'] = read('drho_KS_KS.npy', grid=grid)
        rhos['KS'] = DirectField(grid=grid, data=rhos['KS'])
        rhos['KS'] = np.abs(rhos['KS'])
        drhos['KS']['KS'] = DirectField(grid=grid, data=drhos['KS']['KS'])
    else:
        atoms = ions.to_ase()
        rhos['KS'], drhos['KS']['KS'] = qe_optimize(atoms, qe_options(config), emaxpos, eopreg, eamp, direction)
        rhos['KS'] = rhos['KS'].get_3dinterpolation(grid.nr)
        rhos['KS'] = np.abs(rhos['KS'])
        drhos['KS']['KS'] = drhos['KS']['KS'].get_3dinterpolation(grid.nr)

        write('rho_KS.npy', data=rhos['KS'])
        write('drho_KS_KS.npy', data=drhos['KS']['KS'])

    optimization_options = config["OPT"]
    optimization_options["econv"] *= ions.nat
    optimizer = Optimization(
        EnergyEvaluator=functionals,
        optimization_options=optimization_options,
        optimization_method=config["OPT"]["method"],
    )

    for key, ke in kes.items():
        fname = 'rho_{0:s}.npy'.format(key)
        if exists(fname):
            sprint('Read density ', key)
            rhos[key] = read(fname, grid=grid)
            rhos[key] = DirectField(grid=grid, data=rhos[key])
        else:
            sprint('Optimize density ', key)
            functionals["KineticEnergyFunctional"] = ke
            rhos[key] = optimizer(guess_rho=rho_ini)
            write(fname, data=rhos[key])

    for key, ke in kes.items():
        functionals["KineticEnergyFunctional"] = ke
        functionals_ext = TotalFunctional(**functionals.funcDict)
        functionals_ext.UpdateFunctional(newFuncDict={'EXTFIELD': ext_field})
        for ked in keds:
            fname = 'drho_{0:s}_{1:s}.npy'.format(key, ked)
            if exists(fname):
                sprint('Read density ', key, ' on ', ked)
                drhos[key][ked] = read(fname, grid=grid)
                drhos[key][ked] = DirectField(grid=grid, data=drhos[key][ked])
            else:
                sprint('Optimize density ', key, ' on ', ked)
                rho_ini = rhos[ked]
                if ked == 'KS':
                    inv = Inverter()
                    ext = inv(rho_ini, functionals)
                    v_vw = vw(rho_ini).potential
                    correct_v = ExternalPotential(v=ext.v+v_vw)
                else:
                    v_ked = kes[ked](rho_ini).potential
                    v_ke = kes[key](rho_ini).potential
                    correct_v = ExternalPotential(v=v_ked-v_ke)
                functionals_ext.UpdateFunctional(newFuncDict={'CORRECTION': correct_v})
                optimizer.EnergyEvaluator = functionals_ext
                new_rho = optimizer(guess_rho=rho_ini)
                drhos[key][ked] = new_rho - rho_ini
                write(fname, data=drhos[key][ked])

    sprint('Absolute value')
    for ke in drhos:
        for ked in drhos[ke]:
            sprint(ke + ' in ' + ked)
            sprint('dipoles')
            for direct in range(3):
                sprint('{0:s}: {1:.10e}'.format(direct_txt[direct], dipole(drhos[ke][ked], direct)))
            sprint('quadrupoles')
            for direct1 in range(3):
                for direct2 in range(direct1, 3):
                    sprint('{0:s}{1:s}: {2:.10e}'.format(direct_txt[direct1], direct_txt[direct2],
                                                         quadrupole(drhos[ke][ked], direct1, direct2)))

    sprint('\nDifference')
    for ke in drhos:
        if ke == 'KS':
            continue
        for ked in drhos[ke]:
            sprint(ke + ' in ' + ked)
            sprint('dipoles')
            for direct in range(3):
                sprint('{0:s}: {1:.10e}'.format(direct_txt[direct], np.abs(dipole(drhos[ke][ked], direct)
                                                                           - dipole(drhos['KS']['KS'], direct))))
            sprint('quadrupoles')
            for direct1 in range(3):
                for direct2 in range(direct1, 3):
                    sprint('{0:s}{1:s}: {2:.10e}'.format(direct_txt[direct1], direct_txt[direct2],
                                                         np.abs(quadrupole(drhos[ke][ked], direct1, direct2) -
                                                                quadrupole(drhos['KS']['KS'], direct1, direct2))))

    rho_ks = rhos['KS'] + drhos['KS']['KS']
    for key in kes:
        rho = rhos[key] + drhos[key][key]
        sprint('{0:s}: {1:.10e}'.format(key, den_diff(rho, rho_ks)))


def main():
    args = GetConf()
    sprint("DFTpy {} Begin on : {}".format(__version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if len(args.confs) == 0:
        args.confs.append(args.input)
    for fname in args.confs:
        config, others = ConfigParser(fname, mp=mp)
        sprint("#" * 80)
        TimeData.Begin("TOTAL")
        runner(config, others['field'], others['ions'], others["E_v_Evaluator"])
        TimeData.End("TOTAL")
        TimeData.output(config)
        sprint("-" * 80)
    sprint("#" * 80)
    sprint("DFTpy {} Finished on : {}".format(__version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == '__main__':
    main()



