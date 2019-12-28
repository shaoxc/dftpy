import numpy as np
import time
from dftpy.formats.qepp import PP
from dftpy.optimization import Optimization
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.formats.io import read
from dftpy.ewald import ewald
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.field import DirectField, ReciprocalField
from dftpy.math_utils import TimeData, bestFFTsize, interpolation_3d, prolongation
from dftpy.functional_output import Functional
from dftpy.semilocal_xc import LDAStress
from dftpy.pseudo import LocalPseudo
from dftpy.kedf.tf import ThomasFermiStress
from dftpy.kedf.vw import vonWeizsackerStress
from dftpy.kedf.wt import WTStress
from dftpy.hartree import HartreeFunctionalStress


def OptimizeDensityConf(config, ions=None, rhoini=None):
    print('Begin on :', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('#' * 80)
    TimeData.Begin('TOTAL')
    if ions is None:
        ions = read(config['PATH']['cellpath']+config['CELL']['cellfile'], \
                format = config['CELL']['format'], names=config['CELL']['elename'])
    lattice = ions.pos.cell.lattice
    metric = np.dot(lattice.T, lattice)
    if config['GRID']['nr'] is not None:
        nr = np.asarray(config['GRID']['nr'])
    else:
        # spacing = config['GRID']['spacing']
        spacing = config['GRID']['spacing'] * LEN_CONV['Angstrom']['Bohr']
        nr = np.zeros(3, dtype='int32')
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i]) / spacing)
        print('The initial grid size is ', nr)
        for i in range(3):
            nr[i] = bestFFTsize(nr[i])
    print('The final grid size is ', nr)
    if config['MATH']['multistep'] > 1:
        nr2 = nr.copy()
        nr = nr2 // config['MATH']['multistep']
        print('MULTI-STEP: Perform first optimization step')
        print('Grid size of 1  step is ', nr)

    ############################## Grid  ##############################
    grid = DirectGrid(lattice=lattice,
                      nr=nr,
                      units=None,
                      full=config['GRID']['gfull'])
    ############################## PSEUDO  ##############################
    PPlist = {}
    for key in config['PP']:
        ele = key.capitalize()
        PPlist[ele] = config['PATH']['pppath'] + '/' + config['PP'][key]
    optional_kwargs = {}
    optional_kwargs["PP_list"] = PPlist
    optional_kwargs["ions"] = ions
    optional_kwargs["PME"] = config['MATH']['linearie']
    if not ions.Zval:
        if config['CELL']['zval']:
            elename = config['CELL']['elename']
            zval = config['CELL']['zval']
            for ele, z in zip(elename, zval):
                ions.Zval[ele] = z

    linearie = config['MATH']['linearie']
    PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PPlist, PME=linearie)
    KE = FunctionalClass(type='KEDF',
                         name=config['KEDF']['kedf'],
                         **config['KEDF'])
    ############################## XC and Hartree ##############################
    HARTREE = FunctionalClass(type='HARTREE')
    XC = FunctionalClass(type='XC', name=config['EXC']['xc'], **config['EXC'])
    ############################## Initial density ##############################
    zerosA = np.empty(grid.nnr, dtype=float)
    rho_ini = DirectField(grid=grid, griddata_F=zerosA, rank=1)
    density = None
    if rhoini is not None:
        density = rhoini.reshape(rhoini.shape[:3])
    elif config['DENSITY']['densityini'] == 'HEG':
        charge_total = 0.0
        for i in range(ions.nat):
            charge_total += ions.Zval[ions.labels[i]]
        rho_ini[:] = charge_total / ions.pos.cell.volume
        #-----------------------------------------------------------------------
        # rho_ini[:] = charge_total/ions.pos.cell.volume + np.random.random(np.shape(rho_ini)) * 1E-2
        # rho_ini *= (charge_total / (np.sum(rho_ini) * rho_ini.grid.dV ))
        #-----------------------------------------------------------------------
    elif config['DENSITY']['densityini'] == 'Read':
        with open(config['DENSITY']['densityfile'], 'r') as fr:
            line = fr.readline()
            nr0 = list(map(int, line.split()))
            blocksize = 1024 * 8
            strings = ''
            while True:
                line = fr.read(blocksize)
                if not line:
                    break
                strings += line
        density = np.fromstring(strings, dtype=float, sep=' ')
        density = density.reshape(nr0, order='F')
    # normalization
    if density is not None:
        if not np.all(rho_ini.shape[:3] == density.shape[:3]):
            density = interpolation_3d(density, rho_ini.shape[:3])
            # density = prolongation(density)
            density[density < 1E-12] = 1E-12
        rho_ini[:] = density.reshape(rho_ini.shape)
        charge_total = 0.0
        for i in range(ions.nat):
            charge_total += ions.Zval[ions.labels[i]]
        rho_ini *= (charge_total / (np.sum(rho_ini) * rho_ini.grid.dV))
    # rho_ini[:] = density.reshape(rho_ini.shape, order='F')
    ############################## optimization  ##############################
    E_v_Evaluator = TotalEnergyAndPotential(KineticEnergyFunctional=KE,
                                            XCFunctional=XC,
                                            HARTREE=HARTREE,
                                            PSEUDO=PSEUDO)
    if 'Optdensity' in config['JOB']['task']:
        optimization_options = config['OPT']
        optimization_options["econv"] *= ions.nat
        # if config['MATH']['twostep'] :
        # optimization_options["econv"] *= 10
        opt = Optimization(EnergyEvaluator=E_v_Evaluator,
                           optimization_options=optimization_options,
                           optimization_method=config['OPT']['method'])
        rho = opt.optimize_rho(guess_rho=rho_ini)
        # perform second step, dense grid
        #-----------------------------------------------------------------------
        for istep in range(2, config['MATH']['multistep'] + 1):
            if istep == config['MATH']['multistep']:
                nr = nr2
            else:
                nr = nr * 2
            print('#' * 80)
            print('MULTI-STEP: Perform %d optimization step' % istep)
            print('Grid size of %d' % istep, ' step is ', nr)
            grid2 = DirectGrid(lattice=lattice,
                               nr=nr,
                               units=None,
                               full=config['GRID']['gfull'])
            rho_ini = interpolation_3d(rho[..., 0], nr)
            rho_ini[rho_ini < 1E-12] = 1E-12
            rho_ini = DirectField(grid=grid2, griddata_3d=rho_ini, rank=1)
            rho_ini *= (charge_total / (np.sum(rho_ini) * rho_ini.grid.dV))
            ions.restart()
            opt = Optimization(EnergyEvaluator=E_v_Evaluator,
                               optimization_options=optimization_options,
                               optimization_method=config['OPT']['method'])
            rho = opt.optimize_rho(guess_rho=rho_ini)
        optimization_options["econv"] /= ions.nat  # reset the value
    else:
        rho = rho_ini
    ############################## calctype  ##############################
    linearii = config['MATH']['linearii']
    linearie = config['MATH']['linearie']
    energypotential = {}
    forces = {}
    stress = {}
    print('-' * 80)
    if 'Both' in config['JOB']['calctype']:
        print('Calculate Energy and Potential...')
        energypotential = GetEnergyPotential(ions,
                                             rho,
                                             E_v_Evaluator,
                                             calcType='Both',
                                             linearii=linearii,
                                             linearie=linearie)
    elif 'Potential' in config['JOB']['calctype']:
        print('Calculate Potential...')
        energypotential = GetEnergyPotential(ions,
                                             rho,
                                             E_v_Evaluator,
                                             calcType='Potential',
                                             linearii=linearii,
                                             linearie=linearie)
    # elif 'Energy' in config['JOB']['calctype'] :
    else:
        print('Calculate Energy...')
        energypotential = GetEnergyPotential(ions,
                                             rho,
                                             E_v_Evaluator,
                                             calcType='Energy',
                                             linearii=linearii,
                                             linearie=linearie)
    print(format('Energy information', "-^80"))
    for key in sorted(energypotential.keys()):
        if key == 'TOTAL': continue
        value = energypotential[key]
        if key == 'KE':
            ene = 0.0
            for key1 in value:
                print('{:>10s} energy (eV): {:22.15E}'.format(
                    'KE-' + key1,
                    value[key1].energy * ENERGY_CONV['Hartree']['eV']))
                ene += value[key1].energy
            print('{:>10s} energy (eV): {:22.15E}'.format(
                key, ene * ENERGY_CONV['Hartree']['eV']))
        else:
            print('{:>10s} energy (eV): {:22.15E}'.format(
                key, value.energy * ENERGY_CONV['Hartree']['eV']))
    print('{:>10s} energy (eV): {:22.15E}'.format(
        'TOTAL',
        energypotential['TOTAL'].energy * ENERGY_CONV['Hartree']['eV']))
    print('-' * 80)

    etot = energypotential['TOTAL'].energy
    etot_eV = etot * ENERGY_CONV['Hartree']['eV']
    etot_eV_patom = etot * ENERGY_CONV['Hartree']['eV'] / ions.nat
    fstr = '  {:<30s} : {:30.15f}'
    print(fstr.format('total energy (a.u.)', etot))
    print(fstr.format('total energy (eV)', etot_eV))
    print(fstr.format('total energy (eV/atom)', etot_eV_patom))
    ############################## Force ##############################
    if 'Force' in config['JOB']['calctype']:
        print('Calculate Force...')
        forces = GetForces(ions,
                           rho,
                           E_v_Evaluator,
                           linearii=linearii,
                           linearie=linearie,
                           PPlist=None)
        ############################## Output Force ##############################
        f = np.abs(forces['TOTAL'])
        fmax, fmin, fave = np.max(f), np.min(f), np.mean(f)
        fstr_f = ' ' * 8 + '{0:>22s} : {1:<22.5f}'
        print(fstr_f.format('Max force (a.u.)', fmax))
        print(fstr_f.format('Min force (a.u.)', fmin))
        print(fstr_f.format('Ave force (a.u.)', fave))
        # fstr_f =' ' * 16 + '{0:<22s}{1:<22s}{2:<22s}'
        # print(fstr_f.format('Max (a.u.)', 'Min (a.u.)', 'Ave (a.u.)'))
        # fstr_f =' ' * 16 + '{0:<22.5f} {1:<22.5f} {2:<22.5f}'
        # print(fstr_f.format(fmax, fmin, fave))
        print('-' * 80)
    ############################## Stress ##############################
    if 'Stress' in config['JOB']['calctype']:
        print('Calculate Stress...')
        stress = GetStress(ions, rho, E_v_Evaluator, energypotential=energypotential, xc = config['EXC']['xc'], ke = config['KEDF']['kedf'], \
                ke_options = config['KEDF'], linearii = linearii, linearie = linearie, PPlist = None)
        ############################## Output stress ##############################
        fstr_s = ' ' * 16 + '{0[0]:12.5f} {0[1]:12.5f} {0[2]:12.5f}'
        if config['OUTPUT']['stress']:
            for key in sorted(stress.keys()):
                if key == 'TOTAL': continue
                value = stress[key]
                if key == 'KE':
                    kestress = np.zeros((3, 3))
                    for key1 in value:
                        print('{:>10s} stress (a.u.): '.format('KE-' + key1))
                        for i in range(3):
                            print(fstr_s.format(value[key1][i]))
                        kestress += value[key1]
                    print('{:>10s} stress (a.u.): '.format(key))
                    for i in range(3):
                        print(fstr_s.format(kestress[i]))
                else:
                    print('{:>10s} stress (a.u.): '.format(key))
                    for i in range(3):
                        print(fstr_s.format(value[i]))
        print('{:>10s} stress (a.u.): '.format('TOTAL'))
        for i in range(3):
            print(fstr_s.format(stress['TOTAL'][i]))
        print('{:>10s} stress (GPa): '.format('TOTAL'))
        for i in range(3):
            print(
                fstr_s.format(stress['TOTAL'][i] *
                              STRESS_CONV['Ha/Bohr3']['GPa']))
        print('-' * 80)
    ############################## Output Density ##############################
    # print('N', np.sum(rho) * rho.grid.dV)
    # nr2 = (rho.grid.nr + 1)/2
    # nr2 = nr2.astype(np.int32)
    # newrho  = interpolation_3d(rho[..., 0], nr2)
    # rho2 = interpolation_3d(newrho, rho.grid.nr)
    # np.savetxt('lll', np.c_[rho[..., 0].ravel(), rho2.ravel(), (rho[..., 0] - rho2).ravel()])
    #-----------------------------------------------------------------------
    if config['DENSITY']['densityoutput']:
        print('Write Density...')
        outfile = config['DENSITY']['densityoutput']
        with open(outfile, 'w') as fw:
            fw.write('{0[0]:10d} {0[1]:10d} {0[2]:10d}\n'.format(rho.grid.nr))
            size = np.size(rho)
            nl = size // 3
            outrho = rho.ravel(order='F')
            for line in outrho[:nl * 3].reshape(-1, 3):
                fw.write(
                    '{0[0]:22.15E} {0[1]:22.15E} {0[2]:22.15E}\n'.format(line))
            for line in outrho[nl * 3:]:
                fw.write('{0:22.15E}'.format(line))
    ############################## Output ##############################
    results = {}
    results['density'] = rho
    results['energypotential'] = energypotential
    results['forces'] = forces
    results['stress'] = stress
    # print('-' * 31, 'Time information', '-' * 31)
    TimeData.End('TOTAL')
    print(format('Time information', "-^80"))
    print("{:28s}{:24s}{:20s}".format('Label', 'Cost(s)', 'Number'))
    if config['OUTPUT']['time']:
        for key, cost in sorted(TimeData.cost.items(), key=lambda d: d[1]):
            print("{:28s}{:<24.4f}{:<20d}".format(key, cost,
                                                  TimeData.number[key]))
    else:
        key = 'TOTAL'
        print("{:28s}{:<24.4f}{:<20d}".format(key, TimeData.cost[key],
                                              TimeData.number[key]))
    print('#' * 80)
    print('Finished on :', time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime()))
    # TimeData.reset() #Cleanup the data in TimeData
    #-----------------------------------------------------------------------
    return results


def GetEnergyPotential(ions,
                       rho,
                       EnergyEvaluator,
                       calcType='Both',
                       linearii=True,
                       linearie=True):
    energypotential = {}
    energypotential[
        'KE'] = EnergyEvaluator.KineticEnergyFunctional.ComputeEnergyPotential(
            rho, calcType, split=True)
    energypotential[
        'XC'] = EnergyEvaluator.XCFunctional.ComputeEnergyPotential(
            rho, calcType)
    energypotential[
        'HARTREE'] = EnergyEvaluator.HARTREE.ComputeEnergyPotential(
            rho, calcType)
    energypotential['IE'] = EnergyEvaluator.PSEUDO(rho, calcType)
    ewaldobj = ewald(rho=rho, ions=ions, PME=linearii)
    energypotential['II'] = Functional(name='Ewald',
                                       potential=np.zeros_like(rho),
                                       energy=ewaldobj.energy)
    energypotential['TOTAL'] = energypotential['XC'] + energypotential[
        'HARTREE'] + energypotential['IE'] + energypotential['II']
    for key in energypotential['KE']:
        energypotential['TOTAL'] += energypotential['KE'][key]
    return energypotential


def GetForces(ions,
              rho,
              EnergyEvaluator,
              linearii=True,
              linearie=True,
              PPlist=None):
    forces = {}
    forces['IE'] = EnergyEvaluator.PSEUDO.force(rho)
    ewaldobj = ewald(rho=rho, ions=ions, verbose=False, PME=linearii)
    forces['II'] = ewaldobj.forces
    forces['TOTAL'] = forces['IE'] + forces['II']
    return forces

def GetStress(ions, rho, EnergyEvaluator, energypotential=None, energy=None, xc = 'LDA', ke = 'WT', \
        ke_options  = {'x' :1.0, 'y' :1.0}, linearii = True, linearie = True, PPlist = None):

    if energypotential is not None:
        energy = {}
        energy['IE'] = energypotential['IE'].energy
        energy['XC'] = energypotential['XC'].energy
        energy['HARTREE'] = energypotential['HARTREE'].energy
        energy['KE'] = {}
        for key in energypotential['KE']:
            energy['KE'][key] = energypotential['KE'][key].energy
    elif energy is None:
        energy = {
            'XC': None,
            'HARTREE': None,
            'II': None,
            'IE': None,
            'KE': {
                'TF': None,
                'vW': None,
                'NL': None
            }
        }
    ewaldobj = ewald(rho=rho, ions=ions, verbose=False, PME=linearii)
    stress = {}
    if xc == 'LDA':
        stress['XC'] = LDAStress(rho, energy=energy['XC'])
    else:
        raise AttributeError(
            "%s exchange-correlation have not implemented for stress" % xc)
    stress['HARTREE'] = HartreeFunctionalStress(rho, energy=energy['HARTREE'])
    stress['II'] = ewaldobj.stress
    stress['IE'] = EnergyEvaluator.PSEUDO.stress(rho, energy=energy['IE'])
    ############################## KE ##############################
    stress['KE'] = {}
    KEDFNameList = [
        'TF', 'vW', 'x_TF_y_vW', 'TFvW', 'WT'
    ]  # KEDFNLNameList = ['WT','MGP','FP', 'SM'] # is_nonlocal = True
    if ke not in KEDFNameList:
        raise AttributeError("%s KEDF have not implemented for stress" % ke)
    if ke == 'TF':
        stress['KE']['TF'] = ThomasFermiStress(rho,
                                               x=ke_options['x'],
                                               energy=energy['KE']['TF'])
    elif ke == 'vW':
        stress['KE']['vW'] = vonWeizsackerStress(rho,
                                                 y=ke_options['y'],
                                                 energy=energy['KE']['vW'])
    else:
        stress['KE']['TF'] = ThomasFermiStress(rho,
                                               x=ke_options['x'],
                                               energy=energy['KE']['TF'])
        stress['KE']['vW'] = vonWeizsackerStress(rho,
                                                 y=ke_options['y'],
                                                 energy=energy['KE']['vW'])
        if ke == 'WT':
            stress['KE']['NL'] = WTStress(rho, energy=energy['KE']['NL'])
    stress['TOTAL'] = stress['XC'] + stress['HARTREE'] + stress['II'] + stress[
        'IE']
    for key in stress['KE']:
        stress['TOTAL'] += stress['KE'][key]
    for i in range(1, 3):
        for j in range(i - 1, -1, -1):
            stress['TOTAL'][i, j] = stress['TOTAL'][j, i]

    return stress
