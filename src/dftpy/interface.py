import numpy as np
import os
from dftpy.mpi import sprint
from dftpy.optimization import Optimization
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.constants import LEN_CONV, ENERGY_CONV, STRESS_CONV
from dftpy.formats.io import read_density, write, read
from dftpy.ewald import ewald
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import ecut2nr
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.semilocal_xc import XCStress
from dftpy.functional.kedf import KEDFStress
from dftpy.functional.hartree import HartreeFunctionalStress
from dftpy.config.config import PrintConf, ReadConf
from dftpy.inverter import Inverter
from dftpy.properties import get_electrostatic_potential
from dftpy.utils import field2distrib


def ConfigParser(config, ions=None, rhoini=None, pseudo=None, grid=None, mp = None):
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        # config is a file
        config = ReadConf(config)
        PrintConf(config)

    # check the input
    if grid is not None and config["MATH"]["multistep"] > 1:
        raise AttributeError("Given the 'grid', can't use 'multistep' method anymore.")

    if ions is None:
        ions, field, _ = read(
            config["PATH"]["cellpath"] +os.sep+ config["CELL"]["cellfile"],
            format=config["CELL"]["format"],
            names=config["CELL"]["elename"],
            kind = 'all',
        )
    else :
        field = None
    lattice = ions.cell
    if config["GRID"]["nr"] is not None:
        nr = np.asarray(config["GRID"]["nr"])
    elif field is not None and field is not None :
        nr = field.grid.nr
        rhoini = field
    else:
        spacing = config["GRID"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
        grid_options = config["GRID"].copy()
        grid_options.pop("spacing", None)
        nr = ecut2nr(lattice=lattice, spacing=spacing, **grid_options)
    sprint("The final grid size is ", nr)
    nr2 = nr.copy()
    if config["MATH"]["multistep"] > 1:
        nr = nr2 // config["MATH"]["multistep"]
        sprint("MULTI-STEP: Perform first optimization step")
        sprint("Grid size of 1  step is ", nr)

    ############################## Grid  ##############################
    if grid is None:
        grid = DirectGrid(lattice=lattice, nr=nr, full=config["GRID"]["gfull"], cplx=config["GRID"]["cplx"], mp=mp)
    if mp is None : mp = grid.mp
    ############################## PSEUDO  ##############################
    PPlist = {}
    for key in config["PP"]:
        ele = key.capitalize()
        PPlist[ele] = config["PATH"]["pppath"] +os.sep+ config["PP"][key]
    optional_kwargs = {}
    optional_kwargs["PP_list"] = PPlist
    optional_kwargs["ions"] = ions
    optional_kwargs["PME"] = config["MATH"]["linearie"]
    #
    linearie = config["MATH"]["linearie"]
    if pseudo is None:
        # PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PPlist, PME=linearie)
        PSEUDO = Functional(type ='PSEUDO', grid=grid, ions=ions, PP_list=PPlist, PME=linearie)
    else:
        PSEUDO = pseudo

        PSEUDO.restart(full=False)
        PSEUDO.grid = grid
        PSEUDO.ions = ions
    kedf_config = config["KEDF"].copy()
    if kedf_config.get('temperature', None):
        kedf_config['temperature'] *= ENERGY_CONV['eV']['Hartree']
    if kedf_config.get('temperature0', None):
        kedf_config['temperature0'] *= ENERGY_CONV['eV']['Hartree']
    KE = Functional(type="KEDF", name=config["KEDF"]["kedf"], **kedf_config)
    ############################## XC and Hartree ##############################
    HARTREE = Functional(type="HARTREE")
    XC = Functional(type="XC", name=config["EXC"]["xc"], **config["EXC"])
    if config["NONADIABATIC"]["nonadiabatic"] is None:
        DYNAMIC = None
    else:
        DYNAMIC = Functional(type="DYNAMIC", name=config["NONADIABATIC"]["nonadiabatic"], **config["NONADIABATIC"])
    E_v_Evaluator = TotalFunctional(KineticEnergyFunctional=KE, XCFunctional=XC, HARTREE=HARTREE, PSEUDO=PSEUDO,
                                DYNAMIC=DYNAMIC)
    ############################## Initial density ##############################
    nspin=config["DENSITY"]["nspin"]
    rho_ini = DirectField(grid=grid, rank=nspin)
    density = None
    root = 0
    if rhoini is not None:
        density = rhoini
    elif config["DENSITY"]["densityini"] == "HEG":
        charge_total = ions.get_ncharges()
        rho_ini[:] = charge_total / ions.cell.volume
        if nspin>1 : rho_ini[:] /= nspin
    elif config["DENSITY"]["densityini"] == "Read":
        density = []
        if mp.rank == root :
            density = read_density(config["DENSITY"]["densityfile"])

    if density is not None:
        field2distrib(density, rho_ini, root = root)
    # normalization the charge (e.g. the cell changed)
    charge_total = ions.get_ncharges()
    rho_ini *= charge_total / np.sum(rho_ini.integral())
    ############################## add spin magmom ##############################
    if nspin > 1 :
        magmom = config["DENSITY"]["magmom"]
        for ip in range(nspin):
            rho_ini[ip] += 1.0/nspin *(-1)**ip * magmom/ ions.cell.volume
    #-----------------------------------------------------------------------
    # The last is a dictionary, which return some properties are used for different situations.
    others = {
        "ions": ions,
        "field": rho_ini,
        "E_v_Evaluator": E_v_Evaluator,
        "nr2": nr2,
    }
    return config, others


def OptimizeDensityConf(config, ions, rho, E_v_Evaluator, nr2 = None):
    rho_ini = rho
    grid = rho_ini.grid
    if hasattr(E_v_Evaluator, 'PSEUDO'):
        PSEUDO=E_v_Evaluator.PSEUDO
    nr = grid.nr
    charge_total = ions.get_ncharges()
    if "Optdensity" in config["JOB"]["task"]:
        optimization_options = config["OPT"]
        optimization_options["econv"] *= ions.nat
        # if config['MATH']['twostep'] :
        # optimization_options["econv"] *= 10
        opt = Optimization(
            EnergyEvaluator=E_v_Evaluator,
            optimization_options=optimization_options,
            optimization_method=config["OPT"]["method"],
        )
        rho = opt.optimize_rho(guess_rho=rho_ini)
        # perform second step, dense grid
        # -----------------------------------------------------------------------
        for istep in range(2, config["MATH"]["multistep"] + 1):
            if istep == config["MATH"]["multistep"]:
                if nr2 is None :
                    nr2 = nr * 2
                nr = nr2
            else:
                nr = nr * 2
            sprint("#" * 80)
            sprint("MULTI-STEP: Perform %d optimization step" % istep)
            sprint("Grid size of %d" % istep, " step is ", nr)
            grid2 = DirectGrid(lattice=grid.lattice, nr=nr, full=config["GRID"]["gfull"], mp=grid.mp)
            rho_ini = DirectField(grid=grid2, rank=rho.rank)
            field2distrib(rho, rho_ini, root = 0)
            rho_ini *= charge_total / (np.sum(rho_ini.integral()))
            # ions.restart()
            if hasattr(E_v_Evaluator, 'PSEUDO'):
                PSEUDO=E_v_Evaluator.PSEUDO
                PSEUDO.restart(full=False, ions=PSEUDO.ions, grid=grid2)
            opt = Optimization(
                EnergyEvaluator=E_v_Evaluator,
                optimization_options=optimization_options,
                optimization_method=config["OPT"]["method"],
            )
            rho = opt.optimize_rho(guess_rho=rho_ini)
        optimization_options["econv"] /= ions.nat  # reset the value
    ############################## calctype  ##############################
    linearii = config["MATH"]["linearii"]
    linearie = config["MATH"]["linearie"]
    energypotential = {}
    forces = {}
    stress = {}
    sprint("-" * 80)
    if "Both" in config["JOB"]["calctype"]:
        sprint("Calculate Energy and Potential...")
        energypotential = GetEnergyPotential(
            ions, rho, E_v_Evaluator, calcType=["E","V"], linearii=linearii, linearie=linearie
        )
    elif "Potential" in config["JOB"]["calctype"]:
        sprint("Calculate Potential...")
        energypotential = GetEnergyPotential(
            ions, rho, E_v_Evaluator, calcType=["V"], linearii=linearii, linearie=linearie
        )
    elif "Density" in config["JOB"]["calctype"]:
        sprint("Only return density...")
        energypotential = {'TOTAL' : FunctionalOutput(name ='TOTAL', energy = 0.0)}
    else:
        sprint("Calculate Energy...")
        energypotential = GetEnergyPotential(
            ions, rho, E_v_Evaluator, calcType=["E"], linearii=linearii, linearie=linearie
        )
    sprint(format("Energy information", "-^80"))
    keys, ep = zip(*energypotential.items())
    values = [item.energy for item in ep]
    values = rho.mp.vsum(values)
    ep_w = dict(zip(keys, values))
    ke_energy = 0.0
    for key in sorted(keys):
        if key == "TOTAL":
            continue
        value = ep_w[key]
        sprint("{:>10s} energy (eV): {:22.15E}".format(key, ep_w[key]* ENERGY_CONV["Hartree"]["eV"]))
        if key.startswith('KEDF'): ke_energy += ep_w[key]
    etot = ep_w['TOTAL']
    sprint("{:>10s} energy (eV): {:22.15E}".format("TOTAL", etot * ENERGY_CONV["Hartree"]["eV"]))
    sprint("-" * 80)

    etot_eV = etot * ENERGY_CONV["Hartree"]["eV"]
    etot_eV_patom = etot * ENERGY_CONV["Hartree"]["eV"] / ions.nat
    fstr = "  {:<30s} : {:30.15f}"
    sprint(fstr.format("kedfs energy (a.u.)", ke_energy))
    sprint(fstr.format("kedfs energy (eV)", ke_energy* ENERGY_CONV["Hartree"]["eV"]))
    sprint(fstr.format("total energy (a.u.)", etot))
    sprint(fstr.format("total energy (eV)", etot_eV))
    sprint(fstr.format("total energy (eV/atom)", etot_eV_patom))
    ############################## Force ##############################
    if "Force" in config["JOB"]["calctype"]:
        sprint("Calculate Force...")
        forces = GetForces(ions, rho, E_v_Evaluator, linearii=linearii, linearie=linearie, PPlist=None)
        ############################## Output Force ##############################
        f = np.abs(forces["TOTAL"])
        fmax, fmin, fave = np.max(f), np.min(f), np.mean(f)
        fstr_f = " " * 8 + "{0:>22s} : {1:<22.5f}"
        sprint(fstr_f.format("Max force (a.u.)", fmax))
        sprint(fstr_f.format("Min force (a.u.)", fmin))
        sprint(fstr_f.format("Ave force (a.u.)", fave))
        # fstr_f =' ' * 16 + '{0:<22s}{1:<22s}{2:<22s}'
        # sprint(fstr_f.format('Max (a.u.)', 'Min (a.u.)', 'Ave (a.u.)'))
        # fstr_f =' ' * 16 + '{0:<22.5f} {1:<22.5f} {2:<22.5f}'
        # sprint(fstr_f.format(fmax, fmin, fave))
        sprint("-" * 80)
    ############################## Stress ##############################
    if "Stress" in config["JOB"]["calctype"]:
        sprint("Calculate Stress...")
        stress = GetStress(
            ions,
            rho,
            E_v_Evaluator,
            energypotential=energypotential,
            xc_options=config["EXC"],
            ke_options=config["KEDF"],
            linearii=linearii,
            linearie=linearie,
            PPlist=None,
        )
        ############################## Output stress ##############################
        fstr_s = " " * 16 + "{0[0]:12.5f} {0[1]:12.5f} {0[2]:12.5f}"
        if config["OUTPUT"]["stress"]:
            for key in sorted(stress.keys()):
                if key == "TOTAL":
                    continue
                value = stress[key]
                sprint("{:>10s} stress (a.u.): ".format(key))
                for i in range(3):
                    sprint(fstr_s.format(value[i]))
        sprint("{:>10s} stress (a.u.): ".format("TOTAL"))
        for i in range(3):
            sprint(fstr_s.format(stress["TOTAL"][i]))
        sprint("{:>10s} stress (GPa): ".format("TOTAL"))
        for i in range(3):
            sprint(fstr_s.format(stress["TOTAL"][i] * STRESS_CONV["Ha/Bohr3"]["GPa"]))
        sprint("-" * 80)
    ############################## Output Density ##############################
    if config["DENSITY"]["densityoutput"]:
        sprint("Write Density...")
        outfile = config["DENSITY"]["densityoutput"]
        write(outfile, rho, ions)
    ############################## Output ##############################
    if config["OUTPUT"]["electrostatic_potential"]:
        sprint("Write electrostatic potential...")
        outfile = config["OUTPUT"]["electrostatic_potential"]
        v = get_electrostatic_potential(rho, E_v_Evaluator)
        write(outfile, v, ions)
    results = {}
    results["density"] = rho
    results["energypotential"] = energypotential
    results["forces"] = forces
    results["stress"] = stress
    if hasattr(E_v_Evaluator, 'PSEUDO'):
        results["pseudo"] = PSEUDO
    # sprint('-' * 31, 'Time information', '-' * 31)
    # -----------------------------------------------------------------------
    return results


def GetEnergyPotential(ions, rho, EnergyEvaluator, calcType={"E","V"}, linearii=True, linearie=True):
    energypotential = {}
    ewaldobj = ewald(rho=rho, ions=ions, PME=linearii)
    energypotential["II"] = FunctionalOutput(name="Ewald", potential=np.zeros_like(rho), energy=ewaldobj.energy)

    energypotential["TOTAL"] = energypotential["II"].copy()
    funcDict = EnergyEvaluator.funcDict
    for key, func in funcDict.items():
        if func.type == "KEDF":
            results = func(rho, calcType=calcType, split=True)
            for key2 in results:
                energypotential["TOTAL"] += results[key2]
                energypotential["KEDF-" + key2.split('-')[-1]] = results[key2]
        else :
            results = func(rho, calcType=calcType)
            energypotential["TOTAL"] += results
            energypotential[func.type] = results

    # keys, ep = zip(*energypotential.items())
    # values = [item.energy for item in ep]
    # values = mp.vsum(values)
    # for item, v in zip(ep, values):
        # item.energy = v
    return energypotential


def GetForces(ions, rho, EnergyEvaluator, linearii=True, linearie=True, PPlist=None):
    forces = {}
    forces["PSEUDO"] = EnergyEvaluator.PSEUDO.force(rho)
    ewaldobj = ewald(rho=rho, ions=ions, verbose=False, PME=linearii)
    forces["II"] = ewaldobj.forces
    forces["TOTAL"] = forces["PSEUDO"] + forces["II"]
    forces["TOTAL"] = rho.mp.vsum(forces["TOTAL"])
    return forces


def GetStress_old(
    ions,
    rho,
    EnergyEvaluator,
    energypotential=None,
    energy=None,
    xc_options = {'xc' : 'LDA'},
    ke_options={"kedf" : "WT", "x": 1.0, "y": 1.0},
    linearii=True,
    linearie=True,
    PPlist=None,
):
    """
    Get stress tensor
    """
    #-----------------------------------------------------------------------
    KEDF_Stress_L= {
            "TF" : ["TF"],
            "VW": ["VW"],
            "X_TF_Y_VW": ["TF", "VW"],
            "TFVW": ["TF", "VW"],
            "WT": ["TF", "VW", "NL"],
            }
    ke = ke_options["kedf"]
    if ke not in KEDF_Stress_L :
        raise AttributeError("%s KEDF have not implemented for stress" % ke)
    kelist = KEDF_Stress_L[ke]
    #-----------------------------------------------------------------------
    #Initial energy dict
    energy = {}
    if energypotential is not None:
        for key in energypotential :
            energy[key] = energypotential[key].energy
    elif energy is None :
        funcDict = EnergyEvaluator.funcDict
        for key in funcDict :
            func = getattr(EnergyEvaluator, key)
            if func.type.startwith('KEDF'):
                for item in kelist :
                    energy['KEDF-' + item] = None
            else :
                energy[func.type] = None

    stress = {}
    stress['TOTAL'] = np.zeros((3, 3))

    for key1 in energy :
        if key1 == "TOTAL" :
            continue
        elif key1 == "II" :
            ewaldobj = ewald(rho=rho, ions=ions, verbose=False, PME=linearii)
            stress[key1] = ewaldobj.stress
        elif key1 == "XC" :
            stress[key1] = XCStress(rho, energy=energy[key1], **xc_options)
        elif key1 == 'HARTREE' :
            stress[key1] = HartreeFunctionalStress(rho, energy=energy[key1])
        elif key1 == 'PSEUDO' :
            stress[key1] = EnergyEvaluator.PSEUDO.stress(rho, energy=energy[key1])
        elif key1.startswith('KEDF') :
            if "TF" in key1 :
                stress[key1] = KEDFStress(rho, name="TF", energy=energy[key1], **ke_options)
            if "VW" in key1 :
                stress[key1] = KEDFStress(rho, name="VW", energy=energy[key1], **ke_options)
            if 'NL' in key1 :
                stress[key1] = KEDFStress(rho, name=ke, energy=energy[key1], **ke_options)
        else :
            raise AttributeError("%s have not implemented for stress" % key1)
        stress['TOTAL'] += stress[key1]

    for i in range(1, 3):
        for j in range(i - 1, -1, -1):
            stress["TOTAL"][i, j] = stress["TOTAL"][j, i]

    keys, values = zip(*stress.items())
    values = rho.mp.vsum(values)
    stress = dict(zip(keys, values))

    return stress

def InvertRunner(config, ions, EnergyEvaluater):
    file_rho_in = config["INVERSION"]["rho_in"]
    file_v_out = config["INVERSION"]["v_out"]
    field = read_density(file_rho_in)

    if ions.cell != field.cell :
        raise ValueError('The grid of the input density does not match the grid of the system')

    inv = Inverter()
    ext = inv(field, EnergyEvaluater)
    write(file_v_out, data=ext.v, ions=ions, data_type='potential')

    return ext

def GetStress(
    ions,
    rho,
    EnergyEvaluator,
    linearii=True,
    ewaldobj= None,
    **kwargs
):
    """
    Get stress tensor
    """
    #-----------------------------------------------------------------------
    stress = {}
    if ewaldobj is None :
        ewaldobj = ewald(rho=rho, ions=ions, verbose=False, PME=linearii)
    stress['II'] = ewaldobj.stress
    stress['TOTAL'] = stress['II'].copy()

    funcDict = EnergyEvaluator.funcDict
    for key, func in funcDict.items():
        if func.type == "KEDF":
            results = func.stress(rho, split=True)
            for key2 in results:
                stress["TOTAL"] += results[key2]
                stress["KEDF-" + key2.split('-')[-1]] = results[key2]
        else :
            results = func.stress(rho)
            stress["TOTAL"] += results
            stress[func.type] = results

    for i in range(1, 3):
        for j in range(i - 1, -1, -1):
            stress["TOTAL"][i, j] = stress["TOTAL"][j, i]

    keys, values = zip(*stress.items())
    values = rho.mp.vsum(values)
    stress = dict(zip(keys, values))

    return stress
