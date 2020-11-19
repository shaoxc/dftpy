import numpy as np
import time
import os
from dftpy.optimization import Optimization
from dftpy.functionals import FunctionalClass, TotalEnergyAndPotential
from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
from dftpy.formats.io import read, read_density, write
from dftpy.ewald import ewald
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import bestFFTsize, interpolation_3d
from dftpy.time_data import TimeData
from dftpy.functional_output import Functional
from dftpy.semilocal_xc import LDAStress, PBEStress, XCStress
from dftpy.pseudo import LocalPseudo
from dftpy.kedf import KEDFStress
from dftpy.hartree import HartreeFunctionalStress
from dftpy.config.config import PrintConf, ReadConf
from dftpy.system import System
from dftpy.functional_output import Functional
from dftpy.inverter import Inverter
from dftpy.formats.xsf import XSF
from dftpy.formats.qepp import PP
from functools import reduce


def ConfigParser(config, ions=None, rhoini=None, pseudo=None, grid=None):
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
        ions = read(
            config["PATH"]["cellpath"] +os.sep+ config["CELL"]["cellfile"],
            format=config["CELL"]["format"],
            names=config["CELL"]["elename"],
        )
    lattice = ions.pos.cell.lattice
    metric = np.dot(lattice.T, lattice)
    if config["GRID"]["nr"] is not None:
        nr = np.asarray(config["GRID"]["nr"])
    else:
        # spacing = config['GRID']['spacing']
        spacing = config["GRID"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
        nr = np.zeros(3, dtype="int32")
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i]) / spacing)
        print("The initial grid size is ", nr)
        for i in range(3):
            nr[i] = bestFFTsize(nr[i], **config["GRID"])
    print("The final grid size is ", nr)
    nr2 = nr.copy()
    if config["MATH"]["multistep"] > 1:
        nr = nr2 // config["MATH"]["multistep"]
        print("MULTI-STEP: Perform first optimization step")
        print("Grid size of 1  step is ", nr)

    ############################## Grid  ##############################
    if grid is None:
        grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=config["GRID"]["gfull"])
    ############################## PSEUDO  ##############################
    PPlist = {}
    for key in config["PP"]:
        ele = key.capitalize()
        PPlist[ele] = config["PATH"]["pppath"] +os.sep+ config["PP"][key]
    optional_kwargs = {}
    optional_kwargs["PP_list"] = PPlist
    optional_kwargs["ions"] = ions
    optional_kwargs["PME"] = config["MATH"]["linearie"]
    if not ions.Zval:
        if config["CELL"]["zval"]:
            elename = config["CELL"]["elename"]
            zval = config["CELL"]["zval"]
            for ele, z in zip(elename, zval):
                ions.Zval[ele] = z

    linearie = config["MATH"]["linearie"]
    if pseudo is None:
        # PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PPlist, PME=linearie)
        PSEUDO = FunctionalClass(type = 'PSEUDO', grid=grid, ions=ions, PP_list=PPlist, PME=linearie)
    else:
        if isinstance(pseudo, FunctionalClass):
            PSEUDO = pseudo.PSEUDO
        else :
            PSEUDO = pseudo

        PSEUDO.restart(full=False)
        PSEUDO.grid = grid
        PSEUDO.ions = ions
    KE = FunctionalClass(type="KEDF", name=config["KEDF"]["kedf"], **config["KEDF"])
    ############################## XC and Hartree ##############################
    HARTREE = FunctionalClass(type="HARTREE")
    XC = FunctionalClass(type="XC", name=config["EXC"]["xc"], **config["EXC"])
    ############################## Initial density ##############################
    zerosA = np.empty(grid.nnr, dtype=float)
    rho_ini = DirectField(grid=grid, griddata_C=zerosA, rank=1)
    density = None
    if rhoini is not None:
        density = rhoini.reshape(rhoini.shape[:3])
    elif config["DENSITY"]["densityini"] == "HEG":
        charge_total = 0.0
        for i in range(ions.nat):
            charge_total += ions.Zval[ions.labels[i]]
        rho_ini[:] = charge_total / ions.pos.cell.volume
        # -----------------------------------------------------------------------
        # rho_ini[:] = charge_total/ions.pos.cell.volume + np.random.random(np.shape(rho_ini)) * 1E-2
        # rho_ini *= (charge_total / (np.sum(rho_ini) * rho_ini.grid.dV ))
        # -----------------------------------------------------------------------
    elif config["DENSITY"]["densityini"] == "Read":
        density = read_density(config["DENSITY"]["densityfile"])
    # normalization
    if density is not None:
        if not np.all(rho_ini.shape[:3] == density.shape[:3]):
            density = interpolation_3d(density, rho_ini.shape[:3])
            # density = prolongation(density)
            density[density < 1e-12] = 1e-12
        rho_ini[:] = density.reshape(rho_ini.shape)
        charge_total = 0.0
        for i in range(ions.nat):
            charge_total += ions.Zval[ions.labels[i]]
        rho_ini *= charge_total / (np.sum(rho_ini) * rho_ini.grid.dV)
    # rho_ini[:] = density.reshape(rho_ini.shape, order='F')
    ############################## add spin magmom ##############################
    nspin=config["DENSITY"]["nspin"]
    if nspin > 1 :
        magmom = config["DENSITY"]["magmom"]
        rho_spin = np.tile(rho_ini, (nspin, 1, 1, 1))
        for ip in range(nspin):
            rho_spin[ip] = 1.0/nspin * rho_spin[ip] + 1.0/nspin *(-1)**ip * magmom/ ions.pos.cell.volume
        rho_ini = DirectField(grid=grid, griddata_3d=rho_spin, rank=nspin)
    #-----------------------------------------------------------------------
    struct = System(ions, grid, name='density', field=rho_ini)
    E_v_Evaluator = TotalEnergyAndPotential(KineticEnergyFunctional=KE, XCFunctional=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
    # The last is a dictionary, which return some properties are used for different situations.
    others = {
        "struct": struct,
        "E_v_Evaluator": E_v_Evaluator,
        "nr2": nr2,
    }
    return config, others


def OptimizeDensityConf(config, struct, E_v_Evaluator, nr2 = None):
    ions = struct.ions
    rho_ini = struct.field
    grid = rho_ini.grid
    charge_total = 0.0
    if hasattr(E_v_Evaluator, 'PSEUDO'):
        PSEUDO=E_v_Evaluator.PSEUDO
    nr = grid.nr
    for i in range(ions.nat):
        charge_total += ions.Zval[ions.labels[i]]
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
            print("#" * 80)
            print("MULTI-STEP: Perform %d optimization step" % istep)
            print("Grid size of %d" % istep, " step is ", nr)
            grid2 = DirectGrid(lattice=grid.lattice, nr=nr, units=None, full=config["GRID"]["gfull"])
            rho_ini = interpolation_3d(rho, nr)
            rho_ini[rho_ini < 1e-12] = 1e-12
            rho_ini = DirectField(grid=grid2, griddata_3d=rho_ini, rank=1)
            rho_ini *= charge_total / (np.sum(rho_ini) * rho_ini.grid.dV)
            # ions.restart()
            if hasattr(E_v_Evaluator, 'PSEUDO'):
                pseudo=E_v_Evaluator.PSEUDO
                if isinstance(pseudo, FunctionalClass):
                    PSEUDO = pseudo.PSEUDO
                else :
                    PSEUDO = pseudo
                PSEUDO.restart(full=False, ions=PSEUDO.ions, grid=grid2)
            opt = Optimization(
                EnergyEvaluator=E_v_Evaluator,
                optimization_options=optimization_options,
                optimization_method=config["OPT"]["method"],
            )
            rho = opt.optimize_rho(guess_rho=rho_ini)
        optimization_options["econv"] /= ions.nat  # reset the value
    else:
        rho = rho_ini
    ############################## calctype  ##############################
    linearii = config["MATH"]["linearii"]
    linearie = config["MATH"]["linearie"]
    energypotential = {}
    forces = {}
    stress = {}
    print("-" * 80)
    if "Both" in config["JOB"]["calctype"]:
        print("Calculate Energy and Potential...")
        energypotential = GetEnergyPotential(
            ions, rho, E_v_Evaluator, calcType=["E","V"], linearii=linearii, linearie=linearie
        )
    elif "Potential" in config["JOB"]["calctype"]:
        print("Calculate Potential...")
        energypotential = GetEnergyPotential(
            ions, rho, E_v_Evaluator, calcType=["V"], linearii=linearii, linearie=linearie
        )
    elif "Density" in config["JOB"]["calctype"]:
        print("Only return density...")
        energypotential = {'TOTAL' : Functional(name = 'TOTAL', energy = 0.0)}
    else:
        print("Calculate Energy...")
        energypotential = GetEnergyPotential(
            ions, rho, E_v_Evaluator, calcType=["E"], linearii=linearii, linearie=linearie
        )
    print(format("Energy information", "-^80"))
    for key in sorted(energypotential.keys()):
        if key == "TOTAL":
            continue
        value = energypotential[key]
        if key == "KEDF":
            ene = 0.0
            for key1 in value:
                print(
                    "{:>10s} energy (eV): {:22.15E}".format(
                        "KEDF-" + key1, value[key1].energy * ENERGY_CONV["Hartree"]["eV"]
                    )
                )
                ene += value[key1].energy
            print("{:>10s} energy (eV): {:22.15E}".format(key, ene * ENERGY_CONV["Hartree"]["eV"]))
        else:
            print("{:>10s} energy (eV): {:22.15E}".format(key, value.energy * ENERGY_CONV["Hartree"]["eV"]))
    print(
        "{:>10s} energy (eV): {:22.15E}".format("TOTAL", energypotential["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"])
    )
    print("-" * 80)

    etot = energypotential["TOTAL"].energy
    etot_eV = etot * ENERGY_CONV["Hartree"]["eV"]
    etot_eV_patom = etot * ENERGY_CONV["Hartree"]["eV"] / ions.nat
    fstr = "  {:<30s} : {:30.15f}"
    print(fstr.format("total energy (a.u.)", etot))
    print(fstr.format("total energy (eV)", etot_eV))
    print(fstr.format("total energy (eV/atom)", etot_eV_patom))
    ############################## Force ##############################
    if "Force" in config["JOB"]["calctype"]:
        print("Calculate Force...")
        forces = GetForces(ions, rho, E_v_Evaluator, linearii=linearii, linearie=linearie, PPlist=None)
        ############################## Output Force ##############################
        f = np.abs(forces["TOTAL"])
        fmax, fmin, fave = np.max(f), np.min(f), np.mean(f)
        fstr_f = " " * 8 + "{0:>22s} : {1:<22.5f}"
        print(fstr_f.format("Max force (a.u.)", fmax))
        print(fstr_f.format("Min force (a.u.)", fmin))
        print(fstr_f.format("Ave force (a.u.)", fave))
        # fstr_f =' ' * 16 + '{0:<22s}{1:<22s}{2:<22s}'
        # print(fstr_f.format('Max (a.u.)', 'Min (a.u.)', 'Ave (a.u.)'))
        # fstr_f =' ' * 16 + '{0:<22.5f} {1:<22.5f} {2:<22.5f}'
        # print(fstr_f.format(fmax, fmin, fave))
        print("-" * 80)
    ############################## Stress ##############################
    if "Stress" in config["JOB"]["calctype"]:
        print("Calculate Stress...")
        stress = GetStress(
            ions,
            rho,
            E_v_Evaluator,
            energypotential=energypotential,
            xc=config["EXC"]["xc"],
            ke=config["KEDF"]["kedf"],
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
                if key == "KEDF":
                    kestress = np.zeros((3, 3))
                    for key1 in value:
                        print("{:>10s} stress (a.u.): ".format("KEDF-" + key1))
                        for i in range(3):
                            print(fstr_s.format(value[key1][i]))
                        kestress += value[key1]
                    print("{:>10s} stress (a.u.): ".format(key))
                    for i in range(3):
                        print(fstr_s.format(kestress[i]))
                else:
                    print("{:>10s} stress (a.u.): ".format(key))
                    for i in range(3):
                        print(fstr_s.format(value[i]))
        print("{:>10s} stress (a.u.): ".format("TOTAL"))
        for i in range(3):
            print(fstr_s.format(stress["TOTAL"][i]))
        print("{:>10s} stress (GPa): ".format("TOTAL"))
        for i in range(3):
            print(fstr_s.format(stress["TOTAL"][i] * STRESS_CONV["Ha/Bohr3"]["GPa"]))
        print("-" * 80)
    ############################## Output Density ##############################
    if config["DENSITY"]["densityoutput"]:
        print("Write Density...")
        outfile = config["DENSITY"]["densityoutput"]
        write(outfile, rho, ions = ions)
    ############################## Output ##############################
    results = {}
    results["density"] = rho
    results["energypotential"] = energypotential
    results["forces"] = forces
    results["stress"] = stress
    if hasattr(E_v_Evaluator, 'PSEUDO'):
        results["pseudo"] = PSEUDO
    # print('-' * 31, 'Time information', '-' * 31)
    # TimeData.reset() #Cleanup the data in TimeData
    # -----------------------------------------------------------------------
    return results


def GetEnergyPotential(ions, rho, EnergyEvaluator, calcType=["E","V"], linearii=True, linearie=True):
    energypotential = {}
    ewaldobj = ewald(rho=rho, ions=ions, PME=linearii)
    energypotential["II"] = Functional(name="Ewald", potential=np.zeros_like(rho), energy=ewaldobj.energy)

    energypotential["TOTAL"] = energypotential["II"].copy()
    funcDict = EnergyEvaluator.funcDict
    for key in funcDict :
        func = getattr(EnergyEvaluator, key)
        if func.type == "KEDF" :
            results = func(rho, calcType, split=True)
            for key2 in results :
                energypotential["TOTAL"] += results[key2]
        else :
            results = func(rho, calcType)
            energypotential["TOTAL"] += results
        energypotential[func.type] = results
    return energypotential


def GetForces(ions, rho, EnergyEvaluator, linearii=True, linearie=True, PPlist=None):
    forces = {}
    forces["PSEUDO"] = EnergyEvaluator.PSEUDO.force(rho)
    ewaldobj = ewald(rho=rho, ions=ions, verbose=False, PME=linearii)
    forces["II"] = ewaldobj.forces
    forces["TOTAL"] = forces["PSEUDO"] + forces["II"]
    return forces


def GetStress(
    ions,
    rho,
    EnergyEvaluator,
    energypotential=None,
    energy=None,
    xc="LDA",
    ke="WT",
    ke_options={"x": 1.0, "y": 1.0},
    linearii=True,
    linearie=True,
    PPlist=None,
):
    """
    Get stress tensor
    """
    #Initial energy dict
    energy = {}
    if energypotential is not None:
        for key in energypotential :
            if key == 'KEDF' :
                energy[key] = {}
                for key2 in energypotential["KEDF"] :
                    energy["KEDF"][key2] = energypotential["KEDF"][key2].energy
            else :
                energy[key] = energypotential[key].energy
    elif energy is None :
        funcDict = EnergyEvaluator.funcDict
        for key in funcDict :
            func = getattr(EnergyEvaluator, key)
            if func.type == "KEDF" :
                energy[func.type] = {"TF": None, "vW": None, "NL": None}
            else :
                energy[func.type] = None

    stress = {}
    ewaldobj = ewald(rho=rho, ions=ions, verbose=False, PME=linearii)
    stress["II"] = ewaldobj.stress
    stress['TOTAL'] = stress['II'].copy()

    funcDict = EnergyEvaluator.funcDict
    for key in funcDict :
        func = getattr(EnergyEvaluator, key)
        key1 = func.type
        if key1 == "XC" :
            if xc == "LDA":
                stress[key1] = LDAStress(rho, energy=energy[key1])
                # stress[key1] = XCStress(rho, name='LDA')
            elif xc == "PBE" :
                stress[key1] = PBEStress(rho, energy=energy[key1])
            else :
                stress[key1] = XCStress(rho, x_str='gga_x_pbe', c_str='gga_c_pbe', energy=energy[key1])
        elif key1 == 'HARTREE' :
            stress[key1] = HartreeFunctionalStress(rho, energy=energy[key1])
        elif key1 == 'PSEUDO' :
            stress[key1] = EnergyEvaluator.PSEUDO.stress(rho, energy=energy[key1])
        elif key1 == 'KEDF' :
            ############################## KE ##############################
            stress["KEDF"] = {}
            KEDF_Stress_L= {
                    "TF" : ["TF"], 
                    "vW": ["TF"], 
                    "x_TF_y_vW": ["TF", "vW"], 
                    "TFvW": ["TF", "vW"], 
                    "WT": ["TF", "vW", "WT"], 
                    }
            if ke not in KEDF_Stress_L :
                raise AttributeError("%s KEDF have not implemented for stress" % ke)
            kelist = KEDF_Stress_L[ke]

            if "TF" in kelist :
                stress["KEDF"]["TF"] = KEDFStress(rho, name="TF", energy=energy["KEDF"]["TF"], **ke_options)
            if "vW" in kelist :
                stress["KEDF"]["vW"] = KEDFStress(rho, name="vW", energy=energy["KEDF"]["vW"], **ke_options)
            if 'NL' in energy["KEDF"] :
                stress["KEDF"]["NL"] = KEDFStress(rho, name=kelist[2], energy=energy["KEDF"]["NL"], **ke_options)
            #-----------------------------------------------------------------------
        else :
            raise AttributeError("%s have not implemented for stress" % key)
        if key1 == 'KEDF' :
            for key2 in stress["KEDF"]:
                stress["TOTAL"] += stress["KEDF"][key2]
        else :
            stress['TOTAL'] += stress[key1]

    for i in range(1, 3):
        for j in range(i - 1, -1, -1):
            stress["TOTAL"][i, j] = stress["TOTAL"][j, i]

    return stress

def InvertRunner(config, struct, EnergyEvaluater):
    file_rho_in = config["INVERSION"]["rho_in"]
    file_v_out = config["INVERSION"]["v_out"]
    rho_in_struct = PP(file_rho_in).read()

    if struct.cell != rho_in_struct.cell:
        raise ValueError('The grid of the input density does not match the grid of the system')

    inv = Inverter()
    ext, rho = inv(rho_in_struct.field, EnergyEvaluater)
    xsf = XSF(file_v_out)
    xsf.write(struct, field=ext.v)
    #xsf = XSF('./rho.xsf')
    #xsf.write(struct, field=rho)

    return ext
