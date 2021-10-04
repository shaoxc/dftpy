import numpy as np
import os
from dftpy.mpi import sprint
from dftpy.optimization import Optimization
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.constants import LEN_CONV, ENERGY_CONV, STRESS_CONV
from dftpy.formats.io import read, read_density, write, read_system
from dftpy.ewald import ewald
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import bestFFTsize, interpolation_3d
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.semilocal_xc import XCStress
from dftpy.functional.kedf import KEDFStress
from dftpy.functional.hartree import HartreeFunctionalStress
from dftpy.config.config import PrintConf, ReadConf
from dftpy.system import System
from dftpy.inverter import Inverter
from dftpy.formats.xsf import XSF
from dftpy.formats.qepp import PP
from dftpy.properties import get_electrostatic_potential


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
        struct = read_system(
            config["PATH"]["cellpath"] +os.sep+ config["CELL"]["cellfile"],
            format=config["CELL"]["format"],
            names=config["CELL"]["elename"],
        )
        ions = struct.ions
    else :
        struct = None
    lattice = ions.pos.cell.lattice
    metric = np.dot(lattice.T, lattice)
    if config["GRID"]["nr"] is not None:
        nr = np.asarray(config["GRID"]["nr"])
    elif struct is not None and struct.field is not None :
        nr = struct.field.grid.nr
        rhoini = struct.field
    else:
        # spacing = config['GRID']['spacing']
        spacing = config["GRID"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
        nr = np.zeros(3, dtype="int32")
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i]) / spacing)
        sprint("The initial grid size is ", nr)
        nproc = 1
        # nr0 = nr.copy()
        for i in range(3):
            # if i == 0 :
                # if mp is not None : nproc = mp.comm.size
            # else :
                # nr[i] *= nr[0]/nr0[0]
            nr[i] = bestFFTsize(nr[i], nproc=nproc, **config["GRID"])
    sprint("The final grid size is ", nr)
    nr2 = nr.copy()
    if config["MATH"]["multistep"] > 1:
        nr = nr2 // config["MATH"]["multistep"]
        sprint("MULTI-STEP: Perform first optimization step")
        sprint("Grid size of 1  step is ", nr)

    ############################## Grid  ##############################
    if grid is None:
        grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=config["GRID"]["gfull"], cplx=config["GRID"]["cplx"], mp=mp)
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
        PSEUDO = Functional(type ='PSEUDO', grid=grid, ions=ions, PP_list=PPlist, PME=linearie)
    else:
        PSEUDO = pseudo

        PSEUDO.restart(full=False)
        PSEUDO.grid = grid
        PSEUDO.ions = ions
    kedf_config = config["KEDF"].copy()
    if kedf_config.get('temperature', None):
        kedf_config['temperature'] *= ENERGY_CONV['eV']['Hartree']
    print(kedf_config)
    KE = Functional(type="KEDF", name=config["KEDF"]["kedf"], **kedf_config)
    ############################## XC and Hartree ##############################
    HARTREE = Functional(type="HARTREE")
    XC = Functional(type="XC", name=config["EXC"]["xc"], **config["EXC"])
    ############################## Initial density ##############################
    zerosA = np.empty(grid.nnr, dtype=float)
    rho_ini = DirectField(grid=grid, griddata_C=zerosA, rank=1)
    density = None
    if rhoini is not None:
        density = rhoini
    elif config["DENSITY"]["densityini"] == "HEG":
        charge_total = ions.ncharge
        rho_ini[:] = charge_total / ions.pos.cell.volume
        # -----------------------------------------------------------------------
        # rho_ini[:] = charge_total/ions.pos.cell.volume + np.random.random(np.shape(rho_ini)) * 1E-2
        # rho_ini *= (charge_total / (np.sum(rho_ini) * rho_ini.grid.dV ))
        # -----------------------------------------------------------------------
    elif config["DENSITY"]["densityini"] == "Read":
        density = read_density(config["DENSITY"]["densityfile"])
    # normalization
    if density is not None:
        if not np.all(grid.nrR == density.shape[:3]):
            density = interpolation_3d(density, grid.nrR)
            # density = prolongation(density)
            density[density < 1e-12] = 1e-12
        grid.scatter(density, out = rho_ini)
        charge_total = ions.ncharge
        rho_ini *= charge_total / rho_ini.integral()
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
    E_v_Evaluator = TotalFunctional(KineticEnergyFunctional=KE, XCFunctional=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
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
    if hasattr(E_v_Evaluator, 'PSEUDO'):
        PSEUDO=E_v_Evaluator.PSEUDO
    nr = grid.nr
    charge_total = ions.ncharge
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
            grid2 = DirectGrid(lattice=grid.lattice, nr=nr, units=None, full=config["GRID"]["gfull"], mp=grid.mp)
            rho_ini = interpolation_3d(rho, nr)
            rho_ini[rho_ini < 1e-12] = 1e-12
            rho_ini = DirectField(grid=grid2, griddata_3d=rho_ini, rank=1)
            rho_ini *= charge_total / (np.sum(rho_ini) * rho_ini.grid.dV)
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
    else:
        rho = rho_ini
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
    for key in sorted(keys):
        if key == "TOTAL":
            continue
        value = ep_w[key]
        sprint("{:>10s} energy (eV): {:22.15E}".format(key, ep_w[key]* ENERGY_CONV["Hartree"]["eV"]))
    etot = ep_w['TOTAL']
    sprint("{:>10s} energy (eV): {:22.15E}".format("TOTAL", etot * ENERGY_CONV["Hartree"]["eV"]))
    sprint("-" * 80)

    etot_eV = etot * ENERGY_CONV["Hartree"]["eV"]
    etot_eV_patom = etot * ENERGY_CONV["Hartree"]["eV"] / ions.nat
    fstr = "  {:<30s} : {:30.15f}"
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
        write(outfile, rho, ions = ions)
    ############################## Output ##############################
    if config["OUTPUT"]["electrostatic_potential"]:
        sprint("Write electrostatic potential...")
        outfile = config["OUTPUT"]["electrostatic_potential"]
        v = get_electrostatic_potential(rho, E_v_Evaluator)
        write(outfile, v, ions = ions)
    results = {}
    results["density"] = rho
    results["energypotential"] = energypotential
    results["forces"] = forces
    results["stress"] = stress
    if hasattr(E_v_Evaluator, 'PSEUDO'):
        results["pseudo"] = PSEUDO
    # sprint('-' * 31, 'Time information', '-' * 31)
    # TimeData.reset() #Cleanup the data in TimeData
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
                energypotential["KEDF-" + key2] = results[key2]
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


def GetStress(
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
            "vW": ["TF"],
            "x_TF_y_vW": ["TF", "vW"],
            "TFvW": ["TF", "vW"],
            "WT": ["TF", "vW", "NL"],
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
            if "vW" in key1 :
                stress[key1] = KEDFStress(rho, name="vW", energy=energy[key1], **ke_options)
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
