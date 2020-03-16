import numpy as np
import copy
import configparser
from dftpy.constants import ENERGY_CONV, LEN_CONV


def DefaultOption():
    JOB = {"task": "Optdensity", "calctype": "Energy"}

    PATH = {
        "pppath": "./",
        "cellpath": "./",
    }

    MATH = {
        "linearii": True,
        "linearie": True,
        "twostep": False,
        "multistep": 1,
        "reuse": True,
    }

    PP = {}

    CELL = {
        "cellfile": "POSCAR",
        "elename": "Al",
        "zval": None,
        "format": None,
    }

    GRID = {
        "ecut": 600,
        "spacing": None,
        "gfull": False,
        "nr": None,
    }

    DENSITY = {
        "nspin"  : 1,
        "magmom" : 0.0,
        "densityini": "HEG",
        "densityfile": None,
        "densityoutput": None,
    }

    EXC = {
        "xc": "LDA",
        "x_str": "lda_x",
        "c_str": "lda_c_pz",
        "polarization": "unpolarized",
    }

    KEDF = {
        "kedf": "WT",
        "x": 1.0,
        "y": 1.0,
        "alpha": 5.0 / 6.0,
        "beta": 5.0 / 6.0,
        # "sigma": 0.025,
        "sigma": None,
        "nsp": None,  # The number of spline
        "interp": "hermite",
        "kerneltype": "WT",
        "symmetrization": None,
        "lumpfactor": None,  # factor for MGP
        "neta": 50000,
        "etamax": 50.0,
        "order": 3,
        "ratio": 1.2,
        "maxpoints": 1000,
        "delta": None,  # The gap of spline
        "fd": 0,
        "kdd": 3,  # kernel density denpendent
        "rho0": None,
        "k_str": "gga_k_revapbe",
        "kfmin": None,
        "kfmax": None,
    }

    OUTPUT = {
        "time": True,
        "stress": True,
    }

    OPT = {
        "method": "CG-HS",
        "algorithm": "EMM",  # Residual minimization method or Energy minimization method
        "vector": "Orthogonalization",  # or Scaling
        "c1": 1e-4,
        "c2": 2e-1,
        "maxls": 10,
        "econv": 1e-6,  # Energy Convergence (a.u./atom)
        "maxfun": 50,  # For TN method, it's the max steps for searching direction
        "maxiter": 100,  # The max steps for optimization
        "xtol": 1e-12,
        "h0": 1.0,  # for LBFGS
    }

    PROPAGATOR = {
        "type": "crank-nicolson",
        "int_t": 1e-3,
        "order": 20, 
        "linearsolver": "bicgstab", 
        "tol": 1e-10, 
        "maxiter": 100, 
    }

    TD = {
        "outfile": "td_out",
        "tmax": 1.0,
        "order": 1,
        "direc": 0,
        "strength": 1.0e-3,
    }

    conf = {
        "JOB": JOB,
        "PATH": PATH,
        "MATH": MATH,
        "PP": PP,
        "KEDF": KEDF,
        "CELL": CELL,
        "GRID": GRID,
        "EXC": EXC,
        "KEDF": KEDF,
        "OPT": OPT,
        "DENSITY": DENSITY,
        "OUTPUT": OUTPUT,
        "PROPAGATOR": PROPAGATOR,
        "TD": TD,
    }

    for section in conf:
        for key in conf[section]:
            conf[section][key] = str(conf[section][key])
    return conf


def OptionFormat(config):
    if not isinstance(config, dict):
        raise TypeError("config must be dict")
    conf = copy.deepcopy(config)
    for section in conf:
        for key in conf[section]:
            if conf[section][key] == "None":
                conf[section][key] = None

    def bools(strings):
        s = strings.lower()[0]
        if s == "n" or s == "f" or s == "0":
            return False
        else:
            return True

    conf["JOB"]["task"] = conf["JOB"]["task"].capitalize()
    conf["JOB"]["calctype"] = [s.capitalize() for s in conf["JOB"]["calctype"].split()]

    conf["MATH"]["linearii"] = bools(conf["MATH"]["linearii"])
    conf["MATH"]["linearie"] = bools(conf["MATH"]["linearie"])
    conf["MATH"]["twostep"] = bools(conf["MATH"]["twostep"])
    conf["MATH"]["multistep"] = int(conf["MATH"]["multistep"])
    if conf["MATH"]["twostep"]:
        conf["MATH"]["multistep"] = 2
    conf["MATH"]["reuse"] = bools(conf["MATH"]["reuse"])

    conf["KEDF"]["x"] = float(eval(conf["KEDF"]["x"]))
    conf["KEDF"]["y"] = float(eval(conf["KEDF"]["y"]))
    conf["KEDF"]["alpha"] = float(eval(conf["KEDF"]["alpha"]))
    conf["KEDF"]["beta"] = float(eval(conf["KEDF"]["beta"]))
    if conf["KEDF"]["sigma"] is not None :
        conf["KEDF"]["sigma"] = float(eval(conf["KEDF"]["sigma"]))
    conf["KEDF"]["etamax"] = float(eval(conf["KEDF"]["etamax"]))
    conf["KEDF"]["neta"] = int(conf["KEDF"]["neta"])
    conf["KEDF"]["order"] = int(conf["KEDF"]["order"])
    conf["KEDF"]["maxpoints"] = int(conf["KEDF"]["maxpoints"])
    conf["KEDF"]["ratio"] = float(eval(conf["KEDF"]["ratio"]))
    conf["KEDF"]["fd"] = int(conf["KEDF"]["fd"])
    conf["KEDF"]["kdd"] = int(conf["KEDF"]["kdd"])
    if conf["KEDF"]["kfmin"]:
        conf["KEDF"]["kfmin"] = float(eval(conf["KEDF"]["kfmin"]))
    if conf["KEDF"]["kfmax"]:
        conf["KEDF"]["kfmax"] = float(eval(conf["KEDF"]["kfmax"]))
    if conf["KEDF"]["rho0"]:
        conf["KEDF"]["rho0"] = float(eval(conf["KEDF"]["rho0"]))
    if conf["KEDF"]["nsp"]:
        conf["KEDF"]["nsp"] = int(conf["KEDF"]["nsp"])
    if conf["KEDF"]["delta"]:
        conf["KEDF"]["delta"] = float(eval(conf["KEDF"]["delta"]))
    if conf["KEDF"]["lumpfactor"]:
        l = conf["KEDF"]["lumpfactor"].split()
        if len(l) > 1:
            lump = [float(eval(item)) for item in l]
        else:
            lump = float(eval(l[0]))
        conf["KEDF"]["lumpfactor"] = lump

    conf["GRID"]["gfull"] = bools(conf["GRID"]["gfull"])
    # conf['GRID']['spacing']        = float(eval(conf['GRID']['spacing']))
    if conf["GRID"]["nr"]:
        conf["GRID"]["nr"] = list(map(int, conf["GRID"]["nr"].split()))

    if conf["CELL"]["elename"]:
        conf["CELL"]["elename"] = [s.capitalize() for s in conf["CELL"]["elename"].split()]
    if conf["CELL"]["zval"]:
        conf["CELL"]["zval"] = list(map(float, conf["CELL"]["zval"].split()))

    conf["OPT"]["c1"] = float(conf["OPT"]["c1"])
    conf["OPT"]["c2"] = float(conf["OPT"]["c2"])
    conf["OPT"]["econv"] = float(eval(conf["OPT"]["econv"]))
    conf["OPT"]["maxls"] = int(conf["OPT"]["maxls"])
    conf["OPT"]["maxfun"] = int(conf["OPT"]["maxfun"])
    conf["OPT"]["maxiter"] = int(conf["OPT"]["maxiter"])
    conf["OPT"]["xtol"] = float(eval(conf["OPT"]["xtol"]))
    conf["OPT"]["h0"] = float(eval(conf["OPT"]["h0"]))

    conf["DENSITY"]["nspin"] = int(conf["DENSITY"]["nspin"])
    conf["DENSITY"]["magmom"] = float(eval(conf["DENSITY"]["magmom"]))

    conf["OUTPUT"]["time"] = bools(conf["OUTPUT"]["time"])
    conf["OUTPUT"]["stress"] = bools(conf["OUTPUT"]["stress"])

    conf['PROPAGATOR']['int_t'] = float(conf['PROPAGATOR']['int_t'])
    conf['PROPAGATOR']['order'] = int(conf['PROPAGATOR']['order'])
    conf['PROPAGATOR']['tol'] = float(conf['PROPAGATOR']['tol'])
    conf['PROPAGATOR']['maxiter'] = int(conf['PROPAGATOR']['maxiter'])

    conf['TD']['tmax'] = float(conf['TD']['tmax'])
    conf['TD']['order'] = int(conf['TD']['order'])
    conf['TD']['strength'] = float(conf['TD']['strength'])
    if conf['TD']['direc']:
        if conf['TD']['direc'] == 'x':
            conf['TD']['direc'] = 0
        elif conf['TD']['direc'] == 'y':
            conf['TD']['direc'] = 1
        elif conf['TD']['direc'] == 'z':
            conf['TD']['direc'] = 2
        else:
            conf['TD']['direc'] = int(conf['TD']['direc'])

    ############################## Conversion of units  ##############################
    """
    Ecut = pi^2/(2 * h^2)
    Ref : Briggs, E. L., D. J. Sullivan, and J. Bernholc. "Real-space multigrid-based approach to large-scale electronic structure calculations." Physical Review B 54.20 (1996): 14362.
    """
    if conf["GRID"]["spacing"]:  # Here units are : spacing (Angstrom),  ecut (eV), same as input.
        conf["GRID"]["spacing"] = float(eval(conf["GRID"]["spacing"]))
        conf["GRID"]["ecut"] = (
            np.pi ** 2
            / (2 * conf["GRID"]["spacing"] ** 2)
            * ENERGY_CONV["Hartree"]["eV"]
            / LEN_CONV["Angstrom"]["Bohr"] ** 2
        )
    else:
        conf["GRID"]["ecut"] = float(eval(conf["GRID"]["ecut"]))
        conf["GRID"]["spacing"] = (
            np.sqrt(np.pi ** 2 / conf["GRID"]["ecut"] * 0.5 / ENERGY_CONV["eV"]["Hartree"])
            * LEN_CONV["Bohr"]["Angstrom"]
        )

    for key in conf["PP"]:
        conf["PP"][key.capitalize()] = conf["PP"][key]

    return conf


def PrintConf(conf):
    if not isinstance(conf, dict):
        raise TypeError("conf must be dict")
    try:
        import json
        print(json.dumps(conf, indent=4, sort_keys=True))
    except Exception:
        import pprint

        pprint.pprint(conf)
        pretty_dict_str = pprint.pformat(conf)
        return pretty_dict_str


def ReadConf(infile):
    config = configparser.ConfigParser()
    config.read(infile)

    conf = DefaultOption()
    for section in config.sections():
        for key in config.options(section):
            if section != 'PP' and key not in conf[section]:
                print('!WARN : "%s.%s" not in the dictionary' % (section, key))
            elif section == 'PP':
                conf['PP'][key.capitalize()] = config.get(section, key)
            else:
                conf[section][key] = config.get(section, key)
    return conf
