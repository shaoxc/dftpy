import numpy as np
import copy
import configparser
from dftpy.constants import ENERGY_CONV, LEN_CONV
from dftpy.config.config_entry import ConfigEntry


def config_map(mapping_function, premap_conf):

    def section_map(sectiondict):
        return dict(zip(sectiondict, map(mapping_function, sectiondict.values())))

    return dict(zip(premap_conf, map(section_map, premap_conf.values())))


def readJSON(JSON_file):

    import json
    with open(JSON_file) as f:
        conf_JSON = json.load(f)

    def map_JSON_ConfigEntry(value):
        return ConfigEntry(**value)

    return config_map(map_JSON_ConfigEntry, conf_JSON)


def DefaultOptionFromEntries(conf):

    def map_ConfigEntry_default(config_entry):
        return config_entry.default

    return config_map(map_ConfigEntry_default, conf)


def DefaultOption():
    import os
    fileJSON = os.path.join(os.path.dirname(__file__), 'configentries.json')
    configentries = readJSON(fileJSON)
    return DefaultOptionFromEntries(configentries)


def ConfSpecialFormat(conf):
    ############################## Conversion of units  ##############################
    """
    Ecut = pi^2/(2 * h^2)
    Ref : Briggs, E. L., D. J. Sullivan, and J. Bernholc. "Real-space multigrid-based approach to large-scale electronic structure calculations." Physical Review B 54.20 (1996): 14362.
    """
    if conf["GRID"]["spacing"]:  # Here units are : spacing (Angstrom),  ecut (eV), same as input.
        conf["GRID"]["ecut"] = (
            np.pi ** 2
            / (2 * conf["GRID"]["spacing"] ** 2)
            * ENERGY_CONV["Hartree"]["eV"]
            / LEN_CONV["Angstrom"]["Bohr"] ** 2
        )
    else:
        conf["GRID"]["spacing"] = (
            np.sqrt(np.pi ** 2 / conf["GRID"]["ecut"] * 0.5 / ENERGY_CONV["eV"]["Hartree"])
            * LEN_CONV["Bohr"]["Angstrom"]
        )

    if conf["KEDF"]["lumpfactor"]:
        if len(conf["KEDF"]["lumpfactor"]) == 1:
            conf["KEDF"]["lumpfactor"] = conf["KEDF"]["lumpfactor"][0]

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

    import os
    fileJSON = os.path.join(os.path.dirname(__file__), 'configentries.json')
    configentries = readJSON(fileJSON)
    pp_entry = ConfigEntry(type='str')
    conf = DefaultOptionFromEntries(configentries)
    for section in config.sections():
        for key in config.options(section):
            if section != 'PP' and key not in conf[section]:
                print('!WARN : "%s.%s" not in the dictionary' % (section, key))
            elif section == 'PP':
                conf['PP'][key.capitalize()] = pp_entry.format(config.get(section, key))
            else:
                conf[section][key] = configentries[section][key].format(config.get(section, key))
    conf = ConfSpecialFormat(conf)
    return conf
