import numpy as np
import copy
import configparser
from dftpy.mpi import sprint
from dftpy.constants import ENERGY_CONV, LEN_CONV
from dftpy.config.config_entry import ConfigEntry


def config_map(mapping_function, premap_conf):
    return dict(zip(premap_conf, map(mapping_function, premap_conf.values())))


def readJSON(JSON_file):

    import json
    with open(JSON_file) as f:
        conf_JSON = json.load(f)

    def map_JSON_ConfigEntry(value):
        if 'type' in value and 'default' in value :
            return ConfigEntry(**value)
        else :
            return config_map(map_JSON_ConfigEntry, value)

    conf = config_map(map_JSON_ConfigEntry, conf_JSON)
    return conf


def DefaultOptionFromEntries(conf):

    def map_ConfigEntry_default(config_entry):
        if isinstance(config_entry, ConfigEntry):
            return config_entry.default
        else :
            return config_map(map_ConfigEntry_default, config_entry)

    results = config_map(map_ConfigEntry_default, conf)
    results['CONFDICT'] = copy.deepcopy(conf)
    return results
    # return config_map(map_ConfigEntry_default, conf)


def default_json():
    import os
    fileJSON = os.path.join(os.path.dirname(__file__), 'configentries.json')
    configentries = readJSON(fileJSON)
    return configentries

def DefaultOption():
    import os
    fileJSON = os.path.join(os.path.dirname(__file__), 'configentries.json')
    configentries = readJSON(fileJSON)
    return DefaultOptionFromEntries(configentries)


def ConfSpecialFormat(conf):
    ############################## Conversion of units  ##############################
    """
    Ecut = pi^2/(2 * h^2)
    Ref : Briggs, E. L., D. J. Sullivan, and J. Bernholc. Physical Review B 54.20 (1996): 14362.
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

    # for key in conf["PP"]:
        # conf["PP"][key.capitalize()] = conf["PP"][key]

    if conf["MATH"]["twostep"] and conf["MATH"]["multistep"] == 1 :
        conf["MATH"]["multistep"] = 2

    if 'CONFDICT' in conf :
        del conf['CONFDICT']

    return conf


def PrintConf(conf, comm = None):
    if not isinstance(conf, dict):
        raise TypeError("conf must be dict")
    try:
        import json
        pretty_dict_str = json.dumps(conf, indent=4, sort_keys=True)
    except Exception:
        import pprint
        # pprint.pprint(conf)
        pretty_dict_str = pprint.pformat(conf)
    sprint(pretty_dict_str, comm = comm)
    return pretty_dict_str


def ReadConfbak(infile):
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
                sprint('!WARN : "%s.%s" not in the dictionary' % (section, key))
            elif section == 'PP':
                conf['PP'][key.capitalize()] = pp_entry.format(config.get(section, key))
            else:
                conf[section][key] = configentries[section][key].format(config.get(section, key))
    conf = ConfSpecialFormat(conf)
    return conf


def ReadConf(infile):
    config = configparser.ConfigParser()
    config.read(infile)

    conf = DefaultOption()
    for section in config.sections():
        for key in config.options(section):
            if section != 'PP' and key not in conf[section]:
                sprint('!WARN : "%s.%s" not in the dictionary' % (section, key))
            elif section == 'PP':
                conf['PP'][key.capitalize()] = config.get(section, key)
            else:
                conf[section][key] = config.get(section, key)
    conf = OptionFormat(conf)
    return conf


def OptionFormat(config):
    conf = {}
    for section in config :
        if section == 'CONFDICT':
            continue
        else :
            conf[section] = {}
        for key in config[section] :
            if section == 'PP':
                conf["PP"][key.capitalize()] = config["PP"][key]
            elif config[section][key] :
                conf[section][key] = config['CONFDICT'][section][key].format(str(config[section][key]))
            else :
                conf[section][key] = config[section][key]

    conf = ConfSpecialFormat(conf)
    return conf
