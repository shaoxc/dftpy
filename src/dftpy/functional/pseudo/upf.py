import importlib.util
import re

import numpy as np


class UPF(object):
    def __init__(self, fname):
        self.fname = fname
        self.pattern = re.compile('\s+')
        self.read(fname)

    def read(self, fname):
        """Reads QE UPF type PP"""
        found = importlib.util.find_spec("xmltodict")
        if found:
            import xmltodict
        else:
            raise ModuleNotFoundError("Must pip install xmltodict")
        with open(fname) as fd:
            doc = xmltodict.parse(fd.read(), attr_prefix='')
            info = doc[next(iter(doc.keys()))]
        r = self.get_array(info['PP_MESH']['PP_R'])
        v = self.get_array(info['PP_LOCAL']) * 0.5  # Ry to a.u.
        self.r = r
        self.v = v
        self.info = info

    def get_array(self, attr):
        if isinstance(attr, dict): attr = attr['#text']
        if attr is None or len(attr) == 0: return None
        value = self.pattern.split(attr)
        return np.array(value, dtype=np.float64)

    @property
    def zval(self):
        return float(self.info["PP_HEADER"]["z_valence"])

    @property
    def core_density(self):
        if 'PP_NLCC' in self.info:
            return self.get_array(self.info['PP_NLCC'])
        else:
            return None


class UPFJSON(object):
    def __init__(self, fname):
        self.fname = fname
        self.read(fname)

    def read(self, fname):
        """Reads QE UPF type PP"""
        import importlib.util

        upf2json = importlib.util.find_spec("upf_to_json")
        found = upf2json is not None
        if found:
            from upf_to_json import upf_to_json
        else:
            raise ModuleNotFoundError("Must pip install upf_to_json")
        with open(fname, "r") as outfil:
            info = upf_to_json(upf_str=outfil.read(), fname=fname)["pseudo_potential"]
        r = np.array(info["radial_grid"], dtype=np.float64)
        v = np.array(info["local_potential"], dtype=np.float64)
        self.r = r
        self.v = v
        self.info = info

    @property
    def zval(self):
        return self.info["header"]["z_valence"]

    @property
    def core_density(self):
        if 'core_charge_density' in self.info:
            return np.array(self.info["core_charge_density"], dtype=np.float64)
        else:
            return None
