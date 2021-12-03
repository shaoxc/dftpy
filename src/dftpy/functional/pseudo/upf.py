import importlib.util
import re
import numpy as np

from dftpy.functional.pseudo.abstract_pseudo import BasePseudo

"""
Ref :
    http://www.quantum-espresso.org/pseudopotentials/unified-pseudopotential-format
    https://esl.cecam.org/data/upf/
"""

class UPFDICT(BasePseudo):
    def __init__(self, fname, direct = True, **kwargs):
        super().__init__(fname, direct = direct, **kwargs)

    def read(self, fname):
        """Reads QE UPF type PP"""
        self.pattern = re.compile('\s+')
        found = importlib.util.find_spec("xmltodict")
        if found:
            import xmltodict
        else:
            raise ModuleNotFoundError("Please pip install xmltodict")
        with open(fname) as fd:
            doc = xmltodict.parse(fd.read(), attr_prefix='')
            info = doc[next(iter(doc.keys()))]
        r = self.get_array(info['PP_MESH']['PP_R'])
        v = self.get_array(info['PP_LOCAL']) * 0.5  # Ry to a.u.
        self.r = r
        self.v = v
        self.info = info
        self._zval = float(self.info["PP_HEADER"]["z_valence"])
        if 'PP_NLCC' in self.info:
            self._core_density = self.get_array(self.info['PP_NLCC'])
        if 'PP_RHOATOM' in self.info:
            rho = self.get_array(self.info['PP_RHOATOM'])
            if self.r[0] > 1E-10 :
                rho[:] /= (4*np.pi*self.r[:]**2)
            else :
                rho[1:] /= (4*np.pi*self.r[1:]**2)
            self._atomic_density = rho

    def get_array(self, attr):
        if isinstance(attr, dict): attr = attr['#text']
        if attr is None or len(attr) == 0: return None
        value = self.pattern.split(attr)
        return np.array(value, dtype=np.float64)


class UPFJSON(BasePseudo):
    def __init__(self, fname, direct = True, **kwargs):
        super().__init__(fname, direct = direct, **kwargs)

    def read(self, fname):
        """Reads QE UPF type PP"""
        import importlib.util

        upf2json = importlib.util.find_spec("upf_to_json")
        found = upf2json is not None
        if found:
            from upf_to_json import upf_to_json
        else:
            raise ModuleNotFoundError("Please pip install upf_to_json")
        with open(fname, "r") as outfil:
            info = upf_to_json(upf_str=outfil.read(), fname=fname)["pseudo_potential"]
        r = np.array(info["radial_grid"], dtype=np.float64)
        v = np.array(info["local_potential"], dtype=np.float64)
        self.r = r
        self.v = v
        self.info = info
        self._zval = self.info["header"]["z_valence"]
        if 'core_charge_density' in self.info:
            self._core_density = np.array(self.info["core_charge_density"], dtype=np.float64)
        if 'total_charge_density' in self.info:
            rho = np.array(self.info["total_charge_density"], dtype=np.float64)
            if self.r[0] > 1E-10 :
                rho[:] /= (4*np.pi*self.r[:]**2)
            else :
                rho[1:] /= (4*np.pi*self.r[1:]**2)
            self._atomic_density = rho

class UPF:
    def __new__(cls, fname):
        """
        Note :
            Prefer xmltodict which is more robust
            xmltodict not work for UPF v1
        """
        has_xml = importlib.util.find_spec("xmltodict")
        has_json = importlib.util.find_spec("upf_to_json")
        if has_xml:
            try:
                obj = UPFDICT(fname)
            except:
                if has_json:
                    try:
                        obj = UPFJSON(fname)
                    except:
                        raise ModuleNotFoundError("Please use standard 'UPF' file")
                else :
                    raise ModuleNotFoundError("Maybe you can try install upf_to_json or use standard 'UPF' file")
        elif has_json:
            try:
                obj = UPFJSON(fname)
            except:
                raise ModuleNotFoundError("Maybe you can try install xmltodict or use standard 'UPF' file")
        else:
            raise ModuleNotFoundError("Please pip install xmltodict or upf_to_json")
        return obj
