from typing import Union

from dftpy.td.propagator.crank_nicholson import CrankNicholson
from dftpy.td.propagator.taylor import Taylor


class Propagator(object):
    NameDict = {
        'taylor': Taylor,
        'crank-nicholson': CrankNicholson,
    }

    def __new__(cls, *args, name: Union[str, None] = None, **kwargs):
        try:
            return cls.NameDict[name](*args, **kwargs)
        except KeyError:
            raise AttributeError("{0:s} is not a supported propagator type".format(name))
