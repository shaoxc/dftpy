from dftpy.functional.kedf import KEDF
from dftpy.functional.pseudo import LocalPseudo
from dftpy.functional.hartree import Hartree
from dftpy.functional.semilocal_xc import XC
from dftpy.functional.external_potential import ExternalPotential
from dftpy.functional.nonadiabatic import Dynamic
from dftpy.functional.total_functional import TotalFunctional
from dftpy.functional.functional_output import FunctionalOutput

FunctionalTypeDict = {
    'KEDF': KEDF,
    'PSEUDO': LocalPseudo,
    'HARTREE': Hartree,
    'XC': XC,
    'EXT': ExternalPotential,
    'DYNAMIC': Dynamic,
}


def Functional(type=None, optional_kwargs=None, **kwargs):
    if optional_kwargs is not None:
        kwargs.update(optional_kwargs)

    try:
        return FunctionalTypeDict[type](**kwargs)
    except KeyError:
        raise AttributeError("Unknown functional type: {type}.")
