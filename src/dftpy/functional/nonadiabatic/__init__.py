from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.field import DirectField
from dftpy.functional.nonadiabatic.wcdhc import WCDHC
from dftpy.functional.nonadiabatic.jp import JP1
from typing import List, Optional

__all__ = ["Dynamic"]

Dynamic_Dict = {
    "WCDHC": WCDHC,
    "JP1": JP1,
}


class Dynamic(AbstractFunctional):
    def __init__(self, name: str, **kwargs) -> None:
        if name not in Dynamic_Dict:
            raise AttributeError("{0:s} is not a correct dynamic functional name.".format(name))
        self.name = name

    def compute(self, density: DirectField, current: DirectField = None, der_current: Optional[DirectField] = None,
                calcType: List[str] = ["V"], **kwargs) -> FunctionalOutput:
        return Dynamic_Dict[self.name](density, current, der_current=der_current, calcType=calcType, **kwargs)
