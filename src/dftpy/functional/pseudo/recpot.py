import numpy as np

from dftpy.constants import LEN_CONV, ENERGY_CONV


class RECPOT:
    def __init__(self, fname):
        self.fname = fname
        self.read(fname)

    def read(self, fname):
        """Reads CASTEP-like recpot PP file
        Returns tuple (g, v)"""
        # HARTREE2EV = 27.2113845
        # BOHR2ANG   = 0.529177211
        HARTREE2EV = ENERGY_CONV["Hartree"]["eV"]
        BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
        with open(fname, "r") as outfil:
            lines = outfil.readlines()

        ibegin = 0
        for i in range(0, len(lines)):
            line = lines[i]
            if "END COMMENT" in line:
                ibegin = i + 3
            elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 1):
                iend = i
                break

        line = " ".join([line.strip() for line in lines[ibegin:iend]])

        if "1000" in lines[iend] or len(lines[iend].strip()) == 1:
            pass
        else:
            raise AttributeError("Error : Check the PP file : {}".format(fname))
        gmax = np.float(lines[ibegin - 1].strip()) * BOHR2ANG
        self.v_g = np.array(line.split()).astype(np.float) / HARTREE2EV / BOHR2ANG ** 3
        self.r_g = np.linspace(0, gmax, num=len(self.v_g))

    @property
    def zval(self):
        gp = self.r_g
        vp = self.v_g
        val = (vp[0] - vp[1]) * (gp[1] ** 2) / (4.0 * np.pi)
        # val = (vp[0] - vp[1]) * (gp[-1] / (gp.size - 1)) ** 2 / (4.0 * np.pi)
        return round(val)
