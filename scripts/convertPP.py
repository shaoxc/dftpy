#!/usr/bin/env python3

from dftpy.constants import LEN_CONV, ENERGY_CONV
from dftpy.functional.pseudo import ReadPseudo
import argparse


class PseudoPotential(ReadPseudo):
    def __init__(self, infile, MaxPoints = 15000, Gmax = 30, Rmax = 10, **kwargs):
        key = 'DFTPY'
        PP_list = {key:infile}
        super().__init__(PP_list, MaxPoints = MaxPoints, Gmax = Gmax, **kwargs)
        self.key = 'DFTPY'

    def out_recpot(self, outf='dftpy_convert'+'.recpot', header='DFTPY CONVERT', **kwargs):
        HARTREE2EV = ENERGY_CONV["Hartree"]["eV"]
        BOHR2ANG = LEN_CONV["Bohr"]["Angstrom"]
        gmax = self._gp[self.key][-1] / BOHR2ANG
        v = self._vp[self.key] * HARTREE2EV * BOHR2ANG ** 3
        fw = open(outf, 'w')
        fw.write('START COMMENT\n' + header + '\nEND COMMENT\n')
        fw.write('3     5\n') # Now only support 5
        fw.write(str(gmax) + '\n')
        nl = v.size // 3
        # seq = ' ' * 5
        for line in v[: nl * 3].reshape(-1, 3):
            fw.write("     {0[0]:22.15E}     {0[1]:22.15E}     {0[2]:22.15E}\n".format(line))
        for line in v[nl * 3 :]:
            fw.write("{0:22.15E}".format(line))
        if len(v[nl * 3 :]) > 0 :
            fw.write("\n")
        fw.write("1000\n")
        fw.close()

    def output(self, outf='dftpy_convert'+'.recpot', header='DFTPY CONVERT', **kwargs):
        if outf[-6:].lower() == "recpot":
            self.out_recpot(outf, header=header, **kwargs)
        elif outf[-3:].lower() == "upf":
            raise Exception("Pseudopotential not supported")
        elif outf[-3:].lower() == "psp" or outf[-4:].lower() == "psp8":
            raise Exception("Pseudopotential not supported")
        else :
            raise Exception("Pseudopotential not supported")


def RunJob():
    parser = argparse.ArgumentParser(description='Convert Pseudopotential')
    parser.add_argument('-i', '--ini', '--input', dest='inputPP', type=str, action='store',
            default='config.ini', help='The input Pseudopotential')
    parser.add_argument('-o', '--out', '--output', dest='outPP', type=str, action='store',
            default='dftpy_convert.recpot', help='The output Pseudopotential')
    parser.add_argument('--mp', '--maxpoints', dest='MaxPoints', type=int, action='store',
            default=200000, help='The number of points')
    parser.add_argument('--gmax', dest='Gmax', type=float, action='store',
            default=30.0, help='The max G in reciprocal space')
    parser.add_argument('--rmax', dest='Rmax', type=float, action='store',
            default=10.0, help='The max Rcut in real space')

    args = parser.parse_args()
    inputPP = args.inputPP
    outPP = args.outPP
    MaxPoints = args.MaxPoints
    Gmax = args.Gmax
    PP = PseudoPotential(inputPP, MaxPoints = MaxPoints, Gmax = Gmax)
    PP.output(outPP)

if __name__ == "__main__":
    RunJob()
