#!/usr/bin/env python3

from dftpy.interface import ConfigParser
import time
from dftpy.time_data import TimeData
from dftpy.mpi import mp, sprint
from dftpy import __version__
from dftpy.cui.main import GetConf
from dftpy.formats.io import write
import numpy as np


def RunJob(args):

    sprint("DFTpy {} Begin on : {}".format(__version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if mp.is_mpi:
        info = 'Parallel version (MPI) on {0:>8d} processors'.format(mp.comm.size)
    else:
        info = 'Serial version on {0:>8d} processors'.format(mp.comm.size)
    sprint(info)
    if len(args.confs) == 0:
        args.confs.append(args.input)
    for fname in args.confs:
        config, others = ConfigParser(fname, mp=mp)
        sprint("#" * 80)
        TimeData.Begin("TOTAL")

        ions = others['ions']
        field = others['field']
        field = np.abs(field)
        outfile = config["DENSITY"]["densityoutput"]
        write(outfile=outfile, data=field, ions=ions)

        TimeData.End("TOTAL")
        TimeData.output(config)
        sprint("-" * 80)
    sprint("#" * 80)
    sprint("DFTpy {} Finished on : {}".format(__version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


def main():
    import sys
    from dftpy.mpi.utils import sprint

    if sys.version_info < (3, 6):
        sprint('Please upgrade your python to version 3.6 or higher.')
        sys.exit()
    args = GetConf()
    RunJob(args)


if __name__ == '__main__':
    main()