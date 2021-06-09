def GetConf():
    import argparse
    from dftpy.mpi import pmi
    parser = argparse.ArgumentParser(description='Process task')
    parser.add_argument('confs', nargs='*')
    parser.add_argument('-i', '--ini', '--input', dest='input', type=str, action='store',
                        default='config.ini', help='Input file (default: config.ini)')
    parser.add_argument('--mpi', '--mpi4py', dest='mpi', action='store_true',
                        default=False, help='Use mpi4py to be parallel')

    args = parser.parse_args()
    if args.mpi or pmi.size > 0:
        from mpi4py import MPI
        from dftpy.mpi import mp
        mp.comm = MPI.COMM_WORLD
    return args


def RunJob(args):
    from dftpy.interface import ConfigParser, OptimizeDensityConf, InvertRunner
    from dftpy.td.interface import RealTimeRunner, CasidaRunner, DiagonalizeRunner
    import time
    from dftpy.time_data import TimeData
    from dftpy.mpi import mp, sprint

    sprint("Begin on :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
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

        if "Propagate" in config["JOB"]["task"]:
            RealTimeRunner(config, others["struct"].field, others["E_v_Evaluator"])
        elif "Casida" in config["JOB"]["task"]:
            CasidaRunner(config, others["struct"].field, others["E_v_Evaluator"])
        elif "Diagonalize" in config["JOB"]["task"]:
            DiagonalizeRunner(config, others["struct"], others["E_v_Evaluator"])
        elif "Inversion" in config["JOB"]["task"]:
            InvertRunner(config, others["struct"], others["E_v_Evaluator"])
        else:
            OptimizeDensityConf(config, others["struct"], others["E_v_Evaluator"], others["nr2"])

        TimeData.End("TOTAL")
        TimeData.output(config)
        sprint("-" * 80)
    sprint("#" * 80)
    sprint("Finished on :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def main():
    import sys
    from dftpy.mpi.utils import sprint

    if sys.version_info < (3, 6):
        sprint('Please upgrade your python to version 3.6 or higher.')
        sys.exit()
    args = GetConf()
    RunJob(args)
