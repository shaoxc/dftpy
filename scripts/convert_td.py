#!/usr/bin/env python3
import sys
import numpy as np

from dftpy.td.interface import RealTimeRunnerPrintData, RealTimeRunnerPrintTitle


def convert_td(filename, dt, vector_potential):

    with open(filename+'_E', 'r') as fE:
        E = []
        for line in fE:
            E.append(float(line.rstrip('\n')))
        E.append(0)
        E = np.asarray(E)

    with open(filename+'_mu', 'r') as fmu:
        mu = []
        for line in fmu:
            mu.append([float(x) for x in line.rstrip('\n').split()])
        mu = np.asarray(mu)

    with open(filename+'_j', 'r') as fj:
        j = []
        for line in fj:
            j.append([float(x) for x in line.rstrip('\n').split()])
        j = np.asarray(j)

    if vector_potential:
        with open(filename + '_A', 'r') as fA:
            A = []
            for line in fA:
                A.append([float(x) for x in line.rstrip('\n').split()])
            A = np.asarray(A)

    len_t = np.size(E)
    t = np.linspace(0, dt*(len_t-1), len_t)

    with open(filename, 'w') as fin:
        RealTimeRunnerPrintTitle(fin, vector_potential)
        for i in range(len_t):
            RealTimeRunnerPrintData(fin, t[i], E[i], mu[i], j[i], A[i] if vector_potential else None)


if __name__ == '__main__':
    filename = sys.argv[1]
    dt = float(sys.argv[2])
    if len(sys.argv) <= 3 or sys.argv[3] == 0:
        vector_potential = False
    else:
        vector_potential = True
    convert_td(filename, dt, vector_potential)