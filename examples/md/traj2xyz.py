#!/usr/bin/env python3

import os, sys
from ase.io.trajectory import Trajectory
from ase.io import write

infile = sys.argv[1]
outfile = sys.argv[2]

traj = Trajectory(infile)
write(outfile, traj)
