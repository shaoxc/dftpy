.. _md:

=============================
Molecular Dynamics Simulation
=============================

DFTpy performs molecular dynamics (MD) simulations with `ASE`_.
This is one example to run NVT (canonical ensemble) simulation:

 .. literalinclude:: nvt.py

The output `md.traj` can be converted with `ASE`_:

 .. literalinclude:: ase_traj.py


.. _ASE: https://gitlab.com/ase/ase

