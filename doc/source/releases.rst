===============
Stable releases
===============

Version 2.0.0
=============

Release date: `May 2022 <https://gitlab.com/pavanello-research-group/dftpy/-/releases/v2.0.0>`_

This version supports:
 - The Ions object is inherit from `ASE.atoms <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_
 - Parallel implementation with mpi4py_
 - Supported spin polarized runs
 - New td-OF-DFT code for real-time propagation and Casida
 - Nonadiabatic Pauli potential with the JP_ functional
 - Supported non-standard semilocal kinetic functionals , such as LKT_ and PG_
 - Meta-GGA functional
 - Non-local exchange-correlation functional (rVV10)
 - One-orbital ensemble self-consistent field (OE-SCF) solver
 - Initial density generator


Version 1.0.0
=============

Release date: `January 2020 <https://gitlab.com/pavanello-research-group/dftpy/-/releases/dftpy-1.0>`_

This version supports:
 - OFDFT electron density optimization
 - Latest nonlocal KEDFs
 - Ab initio dynamics
 - Hydrodynamic DFT (TD-OFDFT)
 - I/O for plotting and to several file formats
 - Several examples (also on Jupyter notebooks)

.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _LKT: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.041111 
.. _JP: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.235110
.. _PG: https://pubs.acs.org/doi/full/10.1021/acs.jpclett.8b01926

