.. _relax:

=======================
Relaxation of Structure
=======================

Perform the structure relaxation is also easy with `ASE`_.
This is one example to relax the force of given structure:

.. literalinclude:: relax.py
    :lines: 1-32

If we want to optimize the stress, just add the `StrainFilter` :

.. literalinclude:: relax.py
    :lines: 33

We also can optimize the force and stress at same time, just add the `UnitCellFilter` :

.. literalinclude:: relax.py
    :lines: 34

The output `opt.traj` can convert to other format via `ASE`_:

.. literalinclude:: relax.py
    :lines: 40-41

For more functions and detailed introduction, can refer to `ASE`_.

.. _ASE: https://gitlab.com/ase/ase
