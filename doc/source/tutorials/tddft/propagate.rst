.. _propagate:

=======================
Use of Propagator
=======================

Here we provide an example of calculating the dipole moments using OF-TDDFT with the config files.

Suppose we have a GaAs cluster with its density optimized to the ground state. At t=0 we apply a kick at x-direction and
we want to calculate the dipole moment of the cluster at t=0.1 a.u.

The first step is to optimize the ground state density. We create a config file named optimize.ini:

.. literalinclude:: optimize.ini

Then we run the following command:

.. highlight:: bash

::

    $ python -m dftpy optimize.ini

which will generate a density file names density in the working directory.

The next step is to run the propagation. We create another config file named propagate.ini:

.. literalinclude:: propagate.ini

One thing to keep in mind is when running the propagate job, the kedf is the Pauli KEDF (i.e. total KESF minus the von
Weizsacker term).

Then we run the following command:

.. highlight:: bash

::

    $ python -m dftpy propagate.ini

It will generate three files: td_out_E, td_out_mu, td_out_j, which contains the energy of the electrons, the dipole
moment, and the integrated current of each time step, respectively.
