.. _config:

======================
Config of dftpy script
======================

.. list-table::

     * - `JOB`_
       - `PATH`_
       - `MATH`_
       - `PP`_
     * - `CELL`_
       - `GRID`_
       - `DENSITY`_
       - `EXC`_
     * - `KEDF`_
       - `OUTPUT`_
       - `OPT`_
       - 

JOB
----------

    Control the job running. 

.. option:: task

    The task to be performed.
        *Options* : Optdensity, Calculation

        *Default* : Optdensity

.. option:: calctype

    The property to be calculated.
        *Options* : Energy, Potential, Both, Force, Stress

        *Default* : Energy

PATH
----------

    Control the path of input files.

.. option:: pppath

    The path of pseudopotential.
        *Options* : 

        *Default* : None

.. option:: cellpath

    The path of input structure.
        *Options* : 

        *Default* : None


MATH
----------

    Some method and technique to make code faster.

.. option:: linearii

    The linear method to deal with Ion-Ion interaction.
        *Options* : True, False

        *Default* : True

.. option:: linearie

    The linear method to deal with Ion-Electron interaction.
        *Options* : True

        *Default* : True

.. option:: twostep

    The multi-step method (two steps) to perform density optimization. if '`True`' is same as :option:`multistep` = 2.
        *Options* : True, False

        *Default* : False

.. option:: multistep

    The multi-step method to perform density optimization.
        *Options* : 1,2,...

        *Default* : 1

.. option:: reuse

    Except first step, the initial density is given by optimization density of previous step.
        *Options* : True, False

        *Default* : True 


PP
----------

    The pseudopotential file of each elements.

        *e.g.*

        - *Al* = Al_lda.oe01.recpot
        - *Mg* = Mg_lda.oe01.recpot



CELL
----------

    The information of input structure.

.. option:: cellfile

    The file of input structure.
        *Options* :  

        *Default* : POSCAR

.. option:: elename

    The name of atom.
        *Options* : 

        *Default* : Al

.. option:: zval

    The charge of atomic species.
        *Options* : 

        *Default* : None

.. option:: format

    The format of structure file.
        *Options* : pp, vasp, xsf,...

        *Default* : None


GRID
----------

    Control the grid.

.. option:: ecut

    The kinetic energy cutoff (eV).
        *Options* : 

        *Default* : 600

.. option:: spacing

    The spacing of real space grid. If set this, :option:`ecut` will no longer working.
        *Options* : 

        *Default* : None

.. option:: gfull

    The number of grid points in G-space is equal to real space, or not. if '`False`' only use half grid, which will be faster.
        *Options* : True, False

        *Default* : False

.. option:: nr

    Given the number of grid points in three directions.
        *Options* : 

        *Default* : None

        *e.g.*

            *nr* = 32 32 32


DENSITY
----------

    Control the charge density information.

.. option:: densityini

    The initial density is given by homogeneous electron gas (HEG) or read from :option:`densityfile`. If set `Read`, must given the :option:`densityfile`.
        *Options* : HEG, Read

        *Default* : HEG

.. option:: densityfile

    The charge density for initial density, only works when if :option:`densityini` set `Read`.
        *Options* : 

        *Default* : None

.. option:: densityoutput

    The output file of final density. The default is not output the density.
        *Options* : 

        *Default* : None


EXC
----------

    Control the exchange-correlation.

.. option:: xc

    The kind of exchange-correlation. If not set `LDA`, must be make sure already installed pylibxc_, and not contain stress calculation.
        *Options* : LDA, PBE,...

        *Default* : LDA

.. option:: x_str

    The formular of exchange functionals.
        *Options* : 

        *Default* : lda_x

.. option:: c_str

    The format of correlation functionals.
        *Options* : 

        *Default* : lda_c_pz


KEDF
----------

    Control the kinetic energy density functional (KEDF).

.. option:: kedf

    The format of KEDF.
        *Options* : TF, vW, x_TF_y_vW, WT, MGP,...

        *Default* : WT

.. option:: x

    The ratio of TF KEDF.
        *Options* : 

        *Default* : 1.0

.. option:: y

    The ratio of vW KEDF.
        *Options* : 

        *Default* : 1.0

.. option:: alpha

    The parameter of Non-local KEDF :math:`\rho^{\alpha}`.
        *Options* : 

        *Default* : 5.0/6.0

.. option:: beta

    The parameter of Non-local KEDF :math:`\rho^{\beta}`.
        *Options* : 

        *Default* : 5.0/6.0

.. option:: sigma

    The parameter for `FFT`.
        *Options* : 

        *Default* : 0.025

.. option:: nsp

    The number of :math:`{k_{f}}` for spline in `LWT` KEDF. There are three options to do same thing, the priority is :option:`nsp` -> :option:`delta` -> :option:`ratio`. Default is using :option:`ratio`.
        *Options* : 

        *Default* : None 

.. option:: delta

    The gap of :math:`{k_{f}}` for spline in `LWT` KEDF. There are three options to do same thing, the priority is :option:`nsp` -> :option:`delta` -> :option:`ratio`. Default is using :option:`ratio`.
        *Options* : 

        *Default* : None 

.. option:: ratio

    The ratio of :math:`{k_{f}}` for spline in `LWT` KEDF. There are three options to do same thing, the priority is :option:`nsp` -> :option:`delta` -> :option:`ratio`. Default is using :option:`ratio`.
        *Options* : 

        *Default* : 1.2

.. option:: interp

    The interpolate method for `LWT` KEDF.
        *Options* : linear, newton, hermite

        *Default* : hermite

.. option:: kerneltype

    The kernel for `LWT` KEDF.
        *Options* : WT, MGP

        *Default* : WT

.. option:: symmetrization

    The symmetrization way for `MGP` KEDF.
        *Options* : None, Arithmetic, Geometric

        *Default* : None

.. option:: lumpfactor

    The kinetic electron for `LWT` KEDF.
        *Options* : 

        *Default* : None

.. option:: neta

    The max number of discrete :math:`\eta` for `LWT` KEDF.
        *Options* : 

        *Default* : 50000 

.. option:: etamax

    The max value of \eta for kernel in `LWT` KEDF.
        *Options* : 

        *Default* : 50.0

.. option:: order

    The order for interpolate the kernel in `LWT` KEDF. '0' means using the value of nearest-neighbor point.
        *Options* : 

        *Default* : 3

.. option:: maxpoints

    The max number for evaluation of `MGP` kernel.
        *Options* : 

        *Default* : 1000

.. option:: kdd

    The kernel density denpendent for `LWT` KEDF:
        + 1 : The origin `LWT` KEDF.
        + 2 : Conside the :math:`\rho^{\beta}(r')\omega(\rho(r),r-r')`.
        + 3 : Also considering the derivative of kernel.

        *Options* : 1,2,3

        *Default* : 3 

.. option:: rho0

    The 'average' density for the Fermi momentum. Default is None, which means it calculated based on the total charge and system volume.
        *Options* : 

        *Default* : None


OUTPUT 
----------

    Control the output.

.. option:: time

    Output the time information of all parts.
        *Options* : True, False

        *Default* : True

.. option:: stress

    Output the stress information of all terms.
        *Options* : 

        *Default* : True


OPT
----------

    Control the charge density optimization.

.. option:: method

    The density optimization method.
        *Options* : TN, LBFGS, CG-HS, CG-DY, CG-CD, CG-LS, CG-FR, CG-PR

        *Default* : CG-HS

.. option:: algorithm

    The direct minimization method : Energy (EMM) or Residual (RMM).
        *Options* : EMM, RMM

        *Default* : EMM 

.. option:: vector

    The scheme to deal with search direction.
        *Options* :  Orthogonalization, Scaling

        *Default* : Orthogonalization

.. option:: c1

    The wolfe parameters `c1`
        *Options* : 

        *Default* : 1e-4

.. option:: c2

    The wolfe parameters `c2`
        *Options* : 

        *Default* : 2e-1

.. option:: maxls

    The max steps for line search.
        *Options* : 

        *Default* : 10

.. option:: econv

    The energy convergence for last three steps (a.u./atom).
        *Options* : 

        *Default* : 1e-6 

.. option:: maxfun

    The max steps for function calls. For `TN` density optimization method its the max steps for searching direction.
        *Options* : 

        *Default* : 50  

.. option:: maxiter

    The max steps for optimization
        *Options* : 

        *Default* : 100

.. option:: xtol

    Relative tolerance for an acceptable step.
        *Options* : 

        *Default* : 1e-12

.. option:: h0

    The initial approximate of the inverse Hessian for `LBFGS`.
        *Options* : 

        *Default* : 1.0 


.. note::
    The defaults are work well for most arguments, only `PP`_ and `CELL`_ must be given.

    The *Options* not given means, it can accept any `float` or `integer`.

.. _pylibxc: https://tddft.org/programs/libxc/
