####################################################
Example notebooks
####################################################

Several examples available in the pyobjcryst repository:


:doc:`3D Crystal structure display <crystal_3d_widget>`
=======================================================

Example of importing a CIF file from a file or the
`Crystallography Open Database <http://crystallography.net/cod/>`_
and displaying it in a widget using
`3dmol.js <https://3dmol.csb.pitt.edu/>`_.

:doc:`Solving a crystal structure from its powder pattern <cimetidine-structure-solution-powder>`
=================================================================================================

In this example, a powder pattern is used to solve the crystal
structure of Cimetidine. This covers all the steps: loading the
data, indexing the pattern (determining the unit cell), finding
the spacegroup, profile fitting, and solving the structure
using a global optimisation algorithm.

:doc:`Meta-structure solution using multi-processing <structure-solution-multiprocessing>`
==========================================================================================

This is a more advanced example where 8 different spacegroups are
tested in parallel to determine which one is correct. The solutions
can then be compared and displayed individually.

:doc:`Quantitative phase analysis (QPA) <Quantitative-phase-analysis>`
______________________________________________________________________

Example of QPA based on the data available from the `1999 Round Robin
<https://www.iucr.org/__data/iucr/powder/QARR/samples.htm>`_,
in the case where all present crystalline structures are known
and there is no preferred orientation.
