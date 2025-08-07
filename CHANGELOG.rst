=============
Release notes
=============

.. current developments

2025.1.0
=====

**Removed:**

* Removed fallback version handling in `src/pyobjcryst/version.py`.


Version 2024.X.X
----------------

Changes
~~~~~~~

Fixes
~~~~~

- Correct powder pattern plotting with a non-empty name

Version 2024.2.1
----------------

Changes
~~~~~~~

- **PowderPattern**:
  - Fix re-using a matplotlib figure when plotting
  - Add ``figure`` property

Version 2024.2
--------------

Changes
~~~~~~~

- **DiffractionDataSingleCrystal**: add ``SetHklIobs``, ``SetIobs``, ``SetSigma``, ``GetSigma``, ``GetChi2``, ``FitScaleFactorForRw`` and ``FitScaleFactorForR`` (`issue #42 <https://github.com/diffpy/pyobjcryst/issues/42>`_)
- Add a single crystal data notebook example
- Online documentation notebooks now include the plots `<https://pyobjcryst.readthedocs.io/en/latest/examples>`_

Fixes
~~~~~

- From libobjcryst: update the ScatteringComponentList when a Scatterer is removed from a Crystal (`issue #41 <https://github.com/diffpy/pyobjcryst/issues/41>`_)

Version 2024.1
--------------

Changes
~~~~~~~

- Add python access to MolZAtom, for ``Molecule.AsZMatrix()``

Version 2.2.6
--------------

Changes
~~~~~~~

- Support for Windows and Python>=3.8
- Added a zoom limit for 3D crystal views

Fixes
~~~~~

- Correct error preventing pyobjcryst import for Windows and Python>=3.8 (`issue #33 <https://github.com/diffpy/pyobjcryst/issues/33>`_)
- Fix for matplotlib >=3.7.0 when removing hkl labels

Version 2.2.5
--------------

Changes
~~~~~~~

- Raise an exception if ``alpha``, ``beta`` or ``gamma`` are not within ``]0;pi[`` when changing lattice angles
- Add ``UnitCell.ChangeSpaceGroup()``

Fixes
~~~~~

- Avoid duplication of plots when using ipympl (aka ``%matplotlib widget``)
- Correct powder pattern tests to avoid warnings

Deprecated
~~~~~~~~~~

- ``loadCrystal`` – use ``create_crystal_from_cif()`` instead

Version 2.2.4
--------------

Changes
~~~~~~~

- The list of HKL reflections will now be automatically re-generated for a ``PowderPatternDiffraction`` when the Crystal's spacegroup changes, or the lattice parameters are modified by more than 0.5%

Fixes
~~~~~

- Fixed the powder pattern indexing test

Version 2.2.3
--------------

Added
~~~~~

- Support for Windows install (works with Python 3.7, and with PyPy 3.8 and 3.9)
- Native support for Apple arm64 (M1, M2) processors
- Fourier maps calculation
- Add ``gDiffractionDataSingleCrystalRegistry`` to globals

Version 2.2.2
--------------

Changes
~~~~~~~

- Add correct wrapping for C++-instantiated objects available through global registries, e.g. when loading an XML file. The objects are decorated with the python functions when accessed through the global registries ``GetObj()``
- Moved global object registries to ``pyobjcryst.globals``
- Update documentation

Fixed
~~~~~

- Fix access to ``PRISM_TETRAGONAL_DICAP``, ``PRISM_TRIGONAL``, ``ICOSAHEDRON`` and ``TRIANGLE_PLANE``
- Fix powder pattern plot issues (NaN and update of hkl text with recent matplotlib versions)

Version 2.2.1 -- 2021-11-28
----------------------------

- Add quantitative phase analysis with ``PowderPattern.qpa()``, including an example notebook using the QPA Round-Robin data
- Correct import of ``urllib.request.urllopen()`` when loading CIF or z-matrix files from HTTP URLs
- Fix blank line javascript output when updating the Crystal 3D view
- Add ``RefinableObj.xml()`` to directly get the XMLOutput as a string
- Add example notebooks to the sphinx-generated html documentation
- Fix issue when using ``Crystal.XMLInput()`` for a non-empty structure. Existing scattering power will be re-used when possible, and otherwise not deleted anymore (which could lead to crashes)

Version 2.2.0 -- 2021-06-08
----------------------------

- Add access to ``Radiation`` class & functions to change RadiationType, wavelength in ``PowderPattern`` and ``ScatteringData`` (and hence ``DiffractionDataSingleCrystal``) classes
- Fix the custodian_ward when creating a ``PowderPatternDiffraction``: ``PowderPatternDiffraction`` must persist while ``PowderPattern`` exists, and Crystal must persist while ``PowderPatternDiffraction`` exists
- Add 3D Crystal viewer ``pyobjcryst.crystal.Crystal.widget_3d``

Version 2.1.0 -- 2019-03-11
----------------------------

Added
~~~~~

- Support for Python 3.7
- Validation of compiler options from ``python-config``
- Make scons scripts compatible with Python 3 and Python 2
- Support ``np.array`` arguments for ``SetPowderPatternX``, ``SetPowderPatternObs``
- Declare compatible version requirements for client Anaconda packages
- Facility for silencing spurious console output from libobjcryst

Changed
~~~~~~~

- Build Anaconda package with Anaconda C++ compiler
- Update to libobjcryst 2017.2.x

Deprecated
~~~~~~~~~~

- Variable ``__gitsha__`` in the ``version`` module, renamed to ``__git_commit__``

Removed
~~~~~~~

- Support for Python 3.4

Fixed
~~~~~

- Ambiguous use of boost::python classes and functions
- Name suffix resolution of ``boost_python`` shared library
- ``SetPowderPatternX`` crash for zero-length argument
- Incorrectly doubled return value from ``GetInversionCenter``
