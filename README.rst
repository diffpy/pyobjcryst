|Icon| |title|_
===============

.. |title| replace:: pyobjcryst
.. _title: https://diffpy.github.io/pyobjcryst

.. |Icon| image:: https://avatars.githubusercontent.com/diffpy
        :target: https://diffpy.github.io/pyobjcryst
        :height: 100px

|PyPI| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/diffpy/pyobjcryst/actions/workflows/matrix-and-codecov-on-merge-to-main.yml/badge.svg
        :target: https://github.com/diffpy/pyobjcryst/actions/workflows/matrix-and-codecov-on-merge-to-main.yml

.. |Codecov| image:: https://codecov.io/gh/diffpy/pyobjcryst/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/diffpy/pyobjcryst

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/pyobjcryst
        :target: https://anaconda.org/conda-forge/pyobjcryst

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff
        :target: https://github.com/diffpy/pyobjcryst/pulls

.. |PyPI| image:: https://img.shields.io/pypi/v/pyobjcryst
        :target: https://pypi.org/project/pyobjcryst/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/pyobjcryst
        :target: https://pypi.org/project/pyobjcryst/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/diffpy/pyobjcryst/issues

Python bindings to ObjCryst++, the Object-Oriented Crystallographic Library.


For more information about the pyobjcryst library, please consult our `online documentation <https://diffpy.github.io/pyobjcryst>`_.

Citation
--------

If you use pyobjcryst in a scientific publication, we would like you to cite this package as

        pyobjcryst Package, https://github.com/diffpy/pyobjcryst

Installation
------------

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
and install from the "conda-forge" channel of Conda packages.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``pyobjcryst_env`` ::

        conda create -n pyobjcryst_env pyobjcryst
        conda activate pyobjcryst_env

To confirm that the installation was successful, type ::

        python -c "import pyobjcryst; print(pyobjcryst.__version__)"

The output should print the latest version displayed on the badges above.

If the above does not work, you can use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``pyobjcryst_env`` environment, type ::

        pip install pyobjcryst

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/diffpy/pyobjcryst/>`_. Once installed, ``cd`` into your ``pyobjcryst`` directory
and run the following ::

        pip install .

An alternative way of installing pyobjcryst is to use the SCons tool,
which can speed up the process by compiling C++ files in several
parallel jobs (-j4)::

   scons -j4 dev

See ``scons -h`` for description of build targets and options.

Optional graphical dependencies for jupyter notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some of the classes can produce graphical outputs, which can be
displayed in a jupyter notebook:

* a Crystal structure can be displayed in 3D: this requires the
  ``py3dmol`` and ``ipywidgets`` modules. See the notebook
  ``examples/cystal_3d_widget.ipynb``
* a PowderPattern can be displayed (and live-updated) if
  ``matplotlib`` and ``ipympl`` are installed. See the
  notebook ``examples/cimetidine-structure-solution-powder.ipynb``

Getting Started
---------------

You may consult our `online documentation <https://diffpy.github.io/pyobjcryst>`_ for tutorials and API references.

Support and Contribute
----------------------

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/diffpy/pyobjcryst/issues>`_ and/or `submit a fix as a PR <https://github.com/diffpy/pyobjcryst/pulls>`_.

Feel free to fork the project and contribute. To install pyobjcryst
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

When developing it is preferable to compile the C++ files with
SCons using the ``build=debug`` option, which compiles the extension
module with debug information and C-assertions checks ::

   scons -j4 build=debug dev

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/diffpy/pyobjcryst/blob/main/CODE_OF_CONDUCT.rst>`_.

Contact
-------

For more information on pyobjcryst please visit the project `web-page <https://diffpy.github.io/>`_ or email Simon Billinge at sb2896@columbia.edu.

Acknowledgements
----------------

``pyobjcryst`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.
