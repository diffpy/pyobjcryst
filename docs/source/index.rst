#######
|title|
#######

.. |title| replace:: pyobjcryst documentation

``pyobjcryst`` - Python bindings to the ObjCryst++ Object-Oriented Crystallographic Library

| Software version |release|.
| Last updated |today|.

=======
Authors
=======

`pyobjcryst` was written as part of the DANSE_ open-source project by
Christopher Farrow, Pavol Juhás, and Simon J.L. Billinge.
The sources are maintained as a part of the DiffPy-CMI complex
modeling initiative at the Brookhaven National Laboratory.
The underlying library  `ObjCryst++ <https://github.com/vincefn/objcryst>`_
was developed by V. Favre-Nicolin as part of the development of the
`Fox <http://fox.vincefn.net>`_ software.

Further developments including the ability to index and refine
powder patterns, solve and display crystal structures, using the
global optimisation and least squares algorithms (see the
:doc:`examples/index`) are provided by Vincent Favre-Nicolin (ESRF).

For a complete list of contributors, see
https://github.com/diffpy/pyobjcryst/graphs/contributors and
https://github.com/diffpy/libobjcryst/graphs/contributors.

.. _DANSE: https://www.its.caltech.edu/~matsci/btf/DANSE_web_page.html

============
Installation
============

See the `README <https://github.com/diffpy/pyobjcryst#installation>`_
file included with the distribution.

=================
Usage & notebooks
=================

`pyobjcryst` can be used

* **as a backend library** to manage crystal structures description in an application
  like `DiffPy-CMI <https://www.diffpy.org/products/diffpycmi/index.html>`_
* **in python scripts or notebooks**, allowing to:

  * display crystal structures,
  * index and refine powder diffraction patterns
  * solve crystal structures from diffraction data using global optimisation algorithms
  * etc..

  The functionality is similar to what is available in `Fox <http://fox.vincefn.net>`_.
  See the :doc:`examples/index`:

  * :doc:`3D Crystal structure display <examples/crystal_3d_widget>`
  * :doc:`Solving the cimetidine structure from its powder pattern <examples/structure-solution-powder-cimetidine>`
  * :doc:`Solving the PbSO4 structure from its X and N powder patterns <examples/structure-solution-powder-pbso4>`
  * :doc:`Meta-structure solution using multi-processing <examples/structure-solution-multiprocessing>`
  * :doc:`Quantitative phase analysis (QPA) <examples/Quantitative-phase-analysis>`

The **API documentation** can be found in :doc:`api/pyobjcryst`.

You can also read the
`documentation of the underlying ObjCryst++ library <https://vincefn.net/ObjCryst/annotated.html>`_.

================
Acknowledgements
================

``pyobjcryst`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.

=================
Table of contents
=================
.. toctree::
   :titlesonly:

   license
   release
   Examples <examples/index>
   Package API <api/pyobjcryst>

=======
Indices
=======

* :ref:`genindex`
* :ref:`search`
