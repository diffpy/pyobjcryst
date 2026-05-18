#!/usr/bin/env python
##############################################################################
#
# pyobjcryst
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
"""Python wrapping of ReflectionProfile.h.

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io).

A ``ReflectionProfile`` describes the shape of a single Bragg
reflection (pseudo-Voigt by default) and is owned by a
``PowderPatternDiffraction``. The Python bindings expose the
public methods ``GetProfile``, ``GetFullProfileWidth``,
``XMLOutput`` / ``XMLInput`` and ``CreateCopy``. ``GetProfile``
accepts a Python sequence or numpy array for ``x``.

Example
-------

Sample the profile of a single ``(h, k, l)`` reflection from an
existing ``PowderPatternDiffraction`` and broaden it by tweaking
the ``W`` Caglioti parameter::

    import numpy as np
    from pyobjcryst.powderpattern import PowderPattern

    pp = PowderPattern()
    pp.SetWavelength(0.7)
    x = np.deg2rad(np.linspace(0, 40, 1000))
    pp.SetPowderPatternX(x)
    pp.SetPowderPatternObs(np.ones_like(x))
    ppd = pp.AddPowderPatternDiffraction(crystal)   # crystal: pyobjcryst.Crystal

    profile = ppd.GetProfile()
    window = x[100:200]
    xcenter = float(window[len(window) // 2])

    y = profile.GetProfile(window, xcenter, 1, 0, 0)
    fwhm = profile.GetFullProfileWidth(0.5, xcenter, 1, 0, 0)

    profile.GetPar("W").SetValue(0.05)
    y_broader = profile.GetProfile(window, xcenter, 1, 0, 0)
"""

__all__ = ["ReflectionProfile", "ReflectionProfileType"]

from pyobjcryst._pyobjcryst import ReflectionProfile, ReflectionProfileType
