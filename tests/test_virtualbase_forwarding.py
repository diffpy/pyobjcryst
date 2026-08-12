#!/usr/bin/env python
"""Coverage guard for the RefinableObj/ScatteringPower method-forwarding fix.

Scatterer, ScatteringPower, ScatteringData, and PowderPatternComponent all
inherit RefinableObj *virtually* in C++, so nanobind cannot declare
RefinableObj as their Python-visible base (its base<->derived pointer
adjustment assumes a fixed offset, which a virtual base doesn't have).
Boost.Python's ``bases<RefinableObj>`` had no such problem, so under
Boost.Python these four classes -- and everything under them (Atom,
Molecule, ZScatterer, ZPolyhedron, ScatteringPowerAtom, ScatteringPowerSphere,
GlobalScatteringPower, DiffractionDataSingleCrystal, PowderPatternBackground,
PowderPatternDiffraction) -- got RefinableObj's full method surface for free
via ordinary Python inheritance. The initial nanobind port only restored a
handful of these methods as one-off shims; the rest (Print, XMLOutput,
BeginOptimization, GetLogLikelihood, UpdateDisplay, AddPar, ...) were simply
uncallable from Python. See nanobind_migration_notes.md's "Follow-up:
restoring RefinableObj's method surface on virtually-inherited classes".

This file exhaustively enumerates every method the canonical RefinableObj/
ScatteringPower Python types expose and asserts each of the affected classes
also has it -- so a future refactor that touches
``bind_refinableobj_forwarding<T>()``/``bind_scatteringpower_forwarding<T>()``
or any of their call sites in ``helpers_nb.hpp`` can't silently drop coverage
the way the original port did. Functional (not just presence) tests for a
representative sample live in test_scatterer.py, test_refinableobj.py,
test_powderpattern.py, and test_single_crystal_data.py.
"""

import unittest

import pyobjcryst._pyobjcryst as _p
from pyobjcryst.powderpattern import PowderPatternComponent
from pyobjcryst.refinableobj import RefinableObj
from pyobjcryst.scatterer import Scatterer
from pyobjcryst.scatteringdata import ScatteringData
from pyobjcryst.scatteringpower import ScatteringPower, ScatteringPowerAtom
from pyobjcryst.zscatterer import GlobalScatteringPower


def _public_methods(cls):
    return sorted(name for name in dir(cls) if not name.startswith("_"))


REFINABLEOBJ_METHODS = _public_methods(RefinableObj)
SCATTERINGPOWER_METHODS = _public_methods(ScatteringPower)


class TestRefinableObjSurfaceForwarding(unittest.TestCase):
    """Every RefinableObj method must be reachable on the four classes that
    only reach RefinableObj through a C++ virtual base."""

    def _assert_full_coverage(self, cls):
        missing = [m for m in REFINABLEOBJ_METHODS if not hasattr(cls, m)]
        self.assertEqual(
            missing,
            [],
            f"{cls.__name__} is missing RefinableObj methods: {missing}",
        )

    def test_scatterer(self):
        self._assert_full_coverage(Scatterer)

    def test_scatteringpower(self):
        self._assert_full_coverage(ScatteringPower)

    def test_scatteringdata(self):
        self._assert_full_coverage(ScatteringData)

    def test_powderpatterncomponent(self):
        self._assert_full_coverage(PowderPatternComponent)

    # Concrete leaves that inherit the four classes above *non-virtually*
    # should get the surface "for free" via ordinary Python inheritance --
    # verify that actually holds rather than assuming it.
    def test_atom(self):
        self._assert_full_coverage(_p.Atom)

    def test_molecule(self):
        self._assert_full_coverage(_p.Molecule)

    def test_zscatterer(self):
        self._assert_full_coverage(_p.ZScatterer)

    def test_zpolyhedron(self):
        self._assert_full_coverage(_p.ZPolyhedron)

    def test_scatteringpowersphere(self):
        self._assert_full_coverage(_p.ScatteringPowerSphere)

    def test_diffractiondatasinglecrystal(self):
        self._assert_full_coverage(_p.DiffractionDataSingleCrystal)

    def test_powderpatternbackground(self):
        self._assert_full_coverage(_p.PowderPatternBackground)

    def test_powderpatterndiffraction(self):
        self._assert_full_coverage(_p.PowderPatternDiffraction)

    # ScatteringPowerAtom and GlobalScatteringPower have a *second* virtual
    # inheritance jump (they inherit ScatteringPower virtually too), so they
    # can't ride on ScatteringPower's nanobind base either -- confirm they
    # were forwarded directly rather than assuming the cascade covers them.
    def test_scatteringpoweratom(self):
        self._assert_full_coverage(ScatteringPowerAtom)

    def test_globalscatteringpower(self):
        self._assert_full_coverage(GlobalScatteringPower)


class TestScatteringPowerSurfaceForwarding(unittest.TestCase):
    """ScatteringPowerAtom and GlobalScatteringPower need ScatteringPower's
    own method surface (GetScatteringFactor, GetBij/SetBij, GetColour, ...)
    forwarded too, on top of RefinableObj's."""

    def _assert_full_coverage(self, cls):
        missing = [m for m in SCATTERINGPOWER_METHODS if not hasattr(cls, m)]
        self.assertEqual(
            missing,
            [],
            f"{cls.__name__} is missing ScatteringPower methods: {missing}",
        )

    def test_scatteringpoweratom(self):
        self._assert_full_coverage(ScatteringPowerAtom)

    def test_globalscatteringpower(self):
        self._assert_full_coverage(GlobalScatteringPower)


if __name__ == "__main__":
    unittest.main()
