#!/usr/bin/env python
"""Tests for bindings that return ObjCryst's CrystVector / CrystMatrix.

nanobind has no global converter for these types: a binding that returns
them without ``crystvec_to_array`` / ``crystmat_to_array`` fails only
when called, with "Unable to convert function return value to a Python
type".
"""

import unittest

import numpy as np

import pyobjcryst._pyobjcryst as _pyobjcryst
from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal
from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.scatteringpower import ScatteringPowerAtom
from pyobjcryst.scatteringpowersphere import ScatteringPowerSphere


def _corundum():
    crystal = Crystal(4.76, 4.76, 12.99, "R-3c")
    al = ScatteringPowerAtom("Al", "Al")
    o = ScatteringPowerAtom("O", "O")
    crystal.AddScatteringPower(al)
    crystal.AddScatteringPower(o)
    crystal.AddScatterer(Atom(0.0, 0.0, 0.35216, "Al1", al))
    crystal.AddScatterer(Atom(0.3064, 0.0, 0.25, "O1", o))
    return crystal, al


class TestNoRawCrystVectorReturns(unittest.TestCase):

    def test_signatures(self):
        """No binding's signature returns an unconverted C++ type."""
        raw = []
        for class_name, cls in vars(_pyobjcryst).items():
            if not isinstance(cls, type):
                continue
            for name, member in vars(cls).items():
                doc = getattr(member, "__doc__", None) or ""
                for line in doc.splitlines():
                    if "->" not in line:
                        continue
                    return_type = line.rsplit("->", 1)[-1]
                    if "CrystVector" in return_type or (
                        "CrystMatrix" in return_type
                    ):
                        raw.append(f"{class_name}.{name}: {line.strip()}")
        self.assertEqual(raw, [])


class TestArrayReturns(unittest.TestCase):

    def setUp(self):
        self.crystal, self.al = _corundum()

    def test_GetLatticePar(self):
        # b and gamma are derived from the R-3c constraints
        self.crystal.a = 5.0
        lattice = self.crystal.GetLatticePar()
        self.assertIsInstance(lattice, np.ndarray)
        expected = [self.crystal.GetLatticePar(i) for i in range(6)]
        np.testing.assert_allclose(lattice, expected)
        np.testing.assert_allclose(
            lattice, [5.0, 5.0, 12.99, np.pi / 2, np.pi / 2, 2 * np.pi / 3]
        )

    def test_GetOrthonormalCoords(self):
        xyz = self.crystal.GetOrthonormalCoords(0.1, 0.2, 0.3)
        self.assertIsInstance(xyz, np.ndarray)
        np.testing.assert_allclose(
            xyz, self.crystal.FractionalToOrthonormalCoords(0.1, 0.2, 0.3)
        )

    def test_GetMinDistanceTable(self):
        table = self.crystal.GetMinDistanceTable(1.0)
        self.assertIsInstance(table, np.ndarray)
        self.assertEqual(table.shape, (2, 2))
        np.testing.assert_allclose(table, table.T)
        # shortest Al-O distance in corundum
        self.assertAlmostEqual(table[0, 1], 1.85, places=2)

    def test_scattering_power_factors(self):
        pp = PowderPattern()
        pp.SetWavelength(1.0)
        pp.SetPowderPatternX(np.radians(np.linspace(10.0, 60.0, 200)))
        pp.AddPowderPatternDiffraction(self.crystal)
        data = pp.get_crystalline_components()[0]
        pp.GetPowderPatternCalc()
        n_refl = data.GetNbRefl()
        self.assertGreater(n_refl, 0)

        # ScatteringPowerAtom uses the forwarding bindings,
        # ScatteringPowerSphere the ScatteringPower base bindings
        sphere = ScatteringPowerSphere("sphere", 2.0, 1.0)
        for sp in (self.al, sphere):
            with self.subTest(type(sp).__name__):
                for name in ("GetScatteringFactor", "GetTemperatureFactor"):
                    values = getattr(sp, name)(data)
                    self.assertIsInstance(values, np.ndarray)
                    self.assertEqual(values.shape, (n_refl,))
                    self.assertTrue(np.all(np.isfinite(values)))
                for name in (
                    "GetResonantScattFactReal",
                    "GetResonantScattFactImag",
                ):
                    values = getattr(sp, name)(data)
                    self.assertIsInstance(values, np.ndarray)
                    self.assertEqual(values.ndim, 2)

        self.assertGreater(self.al.GetScatteringFactor(data)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
