"""Unit tests for pyobjcryst.reflectionprofile bindings.

TODO:
- ReflectionProfile.GetProfile
- ReflectionProfile.GetFullProfileWidth
- ReflectionProfile.XMLOutput / XMLInput
- ReflectionProfile.CreateCopy
"""

import unittest

import numpy as np
import pytest

from pyobjcryst.powderpattern import PowderPattern
from pyobjcryst.refinableobj import RefinableObj


class TestReflectionProfile(unittest.TestCase):
    """Tests for ReflectionProfile methods."""

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, loadcifdata):
        self.loadcifdata = loadcifdata

    def setUp(self):
        """Set up a ReflectionProfile instance for testing."""
        x = np.linspace(0, 40, 1000)
        c = self.loadcifdata("paracetamol.cif")

        self.pp = PowderPattern()
        self.pp.SetWavelength(0.7)
        self.pp.SetPowderPatternX(np.deg2rad(x))
        self.pp.SetPowderPatternObs(np.ones_like(x))

        self.ppd = self.pp.AddPowderPatternDiffraction(c)

        self.profile = self.ppd.GetProfile()

    def test_get_computed_profile(self):
        """Sample a profile slice and verify broadening lowers the peak height."""
        x = self.pp.GetPowderPatternX()
        hkl = (1, 0, 0)
        window = x[100:200]
        xcenter = float(window[len(window) // 2])

        prof_default = self.profile.GetProfile(window, xcenter, *hkl)
        self.assertEqual(len(prof_default), len(window))
        self.assertGreater(prof_default.max(), 0)

        # broaden and ensure the peak height drops while shape changes
        self.profile.GetPar("W").SetValue(0.05)
        prof_broader = self.profile.GetProfile(window, xcenter, *hkl)

        self.assertFalse(np.allclose(prof_default, prof_broader))
        self.assertLess(prof_broader.max(), prof_default.max())
        self.assertEqual(len(prof_default), len(prof_broader))

    def test_get_profile_width(self):
        """Ensure full-width increases when W increases."""
        xcenter = float(self.pp.GetPowderPatternX()[len(self.pp.GetPowderPatternX()) // 4])
        width_default = self.profile.GetFullProfileWidth(0.5, xcenter, 1, 0, 0)
        self.assertGreater(width_default, 0)

        self.profile.GetPar("W").SetValue(0.05)
        width_broader = self.profile.GetFullProfileWidth(0.5, xcenter, 1, 0, 0)
        self.assertGreater(width_broader, width_default)

    def test_create_copy(self):
        """Ensure copy returns an independent profile with identical initial params."""
        copy = self.profile.CreateCopy()

        self.assertIsNot(copy, self.profile)
        self.assertEqual(copy.GetClassName(), self.profile.GetClassName())

        eta0_original = self.profile.GetPar("Eta0").GetValue()
        eta0_copy = copy.GetPar("Eta0").GetValue()
        self.assertAlmostEqual(eta0_copy, eta0_original)

        self.profile.GetPar("Eta0").SetValue(eta0_original + 0.1)
        copy.GetPar("Eta0").SetValue(eta0_copy + 0.2)

        self.assertAlmostEqual(copy.GetPar("Eta0").GetValue(), eta0_original + 0.2)
        self.assertAlmostEqual(self.profile.GetPar("Eta0").GetValue(), eta0_original + 0.1)

    def test_xml_input(self):
        """Ensure XMLInput restores parameters previously serialized with xml()."""
        xml_state = self.profile.xml()
        eta0_original = self.profile.GetPar("Eta0").GetValue()

        self.profile.GetPar("Eta0").SetValue(eta0_original + 0.3)
        self.assertNotAlmostEqual(self.profile.GetPar("Eta0").GetValue(), eta0_original)

        RefinableObj.XMLInput(self.profile, xml_state)
        self.assertAlmostEqual(self.profile.GetPar("Eta0").GetValue(), eta0_original)

    def test_xml_output(self):
        """Ensure XMLOutput emits parameter tags and the expected root element."""
        xml_state = self.profile.xml()

        self.assertIn("<ReflectionProfile", xml_state)
        for par_name in ("U", "V", "W", "Eta0"):
            self.assertIn(f'Name="{par_name}"', xml_state)

        import io

        buf = io.StringIO()
        RefinableObj.XMLOutput(self.profile, buf, 0)
        xml_from_stream = buf.getvalue()
        self.assertTrue(xml_from_stream.startswith("<ReflectionProfile"))


if __name__ == "__main__":
    unittest.main()
