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


class TestReflectionProfile(unittest.TestCase):
    """Tests for ReflectionProfile methods."""

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, loadcifdata):
        self.loadcifdata = loadcifdata

    def setUp(self):
        """Set up a ReflectionProfile instance for testing."""
        x = np.linspace(0, 40, 8001)
        c = self.loadcifdata("paracetamol.cif")

        self.pp = PowderPattern()
        self.pp.SetWavelength(0.7)
        self.pp.SetPowderPatternX(np.deg2rad(x))
        self.pp.SetPowderPatternObs(np.ones_like(x))

        self.ppd = self.pp.AddPowderPatternDiffraction(c)

        self.profile = self.ppd.GetProfile()

    def test_get_computed_profile(self):
        assert True

    def test_get_profile_width(self):
        assert True

    def test_create_copy(self):
        assert True

    def test_xml_input(self):
        assert True

    def test_xml_output(self):
        assert True


if __name__ == "__main__":
    unittest.main()
