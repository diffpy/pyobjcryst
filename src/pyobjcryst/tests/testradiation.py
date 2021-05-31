#!/usr/bin/env python
##############################################################################
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Tests for Radiation module."""

import unittest

from pyobjcryst.radiation import Radiation, RadiationType, WavelengthType
from pyobjcryst.diffractiondatasinglecrystal import DiffractionDataSingleCrystal
from pyobjcryst.powderpattern import PowderPattern


class TestRadiation(unittest.TestCase):

    def testRadiation(self):
        """Test Radiation creation"""
        r = Radiation()
        return

    def testWavelength(self):
        """Test setting & reading wavelength"""
        r = Radiation()
        r.SetWavelength(1.24)
        self.assertAlmostEqual(r.GetWavelength(), 1.24, places=3)
        return

    def testType(self):
        """Test setting & reading X-ray Tube wavelength"""
        r = Radiation()
        r.SetWavelengthType(WavelengthType.WAVELENGTH_ALPHA12)
        self.assertEqual(r.GetWavelengthType(), WavelengthType.WAVELENGTH_ALPHA12)
        r.SetRadiationType(RadiationType.RAD_NEUTRON)
        self.assertEqual(r.GetRadiationType(), RadiationType.RAD_NEUTRON)
        r.SetWavelength("Cu")
        self.assertAlmostEqual(r.GetWavelength(), 1.5418, places=4)
        self.assertEqual(r.GetWavelengthType(), WavelengthType.WAVELENGTH_ALPHA12)
        return


if __name__ == "__main__":
    unittest.main()
