#!/usr/bin/env python
##############################################################################
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Tests for LSQ module."""

import unittest

from pyobjcryst import refinableobj
from pyobjcryst.diffractiondatasinglecrystal import (
    DiffractionDataSingleCrystal,
)
from pyobjcryst.lsq import LSQ
from pyobjcryst.tests.pyobjcrysttestutils import loadcifdata


class TestGlobalOptim(unittest.TestCase):

    def setUp(self):
        self.c = loadcifdata("caffeine.cif")
        self.d = DiffractionDataSingleCrystal(self.c)
        self.d.GenHKLFullSpace2(0.4, True)
        self.d.SetIobsToIcalc()

    def tearDown(self):
        del self.c
        del self.d

    def test_lsq_create(self):
        """Check Creating a basic LSQ object."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d)

    def test_lsq_get_obs_calc(self):
        """Check Creating a basic LSQ object & get obs&calc arrays."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        junk = lsq.GetLSQObs(), lsq.GetLSQCalc(), lsq.ChiSquare()

    def test_lsq_get_refined_obj(self):
        """Check Creating a basic LSQ object & get obs&calc arrays."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        # print(lsq.GetCompiledRefinedObj())

    def test_lsq_set_pr_fixed(self):
        """Check Creating a basic LSQ object & get obs&calc arrays."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        lsq.SetParIsFixed(refinableobj.refpartype_objcryst, False)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata_scale, False)
        lsq.SetParIsFixed(refinableobj.refpartype_unitcell, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattpow, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata_radiation, True)

    def test_lsq_refine(self):
        lsq = LSQ()
        lsq.SetRefinedObj(self.d)
        # Refine structural parameters
        lsq.SetParIsFixed(refinableobj.refpartype_objcryst, False)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata_scale, False)
        lsq.SetParIsFixed(refinableobj.refpartype_unitcell, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattpow, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata_radiation, True)
        for i in range(5):
            self.c.RandomizeConfiguration()
            lsq.Refine(10, False, True)


if __name__ == "__main__":
    unittest.main()
