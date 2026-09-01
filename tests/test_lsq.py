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

import pytest

from pyobjcryst import ObjCrystException, refinableobj
from pyobjcryst.diffractiondatasinglecrystal import (
    DiffractionDataSingleCrystal,
)
from pyobjcryst.lsq import LSQ


class TestGlobalOptim(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, loadcifdata):
        self.loadcifdata = loadcifdata

    def setUp(self):
        self.c = self.loadcifdata("caffeine.cif")
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
        lsq.GetLSQObs()
        lsq.GetLSQCalc()
        lsq.ChiSquare()

    def test_lsq_get_refined_obj(self):
        """Check Creating a basic LSQ object & get obs&calc arrays."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        # print(lsq.GetCompiledRefinedObj())

    def test_lsq_compiled_refined_obj_erase_all_param_set(self):
        """Check parameter sets can be erased on compiled refined
        objects."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        refobj = lsq.GetCompiledRefinedObj()
        save = refobj.CreateParamSet("save")
        refobj.EraseAllParamSet()
        self.assertRaises(ObjCrystException, refobj.SaveParamSet, save)

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

    def test_lsq_compiled_refined_obj_erase_all_param_set(self):
        """Check parameter sets can be erased on compiled refined objects."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        refobj = lsq.GetCompiledRefinedObj()
        save = refobj.CreateParamSet("save")
        refobj.EraseAllParamSet()
        self.assertRaises(ObjCrystException, refobj.SaveParamSet, save)

    def test_lsq_get_variance_covariance_map(self):
        """GetVarianceCovarianceMap returns a dict keyed by (name, name) tuples."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        lsq.SetParIsFixed(refinableobj.refpartype_objcryst, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata_scale, False)
        lsq.Refine(1, silent=True)
        cov = lsq.GetVarianceCovarianceMap()
        self.assertIsInstance(cov, dict)
        for key in cov:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
            self.assertIsInstance(key[0], str)
            self.assertIsInstance(key[1], str)

    def test_lsq_get_rw_history(self):
        """GetRwHistory returns a list of floats after Refine()."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        lsq.SetParIsFixed(refinableobj.refpartype_objcryst, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata_scale, False)
        history_before = lsq.GetRwHistory()
        self.assertEqual(list(history_before), [])
        lsq.Refine(3, silent=True)
        history = lsq.GetRwHistory()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        for v in history:
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.0)

    def test_lsq_refine_minrwpvar(self):
        """minRwpVar argument is accepted by Refine without error."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList()
        lsq.SetParIsFixed(refinableobj.refpartype_objcryst, True)
        lsq.SetParIsFixed(refinableobj.refpartype_scattdata_scale, False)
        lsq.Refine(5, silent=True, minRwpVar=0.01)

    def test_lsq_prepare_ref_par_list_verbose(self):
        """PrepareRefParList accepts a verbose keyword argument."""
        lsq = LSQ()
        lsq.SetRefinedObj(self.d, 0, True, True)
        lsq.PrepareRefParList(verbose=False)
        lsq.PrepareRefParList(copy_param=False, verbose=True)


if __name__ == "__main__":
    unittest.main()

