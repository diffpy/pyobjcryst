#!/usr/bin/env python
##############################################################################
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Tests for MonteCarlo module."""

import unittest

from pyobjcryst import refinableobj
from pyobjcryst.diffractiondatasinglecrystal import (
    DiffractionDataSingleCrystal,
)
from pyobjcryst.globaloptim import (
    AnnealingSchedule,
    GlobalOptimType,
    MonteCarlo,
)
from utils import loadcifdata


class TestGlobalOptim(unittest.TestCase):

    def setUp(self):
        self.c = loadcifdata("caffeine.cif")
        self.d = DiffractionDataSingleCrystal(self.c)
        self.d.GenHKLFullSpace2(0.4, True)
        self.d.SetIobsToIcalc()

    def tearDown(self):
        del self.c
        del self.d

    def test_mc_create(self):
        """Check Creating a basic Monte-Carlo object."""
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)

    def test_mc_name(self):
        """Check Creating a basic Monte-Carlo object."""
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        mc.SetName("caffeine")
        self.assertEqual(mc.GetName(), "caffeine")

    def test_mc_llk(self):
        """Check Creating a basic Monte-Carlo object."""
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        junk = mc.GetLogLikelihood()

    def test_mc_fix_use_pars(self):
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        junk = mc.GetLogLikelihood()
        mc.FixAllPar()
        mc.SetParIsUsed("Scale factor", False)
        mc.SetParIsUsed("Scale factor", True)
        mc.SetParIsFixed("Scale factor", True)
        mc.SetParIsFixed("Scale factor", False)
        mc.SetParIsUsed(refinableobj.refpartype_scatt, False)
        mc.SetParIsUsed(refinableobj.refpartype_scatt, True)
        mc.SetParIsFixed(refinableobj.refpartype_scatt, True)
        mc.SetParIsFixed(refinableobj.refpartype_scatt, False)

    def test_mc_optim(self):
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        mc.RandomizeStartingConfig()
        mc.Optimize(nb_step=1000)

    def test_mc_optim_multi(self):
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        mc.RandomizeStartingConfig()
        mc.MultiRunOptimize(nb_run=2, nb_step=1000)

    def test_mc_sa(self):
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        mc.RandomizeStartingConfig()
        mc.RunSimulatedAnnealing(nb_step=1000)

    def test_mc_pt(self):
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        mc.RandomizeStartingConfig()
        mc.RunParallelTempering(nb_step=1000)

    # TODO: this is experimental and leads to segfault if testcrystal:testDummyAtom() has been run before (?!)
    # def test_mc_lsq(self):
    #     mc = MonteCarlo()
    #     mc.AddRefinableObj(self.c)
    #     mc.AddRefinableObj(self.d)
    #     mc.RandomizeStartingConfig()
    #     mc.InitLSQ()
    #     # print(mc.GetLSQObj().GetCompiledRefinedObj())
    #     mc.RunRandomLSQ(nbCycle=2)

    def test_mc_set_algo(self):
        mc = MonteCarlo()
        mc.AddRefinableObj(self.c)
        mc.AddRefinableObj(self.d)
        mc.RandomizeStartingConfig()
        mc.SetAlgorithmSimulAnnealing(AnnealingSchedule.SMART, 1000.0, 1.0)
        mc.Optimize(nb_step=1000)
        mc.SetAlgorithmParallTempering(AnnealingSchedule.SMART, 1000.0, 1.0)
        mc.Optimize(nb_step=1000)


if __name__ == "__main__":
    unittest.main()
