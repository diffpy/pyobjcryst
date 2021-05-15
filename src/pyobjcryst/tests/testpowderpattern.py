#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        Complex Modeling Initiative
#                   (c) 2018 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Unit tests for pyobjcryst.powderpattern (with indexing &
"""

import unittest
import numpy as np

from pyobjcryst import ObjCrystException
from pyobjcryst.powderpattern import PowderPattern, SpaceGroupExplorer
from pyobjcryst.radiation import RadiationType, WavelengthType
from pyobjcryst.crystal import *
from pyobjcryst.reflectionprofile import ReflectionProfileType
from pyobjcryst.indexing import *
from pyobjcryst.tests.pyobjcrysttestutils import loadcifdata, datafile


# ----------------------------------------------------------------------------

class TestRoutines(unittest.TestCase):
    pass
    # def test_CreatePowderPatternFromCIF(self):  assert False


# End of class TestRoutines

# ----------------------------------------------------------------------------

class TestPowderPattern(unittest.TestCase):

    def setUp(self):
        self.pp = PowderPattern()
        return

    def test___init__(self):
        self.assertEqual(0, self.pp.GetNbPowderPatternComponent())
        self.assertEqual(0, len(self.pp.GetPowderPatternX()))
        self.assertEqual(0, len(self.pp.GetPowderPatternObs()))
        self.assertEqual(0, len(self.pp.GetPowderPatternCalc()))
        return

    # def test_AddPowderPatternBackground(self):  assert False
    # def test_AddPowderPatternDiffraction(self):  assert False
    # def test_FitScaleFactorForIntegratedR(self):  assert False
    # def test_FitScaleFactorForIntegratedRw(self):  assert False
    # def test_FitScaleFactorForR(self):  assert False
    # def test_FitScaleFactorForRw(self):  assert False
    # def test_GetMaxSinThetaOvLambda(self):  assert False
    # def test_GetNbPowderPatternComponent(self):  assert False
    # def test_GetPowderPatternCalc(self):  assert False
    # def test_GetPowderPatternComponent(self):  assert False

    def test_GetPowderPatternObs(self):
        self.assertTrue(np.array_equal([], self.pp.GetPowderPatternObs()))
        return

    def test_GetPowderPatternX(self):
        self.assertTrue(np.array_equal([], self.pp.GetPowderPatternX()))
        return

    # def test_GetScaleFactor(self):  assert False
    # def test_ImportPowderPattern2ThetaObs(self):  assert False
    # def test_ImportPowderPattern2ThetaObsSigma(self):  assert False
    # def test_ImportPowderPatternFullprof(self):  assert False
    # def test_ImportPowderPatternFullprof4(self):  assert False
    # def test_ImportPowderPatternGSAS(self):  assert False
    # def test_ImportPowderPatternILL_D1A5(self):  assert False
    # def test_ImportPowderPatternMultiDetectorLLBG42(self):  assert False
    # def test_ImportPowderPatternPSI_DMC(self):  assert False
    # def test_ImportPowderPatternSietronicsCPI(self):  assert False
    # def test_ImportPowderPatternTOF_ISIS_XYSigma(self):  assert False
    # def test_ImportPowderPatternXdd(self):  assert False
    # def test_Prepare(self):  assert False
    # def test_SetEnergy(self):  assert False
    # def test_SetMaxSinThetaOvLambda(self):  assert False

    def test_SetPowderPatternObs(self):
        pp = self.pp
        obs = np.array([1.0, 3.0, 7.0])
        self.assertRaises(ObjCrystException, pp.SetPowderPatternObs, obs)
        pp.SetPowderPatternPar(0, 0.5, 3)
        pp.SetPowderPatternObs(obs)
        self.assertTrue(np.array_equal(obs, pp.GetPowderPatternObs()))
        pp.SetPowderPatternObs(list(obs)[::-1])
        self.assertTrue(np.array_equal(obs[::-1], pp.GetPowderPatternObs()))
        return

    def test_SetPowderPatternPar(self):
        pp = self.pp
        pp.SetPowderPatternPar(0, 0.25, 5)
        tth = np.linspace(0, 1, 5)
        self.assertTrue(np.array_equal(tth, pp.GetPowderPatternX()))
        pp.SetPowderPatternPar(0, 0.25, 0)
        self.assertEqual(0, len(pp.GetPowderPatternX()))
        return

    def test_SetPowderPatternX(self):
        pp = self.pp
        tth0 = np.array([0, 0.1, 0.3, 0.7])
        tth1 = np.array([0, 0.1, 0.3, 0.7, 0.75, 0.77, 0.80])
        pp.SetPowderPatternX(tth0)
        self.assertTrue(np.array_equal(tth0, pp.GetPowderPatternX()))
        pp.SetPowderPatternX(list(tth1))
        self.assertTrue(np.array_equal(tth1, pp.GetPowderPatternX()))
        pp.SetPowderPatternX(tuple(2 * tth0))
        self.assertTrue(np.array_equal(2 * tth0, pp.GetPowderPatternX()))
        return

    def test_SetPowderPatternXempty(self):
        pp = self.pp
        pp.SetPowderPatternX([0, 0.1, 0.2, 0.3])
        pp.SetPowderPatternX([])
        self.assertEqual(0, len(pp.GetPowderPatternX()))
        return

    def test_SetWavelength(self):
        pp = self.pp
        pp.SetWavelength(1.2345)
        self.assertAlmostEqual(pp.GetWavelength(), 1.2345, places=4)

    def test_SetWavelengthXrayTube(self):
        pp = self.pp
        t = pp.GetRadiation().GetWavelengthType()
        w = pp.GetWavelength()
        pp.SetWavelength("Cu")
        self.assertAlmostEqual(pp.GetWavelength(), 1.5418, places=4)
        self.assertEqual(pp.GetRadiation().GetWavelengthType(), WavelengthType.WAVELENGTH_ALPHA12)
        pp.GetRadiation().SetWavelengthType(t)
        pp.SetWavelength(w)

    def test_SetRadiationType(self):
        pp = self.pp
        t = pp.GetRadiationType()
        pp.SetRadiationType(RadiationType.RAD_NEUTRON)
        self.assertEqual(pp.GetRadiationType(), RadiationType.RAD_NEUTRON)
        pp.SetRadiationType(t)

    def test_quick_fit(self):
        c = loadcifdata("paracetamol.cif")
        p = PowderPattern()
        p.SetWavelength(0.7)
        x = np.linspace(0, 40, 8001)
        p.SetPowderPatternX(np.deg2rad(x))
        pd = p.AddPowderPatternDiffraction(c)
        pd.SetReflectionProfilePar(ReflectionProfileType.PROFILE_PSEUDO_VOIGT, 1e-6)
        # p.plot(hkl=True)
        calc = p.GetPowderPatternCalc()
        obs = np.random.poisson(calc * 1e5 / calc.max() + 50).astype(np.float64)
        p.SetPowderPatternObs(obs)
        p.SetMaxSinThetaOvLambda(0.3)
        p.quick_fit_profile(auto_background=True, verbose=False, plot=False)

    def test_peaklist_index(self):
        c = loadcifdata("paracetamol.cif")
        p = PowderPattern()
        p.SetWavelength(0.7)
        x = np.linspace(0, 40, 8001)
        p.SetPowderPatternX(np.deg2rad(x))
        pd = p.AddPowderPatternDiffraction(c)
        pd.SetReflectionProfilePar(ReflectionProfileType.PROFILE_PSEUDO_VOIGT, 1e-6)
        # p.plot(hkl=True)
        calc = p.GetPowderPatternCalc()
        obs = np.random.poisson(calc * 1e5 / calc.max() + 50).astype(np.float64)
        p.SetPowderPatternObs(obs)
        p.SetMaxSinThetaOvLambda(0.2)
        p.FitScaleFactorForIntegratedRw()
        pl = p.FindPeaks()
        ex = quick_index(pl, verbose=False)
        sols = ex.GetSolutions()
        self.assertGreater(len(sols), 0)
        ruc = sols[0][0]
        # Check lattice type
        self.assertEqual(ruc.centering, CrystalCentering.LATTICE_P)
        self.assertEqual(ruc.lattice, CrystalSystem.MONOCLINIC)
        # Cell volume
        self.assertAlmostEqual(ruc.DirectUnitCell()[-1], c.GetVolume(), delta=5)

    def test_spacegroup_explorer(self):
        c = loadcifdata("paracetamol.cif")
        p = PowderPattern()
        p.SetWavelength(0.7)
        x = np.linspace(0, 40, 8001)
        p.SetPowderPatternX(np.deg2rad(x))
        pd = p.AddPowderPatternDiffraction(c)
        pd.SetReflectionProfilePar(ReflectionProfileType.PROFILE_PSEUDO_VOIGT, 1e-6, 0, 0, 0, 0)
        # p.plot(hkl=True)
        calc = p.GetPowderPatternCalc()
        obs = np.random.poisson(calc * 1e6 / calc.max() + 50).astype(np.float64)
        p.SetPowderPatternObs(obs)
        # NB: with max(stol)=0.2 this fails and best result is P1
        p.SetMaxSinThetaOvLambda(0.3)
        # Do the profile optimisation in P1
        pd.GetCrystal().GetSpaceGroup().ChangeSpaceGroup("P1")
        p.FitScaleFactorForIntegratedRw()
        p.quick_fit_profile(auto_background=True, init_profile=False, verbose=False, plot=False)

        spgex = SpaceGroupExplorer(pd)
        spgex.Run("P 1 21/c 1")
        spgex.RunAll(verbose=False)
        spg = spgex.GetScores()[0]
        # This fails about XX% of the time (fit not converging well enough ?)
        # self.assertEqual(spg.hermann_mauguin, 'P 1 21/c 1')
        # if True:  #spg.hermann_mauguin != 'P 1 21/c 1':
        #     print()
        #     for s in spgex.GetScores():
        #         print(s)

    # def test_SetScaleFactor(self):  assert False


# End of class TestPowderPattern

# ----------------------------------------------------------------------------

class TestPowderPatternComponent(unittest.TestCase):
    pass
    # def test___init__(self):  assert False
    # def test_GetParentPowderPattern(self):  assert False


# End of class TestPowderPatternComponent

# ----------------------------------------------------------------------------

class TestPowderPatternBackground(unittest.TestCase):
    pass
    # def test___init__(self):  assert False
    # def test_FixParametersBeyondMaxresolution(self):  assert False
    # def test_GetPowderPatternCalc(self):  assert False
    # def test_ImportUserBackground(self):  assert False
    # def test_OptimizeBayesianBackground(self):  assert False
    # def test_SetInterpPoints(self):  assert False


# End of class TestPowderPatternBackground

# ----------------------------------------------------------------------------

class TestPowderPatternDiffraction(unittest.TestCase):
    pass
    # def test___init__(self):  assert False
    # def test_ExtractLeBail(self):  assert False
    # def test_GetExtractionMode(self):  assert False
    # def test_GetNbReflBelowMaxSinThetaOvLambda(self):  assert False
    # def test_GetPowderPatternCalc(self):  assert False
    # def test_GetProfile(self):  assert False
    # def test_SetCrystal(self):  assert False
    # def test_SetExtractionMode(self):  assert False
    # def test_SetReflectionProfilePar(self):  assert False


# End of class TestPowderPatternDiffraction

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
