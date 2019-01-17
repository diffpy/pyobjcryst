#!/usr/bin/env python

"""Unit tests for pyobjcryst.powderpattern
"""


import unittest
import numpy as np

from pyobjcryst import ObjCrystException
from pyobjcryst.powderpattern import PowderPattern

# ----------------------------------------------------------------------------

class TestRoutines(unittest.TestCase):

    pass
    #def test_CreatePowderPatternFromCIF(self):  assert False

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

    #def test_AddPowderPatternBackground(self):  assert False
    #def test_AddPowderPatternDiffraction(self):  assert False
    #def test_FitScaleFactorForIntegratedR(self):  assert False
    #def test_FitScaleFactorForIntegratedRw(self):  assert False
    #def test_FitScaleFactorForR(self):  assert False
    #def test_FitScaleFactorForRw(self):  assert False
    #def test_GetMaxSinThetaOvLambda(self):  assert False
    #def test_GetNbPowderPatternComponent(self):  assert False
    #def test_GetPowderPatternCalc(self):  assert False
    #def test_GetPowderPatternComponent(self):  assert False

    def test_GetPowderPatternObs(self):
        self.assertTrue(np.array_equal([], self.pp.GetPowderPatternObs()))
        return

    def test_GetPowderPatternX(self):
        self.assertTrue(np.array_equal([], self.pp.GetPowderPatternX()))
        return

    #def test_GetScaleFactor(self):  assert False
    #def test_ImportPowderPattern2ThetaObs(self):  assert False
    #def test_ImportPowderPattern2ThetaObsSigma(self):  assert False
    #def test_ImportPowderPatternFullprof(self):  assert False
    #def test_ImportPowderPatternFullprof4(self):  assert False
    #def test_ImportPowderPatternGSAS(self):  assert False
    #def test_ImportPowderPatternILL_D1A5(self):  assert False
    #def test_ImportPowderPatternMultiDetectorLLBG42(self):  assert False
    #def test_ImportPowderPatternPSI_DMC(self):  assert False
    #def test_ImportPowderPatternSietronicsCPI(self):  assert False
    #def test_ImportPowderPatternTOF_ISIS_XYSigma(self):  assert False
    #def test_ImportPowderPatternXdd(self):  assert False
    #def test_Prepare(self):  assert False
    #def test_SetEnergy(self):  assert False
    #def test_SetMaxSinThetaOvLambda(self):  assert False

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

    #def test_SetScaleFactor(self):  assert False
    #def test_SetWavelength(self):  assert False

# End of class TestPowderPattern

# ----------------------------------------------------------------------------

class TestPowderPatternComponent(unittest.TestCase):

    pass
    #def test___init__(self):  assert False
    #def test_GetParentPowderPattern(self):  assert False

# End of class TestPowderPatternComponent

# ----------------------------------------------------------------------------

class TestPowderPatternBackground(unittest.TestCase):

    pass
    #def test___init__(self):  assert False
    #def test_FixParametersBeyondMaxresolution(self):  assert False
    #def test_GetPowderPatternCalc(self):  assert False
    #def test_ImportUserBackground(self):  assert False
    #def test_OptimizeBayesianBackground(self):  assert False
    #def test_SetInterpPoints(self):  assert False

# End of class TestPowderPatternBackground

# ----------------------------------------------------------------------------

class TestPowderPatternDiffraction(unittest.TestCase):

    pass
    #def test___init__(self):  assert False
    #def test_ExtractLeBail(self):  assert False
    #def test_GetExtractionMode(self):  assert False
    #def test_GetNbReflBelowMaxSinThetaOvLambda(self):  assert False
    #def test_GetPowderPatternCalc(self):  assert False
    #def test_GetProfile(self):  assert False
    #def test_SetCrystal(self):  assert False
    #def test_SetExtractionMode(self):  assert False
    #def test_SetReflectionProfilePar(self):  assert False

# End of class TestPowderPatternDiffraction

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
