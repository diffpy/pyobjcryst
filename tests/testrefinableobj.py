#!/usr/bin/env python
"""Tests for refinableobj module."""

from pyobjcryst import *
import unittest

from utils import makeScatterer, makeCrystal

class TestRestraint(unittest.TestCase):

    def setUp(self):
        #self.testpar = RefinablePar("test", 3.0, 0, 10, self.rpt)
        return

    def testEquality(self):
        """See if we get back what we put in."""
        rpt = RefParType("test")
        res1 = Restraint(rpt)
        rpt2 = res1.GetType()
        self.assertEqual(rpt2, rpt)
        return


class TestRefinablePar(unittest.TestCase):

    def setUp(self):
        self.rpt = RefParType("test")
        self.testpar = RefinablePar("test", 3.0, 0, 10, self.rpt)
        return

    def testToFromPython(self):
        """See if refinable parameters can be created from within python and
        within c++."""
        c = makeCrystal(*makeScatterer())

        # Get a parameter created from c++
        par = c.GetPar("a")
        self.assertAlmostEqual(3.52, par.GetValue())

        # pass a parameter and pass it into c++
        c.AddPar(self.testpar);

        # get it back
        testpar2 = c.GetPar("test")

        self.assertAlmostEqual(self.testpar.GetValue(), testpar2.GetValue())

        testpar2.SetValue(2.17)
        self.assertAlmostEqual(2.17, testpar2.GetValue(), places = 6)
        self.assertAlmostEqual(self.testpar.GetValue(), testpar2.GetValue())
        return

    def testGetType(self):
        """See if we can get the proper RefParType from a RefinablePar."""
        rpt2 = self.testpar.GetType()
        self.assertEqual(rpt2, self.rpt)
        return



if __name__ == "__main__":

    unittest.main()

