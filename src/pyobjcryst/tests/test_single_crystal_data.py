#!/usr/bin/env python

"""Tests for diffractiondatasinglecrystal module."""

import unittest
import gc
import numpy as np

from pyobjcryst.crystal import CreateCrystalFromCIF, Crystal
from pyobjcryst.diffractiondatasinglecrystal import *
from pyobjcryst.tests.pyobjcrysttestutils import loadcifdata, datafile


class test_single_crystal_data(unittest.TestCase):

    def test_create(self):
        """Test creating a DiffractionDataSingleCrystal object"""
        c = Crystal(3.52, 3.52, 3.52, "225")
        d = DiffractionDataSingleCrystal(c)

    def test_create_set_hkliobs(self):
        """test SetHklIobs, SetIobs and SetSigma"""
        c = Crystal(3.1, 3.2, 3.3, "Pmmm")
        d = DiffractionDataSingleCrystal(c)
        n0 = 5
        nb = n0 ** 3
        r = np.arange(1, nb + 1, dtype=np.float64)
        h = r % n0
        l = r // n0 ** 2
        k = (r - l * n0 ** 2) // n0
        iobs = np.random.uniform(0, 100, nb)
        sigma = np.sqrt(iobs)

        d.SetHklIobs(h, k, l, iobs, sigma)

        # SetHklIobs sorts reflecions by sin(theta)/lambda, so do the same for comparison
        s = np.sqrt(h ** 2 / 3.1 ** 2 + k ** 2 / 3.2 ** 2 + l ** 2 / 3.3 ** 2) / 2
        idx = np.argsort(s)

        iobs = np.take(iobs, idx)
        sigma = np.take(sigma, idx)
        h = np.take(h, idx)
        k = np.take(k, idx)
        l = np.take(l, idx)
        self.assertTrue(np.all(iobs == d.GetIobs()))
        self.assertTrue(np.all(sigma == d.GetSigma()))
        self.assertTrue(np.all(h == d.GetH()))
        self.assertTrue(np.all(k == d.GetK()))
        self.assertTrue(np.all(l == d.GetL()))

        # Set Iobs and sigma individually
        iobs = np.random.uniform(0, 100, nb)
        d.SetIobs(iobs)
        self.assertTrue(np.all(iobs == d.GetIobs()))

        sigma = np.random.uniform(0, 10, nb)
        d.SetSigma(sigma)
        self.assertTrue(np.all(sigma == d.GetSigma()))

if __name__ == "__main__":
    unittest.main()
