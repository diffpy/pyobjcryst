#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Tests for crystal module."""

import unittest
import numpy

from pyobjcryst.tests.pyobjcrysttestutils import loadcifdata
from pyobjcryst.utils import putAtomsInMolecule


class TestPutAtomsInMolecule(unittest.TestCase):

    def test_caffeine(self):
        """Check molecule conversion for caffeine.
        """
        c = loadcifdata('caffeine.cif')
        xyz0 = [(sc.X, sc.Y, sc.Z) for sc in c.GetScatteringComponentList()]
        self.assertEqual(24, c.GetNbScatterer())
        putAtomsInMolecule(c, name='espresso')
        self.assertEqual(1, c.GetNbScatterer())
        mol = c.GetScatterer(0)
        self.assertEqual('espresso', mol.GetName())
        self.assertEqual(24, mol.GetNbAtoms())
        xyz1 = [(sc.X, sc.Y, sc.Z) for sc in c.GetScatteringComponentList()]
        uc0 = numpy.array(xyz0) - numpy.floor(xyz0)
        uc1 = numpy.array(xyz1) - numpy.floor(xyz1)
        self.assertTrue(numpy.allclose(uc0, uc1))
        return

# End of class TestPutAtomsInMolecule

if __name__ == "__main__":
    unittest.main()
