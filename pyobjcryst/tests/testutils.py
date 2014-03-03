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

import os
import unittest

from pyobjcryst.crystal import CreateCrystalFromCIF
from pyobjcryst.utils import putAtomsInMolecule


class TestPutAtomsInMolecule(unittest.TestCase):

    def _testPutAtomsInMolecule(self):
        """Make sure this utility method is correct."""

        from math import floor
        f = lambda v: v - floor(v)
        import glob
        from pyobjcryst.tests.pyobjcrysttestutils import datafile
        pat = os.path.join(datafile(''), '*.cif')

        for fname in glob.glob(pat):
            print fname

            c = CreateCrystalFromCIF(file(fname))

            from diffpy.Structure import Structure
            s = Structure(filename = fname)

            # Get positions from unmodified structure
            pos1 = []
            scl = c.GetScatteringComponentList()
            for s in scl:
                xyz = map(f, [s.X, s.Y, s.Z])
                xyz = c.FractionalToOrthonormalCoords(*xyz)
                pos1.append(xyz)

            # Get positions from molecular structure
            putAtomsInMolecule(c)
            pos2 = []
            scl = c.GetScatteringComponentList()
            for s in scl:
                xyz = map(f, [s.X, s.Y, s.Z])
                xyz = c.FractionalToOrthonormalCoords(*xyz)
                pos2.append(xyz)

            # Now compare positions
            self.assertEqual(len(pos1), len(pos2))

            for p1, p2 in zip(pos1, pos2):
                for i in range(3):
                    self.assertAlmostEqual(p1[i], p2[i])

        return

if __name__ == "__main__":
    unittest.main()

