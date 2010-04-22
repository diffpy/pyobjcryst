#!/usr/bin/env python
"""Tests for crystal module."""

import unittest

from pyobjcryst.crystal import CreateCrystalFromCIF
from pyobjcryst.utils import putAtomsInMolecule

import os
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

class TestPutAtomsInMolecule(unittest.TestCase):

    def testPutAtomsInMolecule(self):
        """Make sure this utility method is correct."""

        from math import floor
        f = lambda v: v - floor(v)

        import glob
        for fname in glob.glob("%s/*.cif"%testdata_dir):
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

