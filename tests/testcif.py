#!/usr/bin/env python
"""Tests for crystal module."""

import unittest

from pyobjcryst.crystal import CreateCrystalFromCIF
from numpy import pi

import os
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')


class TestCif(unittest.TestCase):

    def testCif(self):
        """Make sure we can read all cif files."""

        import glob
        for fname in glob.glob("%s/*.cif"%testdata_dir):
        #for fname in glob.glob("%s/TiO2_rutile.cif"%testdata_dir):
            c = CreateCrystalFromCIF(file(fname))
            self.assertTrue(c is not None)

            if fname.endswith("caffeine.cif"):
                self.assertEquals(24, c.GetNbScatterer())
                self.assertAlmostEquals(14.9372, c.a, 6)
                self.assertAlmostEquals(14.9372, c.b, 6)
                self.assertAlmostEquals(6.8980, c.c, 6)
                self.assertAlmostEquals(pi/2, c.alpha, 6)
                self.assertAlmostEquals(pi/2, c.beta, 6)
                self.assertAlmostEquals(2*pi/3, c.gamma, 6)

                cifdata = """\
                        C5 -0.06613 -0.06314 0.09562 0.00000 Uiso 1.00000 C
                        C4 0.02779 -0.05534 0.10000 0.00000 Uiso 1.00000 C
                        N3 0.11676 0.04116 0.08226 0.00000 Uiso 1.00000 N
                        C2 0.10866 0.12960 0.06000 0.00000 Uiso 1.00000 C
                        O11 0.18634 0.21385 0.04451 0.00000 Uiso 1.00000 O
                        N1 0.01159 0.12154 0.05547 0.00000 Uiso 1.00000 N
                        C6 -0.07738 0.02504 0.07321 0.00000 Uiso 1.00000 C
                        O13 -0.16212 0.01800 0.06926 0.00000 Uiso 1.00000 O
                        N7 -0.13793 -0.16353 0.11547 0.00000 Uiso 1.00000 N
                        N9 0.01389 -0.15092 0.12255 0.00000 Uiso 1.00000 N
                        C8 -0.08863 -0.21778 0.13210 0.00000 Uiso 1.00000 C
                        C14 -0.25110 -0.20663 0.11847 0.00000 Uiso 1.00000 C
                        C12 0.21968 0.04971 0.08706 0.00000 Uiso 1.00000 C
                        C10 0.00300 0.21530 0.03186 0.00000 Uiso 1.00000 C
                        H8 -0.12146 -0.29285 0.14834 0.00000 Uiso 1.00000 H
                        H14a -0.27317 -0.19028 -0.00382 0.00000 Uiso 1.00000 H
                        H14b -0.28567 -0.28182 0.13462 0.00000 Uiso 1.00000 H
                        H14c -0.26951 -0.17639 0.22660 0.00000 Uiso 1.00000 H
                        H12a 0.22513 0.00926 -0.01943 0.00000 Uiso 1.00000 H
                        H12b 0.27338 0.12237 0.07281 0.00000 Uiso 1.00000 H
                        H12c 0.22878 0.02315 0.21098 0.00000 Uiso 1.00000 H
                        H10a 0.03488 0.24909 -0.09109 0.00000 Uiso 1.00000 H
                        H10b -0.07008 0.19602 0.03170 0.00000 Uiso 1.00000 H
                        H10c 0.03791 0.26293 0.13930 0.00000 Uiso 1.00000 H
                        """

                import cStringIO
                lines = cStringIO.StringIO(cifdata)

                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line: continue
                    name, x, y, z, U, junk, occ, element = line.split()
                    s = c.GetScatt(i)
                    self.assertEquals(name, s.GetName())
                    self.assertAlmostEquals(float(x), s.X, 6)
                    self.assertAlmostEquals(float(y), s.Y, 6)
                    self.assertAlmostEquals(float(z), s.Z, 6)
                    self.assertAlmostEquals(float(occ), s.Occupancy, 6)

            # Check ADPs
            if fname.endswith("CaTiO3.cif"):
                s = c.GetScatt(0)
                name = s.GetName()
                sp = c.GetScatteringPower(name)
                self.assertEquals(name, "Ca1")
                self.assertFalse(sp.IsIsotropic())
                utob = 8 * pi**2
                self.assertAlmostEquals(utob * 0.0077, sp.B11)
                self.assertAlmostEquals(utob * 0.0079, sp.B22)
                self.assertAlmostEquals(utob * 0.0077, sp.B33)
                self.assertAlmostEquals(-utob * 0.0013, sp.B12)
                self.assertAlmostEquals(0, sp.B13)
                self.assertAlmostEquals(0, sp.B23)
                self.assertAlmostEquals(-0.00676, s.X, 5)
                self.assertAlmostEquals(0.03602, s.Y, 5)
                self.assertAlmostEquals(0.25, s.Z, 2)
                self.assertAlmostEquals(1, s.Occupancy)

            if fname.endswith("TiO2_rutile.cif"):
                s = c.GetScatt(0)
                name = s.GetName()
                sp = c.GetScatteringPower(name)
                self.assertEquals(name, "Ti")
                self.assertTrue(sp.IsIsotropic())
                utob = 8 * pi**2
                self.assertAlmostEquals(utob * 0.00532, sp.Biso, 5)
        return


    def testBadCif(self):
        """Make sure we can read all cif files."""
        from pyobjcryst import ObjCrystException

        fname = "%s/ni.stru"%testdata_dir
        infile = file(fname)
        self.assertRaises(ObjCrystException, CreateCrystalFromCIF, infile)
        infile.close()
        return



if __name__ == "__main__":
    unittest.main()

