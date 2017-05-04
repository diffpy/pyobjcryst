#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 The Trustees of Columbia University
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

from pyobjcryst.crystal import CreateCrystalFromCIF
from numpy import pi
from pyobjcryst.tests.pyobjcrysttestutils import loadcifdata, datafile


class TestCif(unittest.TestCase):

    def test_Ag_silver_cif(self):
        '''Check loading of Ag_silver.cif
        '''
        c = loadcifdata('Ag_silver.cif')
        self.assertTrue(c is not None)
        return


    def test_BaTiO3_cif(self):
        '''Check loading of BaTiO3.cif
        '''
        c = loadcifdata('BaTiO3.cif')
        self.assertTrue(c is not None)
        return


    def test_C_graphite_hex_cif(self):
        '''Check loading of C_graphite_hex.cif
        '''
        c = loadcifdata('C_graphite_hex.cif')
        self.assertTrue(c is not None)
        return


    def test_CaF2_fluorite_cif(self):
        '''Check loading of CaF2_fluorite.cif
        '''
        c = loadcifdata('CaF2_fluorite.cif')
        self.assertTrue(c is not None)
        return


    def test_caffeine_cif(self):
        '''Check loading of caffeine.cif and the data inside.
        '''
        c = loadcifdata('caffeine.cif')
        self.assertTrue(c is not None)
        self.assertEqual(24, c.GetNbScatterer())
        self.assertAlmostEqual(14.9372, c.a, 6)
        self.assertAlmostEqual(14.9372, c.b, 6)
        self.assertAlmostEqual(6.8980, c.c, 6)
        self.assertAlmostEqual(pi/2, c.alpha, 6)
        self.assertAlmostEqual(pi/2, c.beta, 6)
        self.assertAlmostEqual(2*pi/3, c.gamma, 6)
        cifdata = """
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
        lines = filter(None,
                       (line.strip() for line in cifdata.split('\n')))
        for i, line in enumerate(lines):
            name, x, y, z, U, junk, occ, element = line.split()
            s = c.GetScatt(i)
            self.assertEqual(name, s.GetName())
            self.assertAlmostEqual(float(x), s.X, 6)
            self.assertAlmostEqual(float(y), s.Y, 6)
            self.assertAlmostEqual(float(z), s.Z, 6)
            self.assertAlmostEqual(float(occ), s.Occupancy, 6)
        return


    def test_CaTiO3_cif(self):
        '''Check loading of CaTiO3.cif and its ADPs.
        '''
        c = loadcifdata('CaTiO3.cif')
        self.assertTrue(c is not None)
        s = c.GetScatt(3)
        name = s.GetName()
        self.assertEqual(name, "O2")
        sp = c.GetScatteringPower(name)
        self.assertFalse(sp.IsIsotropic())
        utob = 8 * pi**2
        self.assertAlmostEqual(utob * 0.0065, sp.B11)
        self.assertAlmostEqual(utob * 0.0060, sp.B22)
        self.assertAlmostEqual(utob * 0.0095, sp.B33)
        self.assertAlmostEqual(utob * 0.0020, sp.B12)
        self.assertAlmostEqual(utob * -0.0008, sp.B13)
        self.assertAlmostEqual(utob * -0.0010, sp.B23)
        self.assertAlmostEqual(0.2897, s.X, 5)
        self.assertAlmostEqual(0.2888, s.Y, 5)
        self.assertAlmostEqual(0.0373, s.Z, 2)
        self.assertAlmostEqual(1, s.Occupancy)
        return


    def test_CdSe_cadmoselite_cif(self):
        '''Check loading of CdSe_cadmoselite.cif
        '''
        c = loadcifdata('CdSe_cadmoselite.cif')
        self.assertTrue(c is not None)
        return


    def test_CeO2_cif(self):
        '''Check loading of CeO2.cif
        '''
        c = loadcifdata('CeO2.cif')
        self.assertTrue(c is not None)
        return


    def test_lidocainementhol_cif(self):
        '''Check loading of lidocainementhol.cif
        '''
        c = loadcifdata('lidocainementhol.cif')
        self.assertTrue(c is not None)
        return


    def test_NaCl_cif(self):
        '''Check loading of NaCl.cif
        '''
        c = loadcifdata('NaCl.cif')
        self.assertTrue(c is not None)
        return


    def test_Ni_cif(self):
        '''Check loading of Ni.cif
        '''
        c = loadcifdata('Ni.cif')
        self.assertTrue(c is not None)
        return


    def test_paracetamol_cif(self):
        '''Check loading of paracetamol.cif
        '''
        c = loadcifdata('paracetamol.cif')
        self.assertTrue(c is not None)
        return


    def test_PbS_galena_cif(self):
        '''Check loading of PbS_galena.cif
        '''
        c = loadcifdata('PbS_galena.cif')
        self.assertTrue(c is not None)
        return


    def test_PbTe_cif(self):
        '''Check loading of PbTe.cif
        '''
        c = loadcifdata('PbTe.cif')
        self.assertTrue(c is not None)
        return


    def test_Si_cif(self):
        '''Check loading of Si.cif
        '''
        c = loadcifdata('Si.cif')
        self.assertTrue(c is not None)
        return


    def test_Si_setting2_cif(self):
        '''Check loading of Si_setting2.cif
        '''
        c = loadcifdata('Si_setting2.cif')
        self.assertTrue(c is not None)
        return


    def test_SrTiO3_tausonite_cif(self):
        '''Check loading of SrTiO3_tausonite.cif
        '''
        c = loadcifdata('SrTiO3_tausonite.cif')
        self.assertTrue(c is not None)
        return


    def test_TiO2_anatase_cif(self):
        '''Check loading of TiO2_anatase.cif
        '''
        c = loadcifdata('TiO2_anatase.cif')
        self.assertTrue(c is not None)
        return


    def test_TiO2_rutile_cif(self):
        '''Check loading of TiO2_rutile.cif and its ADP data
        '''
        c = loadcifdata('TiO2_rutile.cif')
        self.assertTrue(c is not None)
        s = c.GetScatt(0)
        name = s.GetName()
        sp = c.GetScatteringPower(name)
        self.assertEqual(name, "Ti")
        self.assertTrue(sp.IsIsotropic())
        utob = 8 * pi**2
        self.assertAlmostEqual(utob * 0.00532, sp.Biso, 5)
        return


    def test_Zn_zinc_cif(self):
        '''Check loading of Zn_zinc.cif
        '''
        c = loadcifdata('Zn_zinc.cif')
        self.assertTrue(c is not None)
        return


    def test_ZnS_sphalerite_cif(self):
        '''Check loading of ZnS_sphalerite.cif
        '''
        c = loadcifdata('ZnS_sphalerite.cif')
        self.assertTrue(c is not None)
        return


    def test_ZnS_wurtzite_cif(self):
        '''Check loading of ZnS_wurtzite.cif
        '''
        c = loadcifdata('ZnS_wurtzite.cif')
        self.assertTrue(c is not None)
        return


    def testBadCif(self):
        """Make sure we can read all cif files."""
        from pyobjcryst import ObjCrystException
        fname = datafile('ni.stru')
        infile = open(fname, 'rb')
        self.assertRaises(ObjCrystException, CreateCrystalFromCIF, infile)
        infile.close()
        return

# End of class TestCif


if __name__ == "__main__":
    unittest.main()
