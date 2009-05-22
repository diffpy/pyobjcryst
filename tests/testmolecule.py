#!/usr/bin/env python
"""Tests for molecule module."""

from pyobjcryst import *
from utils import *
import unittest

from utils import *
from numpy import pi

numplaces = 6


class TestMolecule(unittest.TestCase):

    def setUp(self):
        self.c = makeC60()
        self.m = self.c.GetScatterer("c60")
        return

    def tearDown(self):
        del self.c
        del self.m
        return

    def testAtoms(self):
        """Make sure the atoms are there.

        This tests AddAtom by association.
        This tests GetAtom.
        
        """
        self.assertTrue(60, self.m.GetNbAtoms())
        for i in range(60):
            a1 = self.m.GetAtom(i)
            self.assertEqual(a1.GetName(), "C%i"%i)
        return

    def testRemoveAtom(self):
        """Test RemoveAtom."""
        # RemoveAtom method.
        a = self.m.GetAtom(0)

        self.m.RemoveAtom(a)

        self.assertTrue(59, self.m.GetNbAtoms())

        # Check to see if a is in our list
        for i in xrange(59):
            self.assertNotEqual(a.GetName(), self.m.GetAtom(i))

        # What happens if we try to remove an atom that is not in the molecule?
        # First, try the same atom again. This will throw a cctbx error.
        # FIXME - change this once exceptions are wrapped
        self.assertRaises(RuntimeError, self.m.RemoveAtom, a)

        ## Try to remove an atom from another molecule
        c = makeC60()
        m = c.GetScatterer("c60")
        self.assertRaises(RuntimeError, self.m.RemoveAtom, m.GetAtom(1))

        # Remove all the atoms.
        for i in xrange(self.m.GetNbAtoms()):
            self.m.RemoveAtom(0)

        self.assertEquals(0, self.m.GetNbAtoms())

        return

    def testFindBond(self):
        """Test the FindBond method."""

        a1 = self.m.GetAtom(0)
        a2 = self.m.GetAtom(1)
        a3 = self.m.GetAtom(2)
        a4 = self.m.GetAtom(3)

        # Check for a bond that doesn't exist
        bond = self.m.FindBond(a1, a2)
        self.assertTrue(bond is None)

        # Make a bond and try to find it
        self.m.AddBond(a1, a2, 5, 0, 0)
        bond1 = self.m.FindBond(a1, a2)
        bond2 = self.m.FindBond(a1, a2)
        bond3 = self.m.FindBond(a1, a3)
        self.assertTrue(bond1 is not None)
        # Cannot expect the python objects to be the same, but they should point
        # to the same internal object
        self.assertEqual(bond1.GetName(), bond2.GetName())

        # Try some bad bonds
        self.assertTrue(bond3 is None)
        return
    
    def testFindBondAngle(self):
        """Test the FindBondAngle method."""
        a1 = self.m.GetAtom(0)
        a2 = self.m.GetAtom(1)
        a3 = self.m.GetAtom(2)
        a4 = self.m.GetAtom(3)

        # Check for a bondangle angle that doesn't exist
        bondangle = self.m.FindBondAngle(a1, a2, a3)
        self.assertTrue(bondangle is None)

        # Make a bondangle and try to find it
        self.m.AddBondAngle(a2, a1, a3, 90, 0, 0)
        bondangle1 = self.m.FindBondAngle(a2, a1, a3)
        bondangle2 = self.m.FindBondAngle(a2, a1, a3)
        bondangle3 = self.m.FindBondAngle(a1, a2, a4)
        self.assertTrue(bondangle1 is not None)
        self.assertEqual(bondangle1.GetName(), bondangle2.GetName())

        self.assertTrue(bondangle3 is None)
        return
    
    def testFindDihedralAngle(self):
        """Test the FindDihedralAngle method."""
        a1 = self.m.GetAtom(0)
        a2 = self.m.GetAtom(1)
        a3 = self.m.GetAtom(2)
        a4 = self.m.GetAtom(3)

        # Check for a dihedralangle angle that doesn't exist
        dihedralangle = self.m.FindDihedralAngle(a1, a2, a3, a4)
        self.assertTrue(dihedralangle is None)

        # Make a dihedralangle and try to find it
        self.m.AddDihedralAngle(a1, a2, a3, a4, 90, 0, 0)
        dihedralangle1 = self.m.FindDihedralAngle(a1, a2, a3, a4)
        dihedralangle2 = self.m.FindDihedralAngle(a1, a2, a3, a4)
        self.assertTrue(dihedralangle1 is not None)
        self.assertEqual(dihedralangle1.GetName(), dihedralangle2.GetName())
        return

# Test how changing a name to one that is already taken messes things up.

class TestMolAtom(unittest.TestCase):

    def setUp(self):
        c = makeC60()
        self.m = c.GetScatterer("c60")
        self.a = self.m.GetAtom("C0")
        return

    def tearDown(self):
        del self.m
        del self.a

    def testAccessors(self):

        a = self.a

        # Test name Get/Set
        self.assertTrue(a.GetName(), "C0")
        a.SetName("test")
        self.assertTrue(a.GetName(), "test")

        # Test xyz & occ Get/Set
        self.assertAlmostEquals(3.451266498, a.x, numplaces)
        self.assertAlmostEquals(0.685, a.y, numplaces)
        self.assertAlmostEquals(0, a.z, numplaces)
        self.assertAlmostEquals(1.0, a.occ, numplaces)

        a.x = 3.40
        a.y = 0.68
        a.z = 0.1
        a.occ = 1.02
        
        self.assertAlmostEquals(3.40, a.x, numplaces)
        self.assertAlmostEquals(0.68, a.y, numplaces)
        self.assertAlmostEquals(0.1, a.z, numplaces)
        self.assertAlmostEquals(1.02, a.occ, numplaces)
        
        # Test GetMolecule. We can't expect the python object to be the same as
        # our molecule above. However, we can verify that it points to the same
        # object.
        m = a.GetMolecule()
        self.assertEquals(m.GetName(), self.m.GetName())
        # Change something with the molecule, and check to see if it appears in
        # self.m
        m.GetAtom("C1").occ = 0.1
        self.assertAlmostEquals(0.1, self.m.GetAtom("C1").occ, numplaces)

        # Test IsDummy
        self.assertFalse(a.IsDummy())

        # Test GetScatteringPower
        sp = a.GetScatteringPower()
        self.assertEquals("ScatteringPowerAtom", sp.GetClassName())
        self.assertEquals("C", sp.GetName())

        # Test Ring Get/Set
        self.assertFalse(a.IsInRing())
        a.SetIsInRing(True)
        self.assertTrue(a.IsInRing())
        a.SetIsInRing(False)
        self.assertFalse(a.IsInRing())

        return





if __name__ == "__main__":
    unittest.main()


