#!/usr/bin/env python
"""Tests for molecule module."""

from pyobjcryst import *
from utils import *
import unittest

from utils import *
from numpy import pi

def makeMolecule():

    c = Crystal(10, 10, 10, "P1")
    m = Molecule(c, "testmolecule")

    c.AddScatterer(m)

    sp = ScatteringPowerAtom("C", "C")
    sp.SetBiso(8*pi*pi*0.003)
    c.AddScatteringPower(sp)

    m.AddAtom(0, 0, 0, sp, "C1")
    m.AddAtom(0, 0, 0.5, sp, "C2")
    m.AddAtom(0, 0.5, 0, sp, "C3")
    m.AddAtom(0.5, 0, 0, sp, "C4")

    return m


class TestMolecule(unittest.TestCase):

    def setUp(self):
        self.m = makeMolecule()

    def tearDown(self):
        del self.m

    def testAtoms(self):
        for i in range(4):
            self.assertEqual(self.m.GetAtom(i).GetName(), "C%i"%(i+1))
        return

    def testGetAtomList(self):
        """Test the GetAtomList method."""
        for i, a in enumerate(self.m.GetAtomList()):
            self.assertEqual(a.GetName(), "C%i"%(i+1))
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
        self.m = makeMolecule()
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
        m = makeMolecule()
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

if __name__ == "__main__":
    unittest.main()


