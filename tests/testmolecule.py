#!/usr/bin/env python
"""Tests for molecule module."""

from pyobjcryst import *
import unittest

from utils import *
from numpy import pi

def makeMolecule():

    c = Crystal(10, 10, 10, "P1")
    m = Molecule(c, "testmolecule")

    sp1 = ScatteringPowerAtom("C1", "C")
    sp1.SetBiso(8*pi*pi*0.003)
    m.AddAtom(0, 0, 0, sp1, "C1")

    sp2 = ScatteringPowerAtom("C2", "C")
    sp2.SetBiso(8*pi*pi*0.003)
    m.AddAtom(0, 0, 0.5, sp2, "C2")

    sp3 = ScatteringPowerAtom("C3", "C")
    sp3.SetBiso(8*pi*pi*0.003)
    m.AddAtom(0, 0.5, 0, sp3, "C3")

    sp4 = ScatteringPowerAtom("C4", "C")
    sp4.SetBiso(8*pi*pi*0.004)
    m.AddAtom(0.5, 0, 0, sp4, "C4")

    return m


class TestMolecule(unittest.TestCase):

    def testAtoms(self):
        m = makeMolecule()
        for i in range(4):
            self.assertEqual(m.GetAtom(i).GetName(), "C%i"%(i+1))
        return

    def testFindBond(self):

        m = makeMolecule()
    
        a1 = m.GetAtom(0)
        a2 = m.GetAtom(1)
        a3 = m.GetAtom(2)
        a4 = m.GetAtom(3)

        # Check for a bond that doesn't exist
        bond = m.FindBond(a1, a2)
        self.assertTrue(bond is None)

        # Make a bond and try to find it
        m.AddBond(a1, a2, 5, 0, 0)
        bond1 = m.FindBond(a1, a2)
        bond2 = m.FindBond(a1, a2)
        bond3 = m.FindBond(a1, a3)
        self.assertTrue(bond1 is not None)
        # Cannot expect the python objects to be the same, but they should point
        # to the same internal object
        self.assertEqual(bond1.GetName(), bond2.GetName())

        # Try some bad bonds
        self.assertTrue(bond3 is None)
        return
    
    def testFindBondAngle(self):
        """Test the FindBondAngle method."""
        m = makeMolecule()
        a1 = m.GetAtom(0)
        a2 = m.GetAtom(1)
        a3 = m.GetAtom(2)
        a4 = m.GetAtom(3)

        # Check for a bondangle angle that doesn't exist
        bondangle = m.FindBondAngle(a1, a2, a3)
        self.assertTrue(bondangle is None)

        # Make a bondangle and try to find it
        m.AddBondAngle(a2, a1, a3, 90, 0, 0)
        bondangle1 = m.FindBondAngle(a2, a1, a3)
        bondangle2 = m.FindBondAngle(a2, a1, a3)
        bondangle3 = m.FindBondAngle(a1, a2, a4)
        self.assertTrue(bondangle1 is not None)
        self.assertEqual(bondangle1.GetName(), bondangle2.GetName())

        self.assertTrue(bondangle3 is None)
        return
    
    def testFindDihedralAngle(self):
        """Test the FindDihedralAngle method."""
        m = makeMolecule()
        a1 = m.GetAtom(0)
        a2 = m.GetAtom(1)
        a3 = m.GetAtom(2)
        a4 = m.GetAtom(3)

        # Check for a dihedralangle angle that doesn't exist
        dihedralangle = m.FindDihedralAngle(a1, a2, a3, a4)
        self.assertTrue(dihedralangle is None)

        # Make a dihedralangle and try to find it
        m.AddDihedralAngle(a1, a2, a3, a4, 90, 0, 0)
        dihedralangle1 = m.FindDihedralAngle(a1, a2, a3, a4)
        dihedralangle2 = m.FindDihedralAngle(a1, a2, a3, a4)
        self.assertTrue(dihedralangle1 is not None)
        self.assertEqual(dihedralangle1.GetName(), dihedralangle2.GetName())
        return
    


if __name__ == "__main__":
    unittest.main()


