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

"""Tests for molecule module."""

import unittest

from pyobjcryst import ObjCrystException
from pyobjcryst.molecule import (
    GetBondLength, StretchModeBondLength,
    GetBondAngle, StretchModeBondAngle,
    GetDihedralAngle, StretchModeTorsion)
from pyobjcryst.refinableobj import RefParType, RefinablePar
from pyobjcryst.tests.pyobjcrysttestutils import makeC60, makeMnO6

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

    def testProperties(self):
        """Make sure we can access the python-only properties."""
        self.m.Q0 *= 1.001
        self.m.Q1 *= 1.001
        self.m.Q2 *= 1.001
        self.m.Q3 *= 1.001
        self.m.X *= 1.001
        self.m.Y *= 1.001
        self.m.Z *= 1.001
        self.m.Occupancy *= 1.001
        return


    def testContainment(self):
        """Make sure we can still use the molecule if the crystal is out of
        scope."""
        c = makeC60()
        m = self.c.GetScatterer("c60")
        self.assertEqual("c60", m.GetName())
        del c
        self.assertEqual("c60", m.GetName())
        return

    def testAddPar(self):
        """See if we crash if we add a parameter and delete the molecule."""
        c = makeC60()
        m = self.c.GetScatterer("c60")
        rpt = RefParType("test")
        par = RefinablePar("testpar", 3, 0, 10, rpt)
        m.AddPar(par)
        self.assertAlmostEqual(3, par.GetValue())
        del m
        self.assertAlmostEqual(3, par.GetValue())
        del c
        self.assertAlmostEqual(3, par.GetValue())
        return

    def testAtoms(self):
        """Make sure the atoms are there.

        This tests AddAtom by association.
        This tests GetAtom.
        """
        self.assertEqual(60, self.m.GetNbAtoms())
        for i in range(60):
            a1 = self.m.GetAtom(i)
            self.assertEqual(a1.GetName(), "C%i"%i)

        a = self.m.GetAtom(0)
        x = a.X

        self.assertEqual(60, self.m.GetNbAtoms())

        self.m.RemoveAtom(a)

        self.assertEqual(59, self.m.GetNbAtoms())

        # Make sure the atom is still valid. We don't want RemoveAtom deleting
        # the memory for an object we still have access to.
        self.assertEqual(a.X, x)

        # Check to see if a is in our list
        for i in range(59):
            self.assertNotEqual(a.GetName(), self.m.GetAtom(i))

        # What happens if we try to remove an atom that is not in the molecule?
        # First, try the same atom again. This will throw an objcryst error.
        self.assertRaises(ObjCrystException, self.m.RemoveAtom, a)

        ## Try to remove an atom from another molecule
        c = makeC60()
        m = c.GetScatterer("c60")
        self.assertRaises(ObjCrystException, self.m.RemoveAtom, m.GetAtom(1))

        # Remove all the atoms.
        for a in self.m[:]:
            self.m.RemoveAtom(a)

        atoms = self.m.GetAtomList()
        self.assertEqual(0, len(atoms))

        self.assertEqual(0, self.m.GetNbAtoms())

        return

    def testBonds(self):
        """Test the Bond methods."""

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
        bond2 = self.m.FindBond(a2, a1)
        bond3 = self.m.FindBond(a1, a3)
        self.assertTrue(bond1 is not None)
        # Cannot expect the python objects to be the same, but they should point
        # to the same internal object
        self.assertEqual(bond1.GetName(), bond2.GetName())

        # Try some bad bonds
        self.assertTrue(bond3 is None)

        # Remove an atom, the bond should disappear as well.
        self.m.RemoveAtom(a1)
        bond4 = self.m.FindBond(a1, a2)
        self.assertTrue(bond4 is None)

        # Try to find a bond from an atom outside of the molecule.
        m = makeC60().GetScatterer("c60")
        b1 = m.GetAtom(0)
        b2 = m.GetAtom(1)
        bond5 = self.m.FindBond(b1, b2)
        self.assertTrue(bond5 is None)

        # Try to make a bond using an atom that is not in the structure...
        # This seems to be allowed by ObjCryst++, and causes no errors. This
        # might be necessary to constrain a distance between two molecules,
        # thus it is allowed.

        # make a good bond.
        bond6 = self.m.AddBond(a3, a4, 5, 0, 0)
        bond7 = self.m.GetBond(0)
        self.assertEqual(bond6.GetName(), bond7.GetName())

        # Delete some bonds and see what happens
        name = bond6.GetName()
        del bond6
        del bond7
        bond8 = self.m.GetBond(0)
        self.assertEqual(name, bond8.GetName())

        # Try to get a bond that doesn't exist by index
        self.assertRaises(IndexError, self.m.GetBond, 1)

        # Remove the bond
        bonds = self.m.GetBondList()
        self.assertEqual(1, len(bonds))
        self.m.RemoveBond(bonds[0])
        # is the bond still in existance?
        self.assertEqual(name, bond8.GetName())
        # Can we get it from the engine?
        self.assertRaises(IndexError, self.m.GetBond, 0)
        bond9 = self.m.FindBond(a3, a4)
        self.assertTrue(bond9 is None)

        # make a good bond again
        bond10 = self.m.AddBond(a3, a4, 5, 0, 0)
        # Get an atom from that
        a = bond10.GetAtom1()
        # Try to remove that atom
        self.m.RemoveAtom(a)
        self.assertEqual(0, self.m.GetNbBonds())

        return

    def testBondAngles(self):
        """Test the BondAngle accessors."""
        a1 = self.m.GetAtom(0)
        a2 = self.m.GetAtom(1)
        a3 = self.m.GetAtom(2)
        a4 = self.m.GetAtom(3)

        # Check for a bondangle angle that doesn't exist
        ba = self.m.FindBondAngle(a1, a2, a3)
        self.assertTrue(ba is None)

        # Make a ba and try to find it
        self.m.AddBondAngle(a2, a1, a3, 90, 0, 0)
        ba1 = self.m.FindBondAngle(a2, a1, a3)
        ba2 = self.m.FindBondAngle(a3, a1, a2)
        ba3 = self.m.FindBondAngle(a1, a2, a4)
        self.assertTrue(ba1 is not None)
        self.assertEqual(ba1.GetName(), ba2.GetName())

        # Try some bad bond angles
        self.assertTrue(ba3 is None)


        # Remove an atom, the bondangle should disappear as well.
        self.m.RemoveAtom(a1)
        ba4 = self.m.FindBondAngle(a2, a1, a3)
        self.assertTrue(ba4 is None)


        # Try to find a bondangle from an atom outside of the molecule.
        m = makeC60().GetScatterer("c60")
        b1 = m.GetAtom(0)
        b2 = m.GetAtom(1)
        b3 = m.GetAtom(1)
        ba5 = self.m.FindBondAngle(b1, b2, b3)
        self.assertTrue(ba5 is None)

        # make a good bond angle
        ba6 = self.m.AddBondAngle(a2, a3, a4, 5, 0, 0)
        ba7 = self.m.GetBondAngle(0)
        self.assertEqual(ba6.GetName(), ba7.GetName())

        # Delete some bond angles and see what happens
        name = ba6.GetName()
        del ba6
        del ba7
        ba8 = self.m.GetBondAngle(0)
        self.assertEqual(name, ba8.GetName())

        # Try to get a bond angle that doesn't exist by index
        self.assertRaises(IndexError, self.m.GetBondAngle, 1)

        # Remove the bond angle
        angles = self.m.GetBondAngleList()
        self.assertEqual(1, len(angles))
        self.m.RemoveBondAngle(angles[0])
        # is the object still in existance?
        self.assertEqual(name, ba8.GetName())
        # Can we get it from the engine?
        self.assertRaises(IndexError, self.m.GetBondAngle, 0)
        ba9 = self.m.FindBondAngle(a2, a3, a4)
        self.assertTrue(ba9 is None)

        # make a good bond angle again
        ba10 = self.m.AddBondAngle(a2, a3, a4, 5, 0, 0)
        # Get an atom from that
        a = ba10.GetAtom1()
        # Try to remove that atom
        self.m.RemoveAtom(a)
        self.assertEqual(0, self.m.GetNbBondAngles())
        return

    def testDihedralAngles(self):
        """Test the FindDihedralAngle method."""
        a1 = self.m.GetAtom(0)
        a2 = self.m.GetAtom(1)
        a3 = self.m.GetAtom(2)
        a4 = self.m.GetAtom(3)
        a5 = self.m.GetAtom(5)

        # Check for a dihedral angle that doesn't exist
        da = self.m.FindDihedralAngle(a1, a2, a3, a4)
        self.assertTrue(da is None)

        # Make a da and try to find it
        self.m.AddDihedralAngle(a1, a2, a3, a4, 90, 0, 0)
        da1 = self.m.FindDihedralAngle(a1, a2, a3, a4)
        da2 = self.m.FindDihedralAngle(a1, a2, a3, a4)
        self.assertTrue(da1 is not None)
        self.assertEqual(da1.GetName(), da2.GetName())


        # Remove an atom, the dihedral angle should disappear as well.
        self.m.RemoveAtom(a1)
        da4 = self.m.FindDihedralAngle(a2, a1, a3, a4)
        self.assertTrue(da4 is None)

        # Try to find a dihedral angle from an atom outside of the molecule.
        m = makeC60().GetScatterer("c60")
        b1 = m.GetAtom(0)
        b2 = m.GetAtom(1)
        b3 = m.GetAtom(1)
        b4 = m.GetAtom(1)
        da5 = self.m.FindDihedralAngle(b1, b2, b3, b4)
        self.assertTrue(da5 is None)

        # make a good dihedral angle
        da6 = self.m.AddDihedralAngle(a2, a3, a4, a5, 5, 0, 0)
        da7 = self.m.GetDihedralAngle(0)
        self.assertEqual(da6.GetName(), da7.GetName())

        # Delete some dihedral angles and see what happens
        name = da6.GetName()
        del da6
        del da7
        da8 = self.m.GetDihedralAngle(0)
        self.assertEqual(name, da8.GetName())

        # Try to get a dihedral angle that doesn't exist by index
        self.assertRaises(IndexError, self.m.GetDihedralAngle, 1)

        # Remove the dihedral angle
        angles = self.m.GetDihedralAngleList()
        self.assertEqual(1, len(angles))
        self.m.RemoveDihedralAngle(angles[0])
        # is the object still in existance?
        self.assertEqual(name, da8.GetName())
        # Can we get it from the engine?
        self.assertRaises(IndexError, self.m.GetDihedralAngle, 0)
        da9 = self.m.FindDihedralAngle(a2, a3, a4, a5)
        self.assertTrue(da9 is None)

        # make a good dihedral angle again
        da10 = self.m.AddDihedralAngle(a2, a3, a4, a5, 5, 0, 0)
        # Get an atom from that
        a = da10.GetAtom1()
        # Try to remove that atom
        self.m.RemoveAtom(a)
        self.assertEqual(0, self.m.GetNbDihedralAngles())

        return

    def testRigidGroup(self):
        """Test adding and manipulating a rigid group."""
        # A rigid group has the interface of a set
        self.assertEqual(0, len(self.m.GetRigidGroupList()))
        self.assertEqual(0, self.m.GetNbRigidGroups())
        rg = self.m.AddRigidGroup(self.m.GetAtomList())
        self.assertEqual(1, self.m.GetNbRigidGroups())
        rgl = self.m.GetRigidGroupList()
        self.assertEqual(1, len(rgl))
        self.assertEqual(60, len(rgl[0]))

        # We would like to check to see if the atoms are the same, but the
        # rigid group is a set, not a list.

        # Test to see if we can remove the list.
        self.m.RemoveRigidGroup(rg)
        self.assertEqual(0, self.m.GetNbRigidGroups())
        rgl = self.m.GetRigidGroupList()
        self.assertEqual(0, len(rgl))

        return

    def testManipulation(self):
        """Test moving the atoms."""
        ac = self.m.AddAtom(0, 0, 0, None, "center")
        self.assertTrue(ac.IsDummy())
        self.m.SetCenterAtom(ac)

        a0 = self.m.GetAtom(0)
        x = a0.X
        y = a0.Y
        z = a0.Z

        # Translate the atoms

        self.m.TranslateAtomGroup(self.m.GetAtomList(), 0, 0, 0.5)

        self.assertAlmostEqual(x, a0.X)
        self.assertAlmostEqual(y, a0.Y)
        self.assertAlmostEqual(z+0.5, a0.Z)

        # Move them back
        self.m.TranslateAtomGroup(self.m.GetAtomList(), 0, 0, -0.5)
        self.assertAlmostEqual(x, a0.X)
        self.assertAlmostEqual(y, a0.Y)
        self.assertAlmostEqual(z, a0.Z)

        # Rotate the atoms

        import numpy
        xyz = [numpy.array([a.X, a.Y, a.Z]) for a in self.m]

        self.m.RotateAtomGroup((0,0,0), (0,0,1),
                self.m.GetAtomList(), pi/2)

        rm = numpy.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for i in range(len(self.m)):
            xyzi = xyz[i]
            newxyz = numpy.dot(rm, xyzi)
            self.assertAlmostEqual(newxyz[0], self.m[i].X, 6)
            self.assertAlmostEqual(newxyz[1], self.m[i].Y, 6)
            self.assertAlmostEqual(newxyz[2], self.m[i].Z, 6)

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
        return

    def testAccessors(self):

        a = self.a

        # Test name Get/Set
        self.assertEqual("C0", a.GetName())
        a.SetName("test")
        self.assertEqual("test", a.GetName())

        # Test xyz & occ Get/Set
        self.assertAlmostEqual(3.451266498, a.X, numplaces)
        self.assertAlmostEqual(0.685, a.Y, numplaces)
        self.assertAlmostEqual(0, a.Z, numplaces)
        self.assertAlmostEqual(1.0, a.Occupancy, numplaces)

        a.x = 3.40
        a.y = 0.68
        a.z = 0.1
        a.occ = 1.02

        self.assertAlmostEqual(3.40, a.x, numplaces)
        self.assertAlmostEqual(0.68, a.y, numplaces)
        self.assertAlmostEqual(0.1, a.z, numplaces)
        self.assertAlmostEqual(1.02, a.occ, numplaces)

        # Test GetMolecule. We can't expect the python object to be the same as
        # our molecule above. However, we can verify that it points to the same
        # object.
        m = a.GetMolecule()
        self.assertEqual(m.GetName(), self.m.GetName())
        # Change something with the molecule, and check to see if it appears in
        # self.m
        m.GetAtom("C1").Occupancy = 0.1
        self.assertAlmostEqual(0.1, self.m.GetAtom("C1").Occupancy, numplaces)

        # Test IsDummy
        self.assertFalse(a.IsDummy())

        # Test GetScatteringPower
        sp = a.GetScatteringPower()
        self.assertEqual("ScatteringPowerAtom", sp.GetClassName())
        self.assertEqual("C", sp.GetName())

        # Test Ring Get/Set
        self.assertFalse(a.IsInRing())
        a.SetIsInRing(True)
        self.assertTrue(a.IsInRing())
        a.SetIsInRing(False)
        self.assertFalse(a.IsInRing())

        return

# End class TestMolAtom

class TestMolBond(unittest.TestCase):

    def setUp(self):
        c = makeC60()
        self.m = c.GetScatterer("c60")

        # Add a bond
        self.a1 = self.m.GetAtom(0)
        self.a2 = self.m.GetAtom(1)

        self.b = self.m.AddBond(self.a1, self.a2, 5, 1, 2)
        return

    def tearDown(self):
        del self.m
        del self.a1
        del self.a2
        del self.b
        return

    def testAccessors(self):

        m = self.m
        b = self.b
        a1 = self.a1
        a2 = self.a2

        # Check the name
        self.assertEqual("C0-C1", b.GetName())

        # Get the atoms
        at1 = b.GetAtom1()
        at2 = b.GetAtom2()
        self.assertEqual(at1.GetName(), a1.GetName())
        self.assertEqual(at2.GetName(), a2.GetName())

        # Data
        self.assertAlmostEqual(5, b.Length0, numplaces)
        self.assertAlmostEqual(1, b.LengthSigma, numplaces)
        self.assertAlmostEqual(2, b.LengthDelta, numplaces)
        self.assertAlmostEqual(1, b.BondOrder, numplaces)
        b.Length0 = 1.2
        b.LengthSigma = 2
        b.LengthDelta = 1
        b.BondOrder = 2
        self.assertAlmostEqual(1.2, b.Length0, numplaces)
        self.assertAlmostEqual(2, b.LengthSigma, numplaces)
        self.assertAlmostEqual(1, b.LengthDelta, numplaces)
        self.assertAlmostEqual(2, b.BondOrder, numplaces)

        # Check the log likelihood of the bond and the containing molecule
        b.Length0 = 4
        ll = ((b.Length - (b.Length0-b.LengthDelta))/b.LengthSigma)**2
        self.assertAlmostEqual(ll, b.GetLogLikelihood(), numplaces)
        self.assertAlmostEqual(ll, m.GetLogLikelihood(), numplaces)

        return

# End class TestMolBond

class TestMolBondAngle(unittest.TestCase):

    def setUp(self):
        c = makeC60()
        self.m = c.GetScatterer("c60")

        # Add a bond
        self.a1 = self.m.GetAtom(0)
        self.a2 = self.m.GetAtom(1)
        self.a3 = self.m.GetAtom(2)

        self.ba = self.m.AddBondAngle(self.a1, self.a2, self.a3, 5, 1, 2)
        return

    def tearDown(self):
        del self.m
        del self.a1
        del self.a2
        del self.a3
        del self.ba
        return

    def testAccessors(self):

        m = self.m
        ba = self.ba
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3

        # Check the name
        self.assertEqual("C0-C1-C2", ba.GetName())

        # Get the atoms
        at1 = ba.GetAtom1()
        at2 = ba.GetAtom2()
        at3 = ba.GetAtom3()
        self.assertEqual(at1.GetName(), a1.GetName())
        self.assertEqual(at2.GetName(), a2.GetName())
        self.assertEqual(at3.GetName(), a3.GetName())

        # Data
        self.assertAlmostEqual(5, ba.Angle0, numplaces)
        self.assertAlmostEqual(1, ba.AngleSigma, numplaces)
        self.assertAlmostEqual(2, ba.AngleDelta, numplaces)
        ba.Angle0 = 1.2
        ba.AngleSigma = 2
        ba.AngleDelta = 1
        self.assertAlmostEqual(1.2, ba.Angle0, numplaces)
        self.assertAlmostEqual(2, ba.AngleSigma, numplaces)
        self.assertAlmostEqual(1, ba.AngleDelta, numplaces)

        # Check the log likelihood of the bond and the containing molecule
        ba.Angle0 = 4
        ll = ((ba.Angle - (ba.Angle0-ba.AngleDelta))/ba.AngleSigma)**2
        self.assertAlmostEqual(ll, ba.GetLogLikelihood(), numplaces)
        self.assertAlmostEqual(ll, m.GetLogLikelihood(), numplaces)

        return

# End class TestMolBondAngle

class TestMolDihedralAngle(unittest.TestCase):

    def setUp(self):
        c = makeC60()
        self.m = c.GetScatterer("c60")

        # Add a bond
        self.a1 = self.m.GetAtom(0)
        self.a2 = self.m.GetAtom(1)
        self.a3 = self.m.GetAtom(2)
        self.a4 = self.m.GetAtom(3)

        self.da = self.m.AddDihedralAngle(self.a1, self.a2, self.a3, self.a4,
                5, 1, 2)
        return

    def tearDown(self):
        del self.m
        del self.a1
        del self.a2
        del self.a3
        del self.a4
        del self.da
        return

    def testAccessors(self):

        m = self.m
        da = self.da
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4

        # Check the name
        self.assertEqual("C0-C1-C2-C3", da.GetName())

        # Get the atoms
        at1 = da.GetAtom1()
        at2 = da.GetAtom2()
        at3 = da.GetAtom3()
        at4 = da.GetAtom4()
        self.assertEqual(at1.GetName(), a1.GetName())
        self.assertEqual(at2.GetName(), a2.GetName())
        self.assertEqual(at3.GetName(), a3.GetName())
        self.assertEqual(at4.GetName(), a4.GetName())


        # Data
        # Note that the angle is in [-pi, pi]
        from math import pi
        self.assertAlmostEqual(5-2*pi, da.Angle0, numplaces)
        self.assertAlmostEqual(1, da.AngleSigma, numplaces)
        self.assertAlmostEqual(2, da.AngleDelta, numplaces)
        da.Angle0 = 1.2
        da.AngleSigma = 2
        da.AngleDelta = 1
        self.assertAlmostEqual(1.2, da.Angle0, numplaces)
        self.assertAlmostEqual(2, da.AngleSigma, numplaces)
        self.assertAlmostEqual(1, da.AngleDelta, numplaces)

        # Check the log likelihood of the bond and the containing molecule
        da.Angle0 = pi-0.2
        da.AngleDelta = 0
        da.AngleSigma = 0.1
        angle = da.Angle + (da.Angle0-da.AngleDelta) - 2*pi
        ll = (angle/da.AngleSigma)**2

        # For some reason these are not very close in value.
        self.assertAlmostEqual(ll, da.GetLogLikelihood(), 2)
        self.assertAlmostEqual(ll, m.GetLogLikelihood(), 2)

        return

# End class TestMolDihedralAngle

class TestStretchModeBondLength(unittest.TestCase):

    def setUp(self):
        self.c = makeMnO6()
        self.m = self.c.GetScatterer("MnO6")
        return

    def tearDown(self):
        del self.m
        del self.c
        return

    def testStretchModeBondLength(self):
        """Test the StretchModeBondLength class."""

        # Measure the distance
        ac = self.m[0]
        # The 0, 0, z atom
        atop = self.m[1]
        # The 0, 0, -z atom
        abot = self.m[6]
        d0 = GetBondLength(atop, abot)
        dc0 = GetBondLength(ac, atop)

        # Now create a stretch mode with just these two
        sm = StretchModeBondLength(atop, abot, None)
        sm.AddAtom(abot)

        self.assertEqual(sm.mpAtom0.GetName(), atop.GetName())
        self.assertEqual(sm.mpAtom1.GetName(), abot.GetName())

        # Stretch the bond by 5%
        delta = 0.05 * d0
        sm.Stretch(delta)

        # Make sure this does what we expected
        d1 = GetBondLength(atop, abot)
        self.assertAlmostEqual(d0+delta, d1, 6)

        # Note that only the second atom has moved
        dc1 = GetBondLength(ac, atop)
        self.assertAlmostEqual(dc0, dc1)

        return

# End class TestStretchModeBondLength

class TestStretchModeBondAngle(unittest.TestCase):

    def setUp(self):
        self.c = makeMnO6()
        self.m = self.c.GetScatterer("MnO6")
        return

    def tearDown(self):
        del self.m
        del self.c
        return

    def testStretchModeBondAngle(self):
        """Test the StretchModeBondLength class."""

        a1 = self.m[1]
        ac = self.m[0]
        a2 = self.m[2]

        # Measure the angle
        angle0 = GetBondAngle(a1, ac, a2)

        # Now create a stretch mode with these
        sm = StretchModeBondAngle(a1, ac, a2, None)
        sm.AddAtom(a2)

        self.assertEqual(sm.mpAtom0.GetName(), a1.GetName())
        self.assertEqual(sm.mpAtom1.GetName(), ac.GetName())
        self.assertEqual(sm.mpAtom2.GetName(), a2.GetName())


        # Stretch the angle by 5%
        delta = 0.05 * angle0
        sm.Stretch(delta)

        # Make sure this does what we expected
        angle1 = GetBondAngle(a1, ac, a2)
        self.assertAlmostEqual(angle0+delta, angle1, 6)

        return

# End class TestStretchModeBondAngle

class TestStretchModeTorsion(unittest.TestCase):

    def setUp(self):
        self.c = makeMnO6()
        self.m = self.c.GetScatterer("MnO6")
        return

    def tearDown(self):
        del self.m
        del self.c
        return

    def testStretchModeTorsion(self):
        """Test the StretchModeBondLength class."""

        a1 = self.m[1]
        ac0 = self.m[3]
        ac1 = self.m[0]
        a2 = self.m[2]

        # Measure the angle
        angle0 = GetDihedralAngle(a1, ac0, ac1, a2)

        # Now create a stretch mode with the central bond
        sm = StretchModeTorsion(ac0, ac1, None)
        # Add the last atom so it can rotate
        sm.AddAtom(a2)

        self.assertEqual(sm.mpAtom1.GetName(), ac0.GetName())
        self.assertEqual(sm.mpAtom2.GetName(), ac1.GetName())

        # Stretch the angle by 5%
        delta = 0.25 * angle0
        sm.Stretch(delta)

        # Make sure this does what we expected
        angle1 = GetDihedralAngle(a1, ac0, ac1, a2)
        self.assertAlmostEqual(angle0+delta, angle1, 6)

        return

    def testDummy(self):
        """Test adding a dummy atom."""
        # In this past, dummy atoms would cause seg-faults in crystal::Print.
        # We test that here.
        self.m.AddAtom(0, 0, 0, None, "center")

        sp = self.m[-1].GetScatteringPower()
        self.assertTrue(sp is None)

        sm = str(self.m)
        self.assertEqual(8, sm.count('Atom Name'))

        sc = str(self.c)
        sclines = sc.splitlines()
        self.assertTrue(sclines[2].endswith(' 8'))
        self.assertTrue('ScattPow: dummy' in sc)

        return

# End class TestStretchTorsion

if __name__ == "__main__":
    unittest.main()
