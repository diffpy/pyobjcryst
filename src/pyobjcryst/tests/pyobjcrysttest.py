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

"""Small tests for pyobjcryst.

To check for memory leaks, run
valgrind --tool=memcheck --leak-check=full /usr/bin/python ./pyobjcrysttest.py
"""

from __future__ import print_function

from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal
from pyobjcryst.refinableobj import RefParType, RefinablePar
from pyobjcryst.scatteringpower import ScatteringPowerAtom

from numpy import pi

def makeScatterer():
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(8*pi*pi*0.003)
    sp.B11 = 8*pi*pi*0.003
    sp.SetBij(2, 2, 8*pi*pi*0.003)
    sp.SetBij(3, 3, 8*pi*pi*0.003)
    atom = Atom(0, 0, 0, "Ni", sp)

    print(sp.B11)
    return sp, atom

def makeCrystal(sp, atom):
    c = Crystal(3.52, 3.52, 3.52, "225")
    c.AddScatterer(atom)
    c.AddScatteringPower(sp)
    return c

def getScatterer():
    """Make a crystal and return scatterer from GetScatt."""
    sp, atom = makeScatterer()
    c = makeCrystal(sp, atom)

    sp2 = c.GetScatt(sp.GetName())
    return sp2

def testCrystalScope():
    """Test to see if the the crystal survives after it is out of scope."""
    sp, atom = makeScatterer()
    makeCrystal(sp, atom)
    # The crystal is out of scope. Since the lifetime of the atom and scatterer
    # are linked, the crystal should stay alive in memory.
    print(sp)
    print(atom)
    print(repr(atom.GetCrystal()))
    return

def testMultiAdd():
    """Test exception for multi-crystal additions."""
    sp, atom = makeScatterer()
    makeCrystal(sp, atom)

    # Force this exception
    try:
        makeCrystal(sp, atom)
        print(sp)
        print(atom)
        print(repr(atom.GetCrystal()))
    except AttributeError as e:
        print("Exception:", e)
    return

def testScattererScope():
    """Test when atoms go out of scope before crystal."""
    c = makeCrystal(*makeScatterer())
    print(c)
    sp2 = getScatterer()
    print(sp2)
    return

def testRemoveFunctions():
    """Test the RemoveScatterer and RemoveScatteringPower method."""
    print("Making Crystal")
    sp, atom = makeScatterer()
    c = makeCrystal(sp, atom)
    print(atom)
    print(sp)
    print(c)

    # Try to add objects with same names
    print("Testing name duplication")
    sp2, atom2 = makeScatterer()
    try:
        c.AddScatterer(atom2)
    except AttributeError as e:
        print(e)
    try:
        c.AddScatteringPower(sp2)
    except AttributeError as e:
        print(e)

    # Remove the scatterers
    print("remove scatterers")
    c.RemoveScatterer(atom)
    c.RemoveScatteringPower(sp)
    print(atom)
    print(sp)
    print(c)

    # Try to remove scatterers that are not in the crystal
    try:
        c.RemoveScatterer(atom2)
    except AttributeError as e:
        print(e)
    try:
        c.RemoveScatteringPower(sp2)
    except AttributeError as e:
        print(e)

    return

def parTest():
    rpt = RefParType("default")
    testpar = RefinablePar("test", 3.0, 0, 10, rpt)
    print(testpar.__class__, testpar)
    sp, atom = makeScatterer()
    c = makeCrystal(sp, atom)
    par = c.GetPar(0)
    print(par.__class__, par)

    c.AddPar(testpar);
    testpar2 = c.GetPar("test")
    print(testpar2.__class__, testpar2)

    del sp, atom, c

    testpar2.SetValue(2.17)
    print(testpar.__class__, testpar)
    return

def test1():
    """Run some tests."""
    testCrystalScope()
    testMultiAdd()
    testScattererScope()
    testRemoveFunctions()
    return

if __name__ == "__main__":
    test1()
