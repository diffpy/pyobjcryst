#!/usr/bin/env python
"""A small test for pyobjcryst."""

from pyobjcryst import *
from numpy import pi

def makeAtom():
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(8*pi*pi*0.003)
    atomp = Atom(0, 0, 0, "Ni", sp)

    return sp, atomp

def makeCrystal(sp, atomp):
    c = Crystal(3.52, 3.52, 3.52, "225")
    c.AddScatterer(atomp)
    c.AddScatteringPower(sp)
    return c

def test1():

    sp, atomp = makeAtom()
    makeCrystal(sp, atomp)
    # The crystal is out of scope. Since the lifetime of the atom and scatterer
    # are linked, the crystal should stay alive in memory.
    print sp
    print atomp
    print repr(atomp.GetCrystal())

    # Now add the objects to a different crystal. This should raise an
    # exception.
    try:
        makeCrystal(sp, atomp)
        print sp
        print atomp
        print repr(atomp.GetCrystal())
    except Exception, e:
        print e

    del sp
    del atomp

    # Now see what happens when the scatterer is allowed to go out of scope
    c = makeCrystal(*makeAtom())
    print c

    return

if __name__ == "__main__":

    test1()
