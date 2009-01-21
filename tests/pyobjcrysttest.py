#!/usr/bin/env python
"""A small test for pyobjcryst."""

from pyobjcryst import *
from numpy import pi

def test1():

    print 'c = Crystal(3.52, 3.52, 3.52, "225")'
    c = Crystal(3.52, 3.52, 3.52, "225")
    print 'c.Print()'
    c.Print()
    print 'sp = ScatteringPowerAtom("Ni", "Ni")'
    sp = ScatteringPowerAtom("Ni", "Ni")
    print 'sp.SetBiso(8*pi*pi*0.003)'
    sp.SetBiso(8*pi*pi*0.003)
    print 'atomp = Atom(0, 0, 0, "Ni", sp)'
    atomp = Atom(0, 0, 0, "Ni", sp)
    print 'atomp.Print()'
    atomp.Print()
    print 'c.AddScatterer(atomp)'
    c.AddScatterer(atomp)
    print 'c.AddScatteringPower(sp)'
    c.AddScatteringPower(sp)
    print 'c.Print()'
    c.Print()
    print 'c.RemoveScatterer(atomp)'
    c.RemoveScatterer(atomp)
    print 'c.Print()'
    c.Print()
    print 'atomp.Print()'
    atomp.Print()
    print 'sp.Print()'
    sp.Print()
    return

if __name__ == "__main__":

    test1()
