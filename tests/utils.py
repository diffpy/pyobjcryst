#!/usr/bin/env python
"""Utilities for tests."""

from pyobjcryst import *
from numpy import pi

def makeScatterer():
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(8*pi*pi*0.003)
    atom = Atom(0, 0, 0, "Ni", sp)
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

