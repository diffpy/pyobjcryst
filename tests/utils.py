#!/usr/bin/env python
##############################################################################
#
# (c) 2025 The Trustees of Columbia University in the City of New York.
# All rights reserved.
#
# File coded by: Chris Farrow and Billinge Group members.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/pyobjcryst/graphs/contributors
#
# See LICENSE.rst and LICENSE_DANSE.rst for license information.
#
##############################################################################
"""Utilities for tests."""

from numpy import pi

from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal, create_crystal_from_cif
from pyobjcryst.molecule import Molecule
from pyobjcryst.polyhedron import MakeOctahedron
from pyobjcryst.scatteringpower import ScatteringPowerAtom


def makeScatterer():
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0, 0, 0, "Ni", sp)
    return sp, atom


def makeScattererAnisotropic():
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.B11 = sp.B22 = sp.B33 = 8 * pi * pi * 0.003
    sp.B12 = sp.B13 = sp.B23 = 0
    atom = Atom(0, 0, 0, "Ni", sp)
    return sp, atom


def makeCrystal(sp, atom):
    c = Crystal(3.52, 3.52, 3.52, "225")
    c.AddScatteringPower(sp)
    c.AddScatterer(atom)
    return c


def getScatterer():
    """Make a crystal and return scatterer from GetScatt."""
    sp, atom = makeScatterer()
    c = makeCrystal(sp, atom)

    sp2 = c.GetScatt(sp.GetName())
    return sp2


c60xyz = """
3.451266498   0.685000000   0.000000000
3.451266498  -0.685000000   0.000000000
-3.451266498   0.685000000   0.000000000
-3.451266498  -0.685000000   0.000000000
0.685000000   0.000000000   3.451266498
-0.685000000   0.000000000   3.451266498
0.685000000   0.000000000  -3.451266498
-0.685000000   0.000000000  -3.451266498
0.000000000   3.451266498   0.685000000
0.000000000   3.451266498  -0.685000000
0.000000000  -3.451266498   0.685000000
0.000000000  -3.451266498  -0.685000000
3.003809890   1.409000000   1.171456608
3.003809890   1.409000000  -1.171456608
3.003809890  -1.409000000   1.171456608
3.003809890  -1.409000000  -1.171456608
-3.003809890   1.409000000   1.171456608
-3.003809890   1.409000000  -1.171456608
-3.003809890  -1.409000000   1.171456608
-3.003809890  -1.409000000  -1.171456608
1.409000000   1.171456608   3.003809890
1.409000000  -1.171456608   3.003809890
-1.409000000   1.171456608   3.003809890
-1.409000000  -1.171456608   3.003809890
1.409000000   1.171456608  -3.003809890
1.409000000  -1.171456608  -3.003809890
-1.409000000   1.171456608  -3.003809890
-1.409000000  -1.171456608  -3.003809890
1.171456608   3.003809890   1.409000000
-1.171456608   3.003809890   1.409000000
1.171456608   3.003809890  -1.409000000
-1.171456608   3.003809890  -1.409000000
1.171456608  -3.003809890   1.409000000
-1.171456608  -3.003809890   1.409000000
1.171456608  -3.003809890  -1.409000000
-1.171456608  -3.003809890  -1.409000000
2.580456608   0.724000000   2.279809890
2.580456608   0.724000000  -2.279809890
2.580456608  -0.724000000   2.279809890
2.580456608  -0.724000000  -2.279809890
-2.580456608   0.724000000   2.279809890
-2.580456608   0.724000000  -2.279809890
-2.580456608  -0.724000000   2.279809890
-2.580456608  -0.724000000  -2.279809890
0.724000000   2.279809890   2.580456608
0.724000000  -2.279809890   2.580456608
-0.724000000   2.279809890   2.580456608
-0.724000000  -2.279809890   2.580456608
0.724000000   2.279809890  -2.580456608
0.724000000  -2.279809890  -2.580456608
-0.724000000   2.279809890  -2.580456608
-0.724000000  -2.279809890  -2.580456608
2.279809890   2.580456608   0.724000000
-2.279809890   2.580456608   0.724000000
2.279809890   2.580456608  -0.724000000
-2.279809890   2.580456608  -0.724000000
2.279809890  -2.580456608   0.724000000
-2.279809890  -2.580456608   0.724000000
2.279809890  -2.580456608  -0.724000000
-2.279809890  -2.580456608  -0.724000000
"""


def makeC60():
    c = Crystal(100, 100, 100, "P1")
    c.SetName("c60frame")
    m = Molecule(c, "c60")

    c.AddScatterer(m)
    sp = ScatteringPowerAtom("C", "C")
    sp.SetBiso(8 * pi * pi * 0.003)
    c.AddScatteringPower(sp)

    for i, l in enumerate(c60xyz.strip().splitlines()):
        x, y, z = map(float, l.split())
        m.AddAtom(x, y, z, sp, "C%i" % i)

    return c


def makeMnO6():
    a = 5.6
    crystal = Crystal(a, a, a, "P1")
    sp1 = ScatteringPowerAtom("Mn", "Mn")
    sp1.SetBiso(8 * pi * pi * 0.003)
    sp2 = ScatteringPowerAtom("O", "O")
    sp2.SetBiso(8 * pi * pi * 0.003)
    crystal.AddScatteringPower(sp1)
    crystal.AddScatteringPower(sp2)

    m = MakeOctahedron(crystal, "MnO6", sp1, sp2, 0.5 * a)

    crystal.AddScatterer(m)

    return crystal


def datafile(filename):
    from importlib.resources import files

    rv = str(files(__name__).joinpath("testdata", filename))
    return rv


def loadcifdata(filename):
    fullpath = datafile(filename)
    crst = create_crystal_from_cif(fullpath)
    return crst
