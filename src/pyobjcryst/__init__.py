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

"""Python wrapping of ObjCryst++.

Objects are wrapped according to their header file in the ObjCryst source.

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Modules

atom                    --  Wrapping of Atom.h
crystal                 --  Wrapping of Crystal.h
general                 --  Wrapping of General.h
io                      --  Wrapping of IO.h
molecule                --  Wrapping of Molecule.h
polyhedron              --  Wrapping of Polyhedron.h
refinableobj            --  Wrapping of RefinableObj.h
scatterer               --  Wrapping of Scatterer.h
scatteringpower         --  Wrapping of ScatteringPower.h
scatteringpowersphere   --  Wrapping of ScatteringPowerSphere.h
spacegroup              --  Wrapping of SpaceGroup.h
unitcell                --  Wrapping of UnitCell.h
zscatterer              --  Wrapping of ZScatterer.h

General Changes

- C++ methods that can return const or non-const objects return non-const
  objects in python.
- Classes with a Print() method have the output of this method exposed in the
  __str__ python method. Thus, obj.Print() == print obj.
- CrystVector and CrystMatrix are converted to numpy arrays.
- Indexing methods raise IndexError when index is out of bounds.

See the modules' documentation for specific changes.
"""

# Let's put this on the package level
from pyobjcryst.general import ObjCrystException

# version data
from pyobjcryst.version import __version__

# import submodules that only import from _pyobjcryst
import pyobjcryst.atom
import pyobjcryst.crystal
import pyobjcryst.diffractiondatasinglecrystal
import pyobjcryst.general
import pyobjcryst.io
import pyobjcryst.molecule
import pyobjcryst.polyhedron
import pyobjcryst.powderpattern
import pyobjcryst.refinableobj
import pyobjcryst.reflectionprofile
import pyobjcryst.scatterer
import pyobjcryst.scatteringdata
import pyobjcryst.scatteringpower
import pyobjcryst.scatteringpowersphere
import pyobjcryst.spacegroup
import pyobjcryst.unitcell
import pyobjcryst.zscatterer


def loadCrystal(filename):
    """Load pyobjcryst Crystal object from a CIF file.

    filename -- CIF file to be loaded

    Return a new Crystal object.

    See pyobjcryst.crystal.CreateCrystalFromCIF for additional
    options for constructing Crystal object from CIF data.
    """
    from pyobjcryst.crystal import CreateCrystalFromCIF
    with open(filename, 'rb') as fp:
        rv = CreateCrystalFromCIF(fp)
    return rv


# silence the pyflakes syntax checker
assert ObjCrystException is not None
assert __version__ or True
assert pyobjcryst.zscatterer
