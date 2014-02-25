#!/usr/bin/env python
##############################################################################
#
# PyObjCryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
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
from general import ObjCrystException

# Atom
import atom

# Crystal
import crystal

# Molecule
import molecule

# Polyhedron
import polyhedron

# RefinableObj
import refinableobj

# General
import general

# Scatterer
import scatterer

# ScatteringPower
import scatteringpower

# ScatteringPowerSphere
import scatteringpowersphere

# SpaceGroup
import spacegroup

# UnitCell
import unitcell

# ZScatterer
import zscatterer

# IO
import io

from version import __version__
