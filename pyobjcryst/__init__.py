#!/usr/bin/env python
########################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 trustees of the Michigan State University
#                   All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
########################################################################

# FIXME - Remove imports and import only the modules.

"""Python wrapping of ObjCryst++.

Objects are wrapped according to their header file in the ObjCryst source. 

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Modules
    atom                    --  Wrapping of Atom.h
    crystal                 --  Wrapping of Crystal.h
    general                 --  Wrapping of General.h
    io                      --  Wrapping of IO.h
    molecule                --  Wrapping of Molecule.h
    refinableobj            --  Wrapping of RefinableObj.h
    scatterer               --  Wrapping of Scatterer.h
    scatteringpowersphere   --  Wrapping of ScatteringPowerSphere.h
    scatteringpower         --  Wrapping of ScatteringPower.h
    spacegroup              --  Wrapping of SpaceGroup.h
    unitcell                --  Wrapping of UnitCell.h
    zscatterer              --  Wrapping of ZScatterer.h

"""

# Atom
import atom

# Crystal
import crystal

# Molecule
import molecule

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

__id__ = "$Id$"
