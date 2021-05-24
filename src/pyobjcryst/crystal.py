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

"""Python wrapping of Crystal.h.

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::Crystal

- CIFOutput accepts a python file-like object
- CalcDynPopCorr is not enabled, as the API states that this is for internal
  use only.

Other Changes

- CreateCrystalFromCIF is placed here instead of in a seperate CIF module. This
  method accepts a python file or a filename rather than a CIF object.
"""

__all__ = ["Crystal", "BumpMergePar", "CreateCrystalFromCIF",
           "create_crystal_from_cif", "gCrystalRegistry"]

import urllib
from pyobjcryst._pyobjcryst import Crystal as Crystal_orig
from pyobjcryst._pyobjcryst import BumpMergePar
from pyobjcryst._pyobjcryst import CreateCrystalFromCIF as CreateCrystalFromCIF_orig
from pyobjcryst._pyobjcryst import gCrystalRegistry


class Crystal(Crystal_orig):

    def CIFOutput(self,file, mindist=0):
        """
        Save the crystal structure to a CIF file.

        :param file: either a filename, or a python file object opened in write mode
        """
        if isinstance(file, str):
            super().CIFOutput(open(file,"w"), mindist)
        else:
            super().CIFOutput(file, mindist)


def create_crystal_from_cif(file, oneScatteringPowerPerElement=False,
                            connectAtoms=False):
    """
    Create a crystal object from a CIF file or URL
    Example:
        create_crystal_from_cif('http://www.crystallography.net/cod/2201530.cif')

    :param file: the filename/URL or python file object (need to open with mode='rb')
                 If the string begins by 'http' it is assumed that it is an URL instead,
                 e.g. from the crystallography open database
    :param oneScatteringPowerPerElement: if False (the default), then there will
          be as many ScatteringPowerAtom created as there are different elements.
          If True, only one will be created per element.
    :param connectAtoms: if True, call Crystal::ConnectAtoms to try to create
          as many Molecules as possible from the list of imported atoms.
    :return: the imported Crystal structure
    :raises: ObjCrystException - if no Crystal object can be imported
    """
    if isinstance(file, str):
        if len(file) > 4:
            if file[:4].lower() == 'http':
                return CreateCrystalFromCIF_orig(urllib.request.urlopen(file),
                                                 oneScatteringPowerPerElement, connectAtoms)
        with open(file, 'rb') as cif:  # Make sure file object is closed
            c = CreateCrystalFromCIF_orig(cif, oneScatteringPowerPerElement, connectAtoms)
    else:
        c = CreateCrystalFromCIF_orig(file, oneScatteringPowerPerElement, connectAtoms)
    return c


# PEP8, functions should be lowercase
CreateCrystalFromCIF = create_crystal_from_cif
