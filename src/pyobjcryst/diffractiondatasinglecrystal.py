#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        Complex Modeling Initiative
#                   (c) 2015 Brookhaven Science Associates
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Kevin Knox
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
"""Python wrapping of DiffractionDataSingleCrystal.h.

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io).

Changes from ObjCryst::DiffractionDataSingleCrystal::
        In development !
"""

__all__ = [
    "DiffractionDataSingleCrystal",
    "gDiffractionDataSingleCrystalRegistry",
    "create_singlecrystaldata_from_cif",
]

from urllib.request import urlopen

from pyobjcryst._pyobjcryst import CreateSingleCrystalDataFromCIF as crcif
from pyobjcryst._pyobjcryst import (
    DiffractionDataSingleCrystal,
    gDiffractionDataSingleCrystalRegistry,
)


def create_singlecrystaldata_from_cif(file, crystal):
    """
    Create a DiffractionDataSingleCrystal object from a CIF file. Note that
    this will use the last created Crystal as a reference structure.
    Example using the COD to load both crystal and data:
        c=create_crystal_from_cif('http://www.crystallography.net/cod/2201530.cif')
        d=create_singlecrystaldata_from_cif('http://www.crystallography.net/cod/2201530.hkl', c)

    param file: the filename/URL or python file object (need to open with mode='rb').
                 If the string begins by 'http' it is assumed that it is an URL instead,
                 e.g. from the crystallography open database
    param crystal: the Crystal object which will be used for this single crystal data.
    :return: the imported DiffractionDataSingleCrystal
    :raises: ObjCrystException - if no DiffractionDataSingleCrystal object can be created
    """
    if isinstance(file, str):
        if len(file) > 4:
            if file[:4].lower() == "http":
                return crcif(urlopen(file), crystal)
        with open(file, "rb") as cif:  # Make sure file object is closed
            c = crcif(cif, crystal)
    else:
        c = crcif(file, crystal)
    return c


# PEP8
CreateSingleCrystalDataFromCIF = create_singlecrystaldata_from_cif
