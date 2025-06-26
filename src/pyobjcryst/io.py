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
"""Python wrapping of IO.h.

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io).

Changes from ObjCryst::XMLCrystTag
- The istream constructor of XMLCrystTag is not wrapped.

Changes from IO functions:
- use lower case for xml_cryst_file_load_all_object and xml_cryst_file_save_global
- both functions accept either a file object or a filename, and can handle compressed
  files ('.xmlgz)
"""

__all__ = [
    "XMLCrystTag",
    "xml_cryst_file_load_all_object",
    "xml_cryst_file_save_global",
]

import gzip
import os

from pyobjcryst._pyobjcryst import (
    XMLCrystFileLoadAllObject as XMLCrystFileLoadAllObject_orig,
)
from pyobjcryst._pyobjcryst import (
    XMLCrystFileSaveGlobal as XMLCrystFileSaveGlobal_orig,
)
from pyobjcryst._pyobjcryst import XMLCrystTag

from .globals import gTopRefinableObjRegistry


def xml_cryst_file_load_all_object(file, verbose=False):
    """
    Load all objects from an ObjCryst-formatted .xml or .xmlgz file
    :param file: the filename or python file object (need to open with mode='rb')
    :param verbose: if True, some information about the loaded objects is printed
    :return: a list of the imported top-level objects (Crystal,
             DiffractionDataSingleCrystal, PowderPattern)
    """
    nb0 = len(gTopRefinableObjRegistry)
    if isinstance(file, str):
        if os.path.splitext(file)[-1] == ".xmlgz":
            o = gzip.open(file, mode="rb")
            XMLCrystFileLoadAllObject_orig(o, verbose=verbose)
        else:
            XMLCrystFileLoadAllObject_orig(open(file, "rb"), verbose=verbose)
    else:
        XMLCrystFileLoadAllObject_orig(file, verbose=verbose)
    return gTopRefinableObjRegistry[nb0:]


def xml_cryst_file_save_global(file):
    """
    Save all top-level objects to an ObjCryst-formatted .xml or .xmlgz file
    :param file: the filename or python file object (need to open with mode='rb').
                 If a filename is given and the extension is 'xmlgz', the file
                 will be compressed by gzip
    :return: nothing
    """
    if isinstance(file, str):
        if os.path.splitext(file)[-1] == ".xmlgz":
            o = gzip.open(
                file,
                mode="wt",
                compresslevel=9,
                encoding=None,
                errors=None,
                newline=None,
            )
            XMLCrystFileSaveGlobal_orig(o)
        else:
            XMLCrystFileSaveGlobal_orig(open(file, "w"))
    else:
        XMLCrystFileSaveGlobal_orig(file)
