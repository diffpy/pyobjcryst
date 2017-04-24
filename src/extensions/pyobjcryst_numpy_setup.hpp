/*****************************************************************************
*
* pyobjcryst        Complex Modeling Initiative
*                   (c) 2017 Brookhaven Science Associates
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Define PY_ARRAY_UNIQUE_SYMBOL for the pyobjcryst extension module.
*
*****************************************************************************/

#ifndef PYOBJCRYST_NUMPY_SETUP_HPP_INCLUDED
#define PYOBJCRYST_NUMPY_SETUP_HPP_INCLUDED

// Specify the version of NumPy API that will be used.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// This macro is required for extension modules that are in several files.
// It must be defined before inclusion of numpy/arrayobject.h
#define PY_ARRAY_UNIQUE_SYMBOL PYOBJCRYST_NUMPY_ARRAY_SYMBOL

#endif  // PYOBJCRYST_NUMPY_SETUP_HPP_INCLUDED
