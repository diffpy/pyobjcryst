/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Converter functions. These are registered in a special extention implemented
* in registerconverters.cpp, which is imported in __init__.py. Creating
* convereters in this fashion keeps them all in one place and eliminates
* duplication of effort.
*
* $Id$
*
*****************************************************************************/

#ifndef _CONVERTERS_H
#define _CONVERTERS_H

#include <vector>

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/to_python_converter.hpp>

#include "CrystVector/CrystVector.h"
#include "ObjCryst/SpaceGroup.h"

#include <numpy/noprefix.h>
#include <numpy/arrayobject.h>

// 
//
using namespace boost::python;

// Make an array out of a data pointer and a dimension vector
PyObject* makeNdArray(float * data, std::vector<int>& dims)
{
    PyObject* pyarray = PyArray_SimpleNewFromData
                (dims.size(), &dims[0], PyArray_FLOAT, (void *) data);
    return incref(PyArray_Copy( (PyArrayObject*) pyarray ));
}

// CrystVector to ndarray
//

struct CrystVector_REAL_to_ndarray
{

    static PyObject* convert(CrystVector<float> const &cv)
    {
        static std::vector<int> dims(1,cv.numElements());
        return makeNdArray((float *) cv.data(), dims);
    }

    //static PyTypeObject const *get_ptype()
    //{
    //    return PyArray_Type;
    //    //return PyArray_FLOAT;
    //}

};

struct CrystMatrix_REAL_to_ndarray
{

    static PyObject* convert(CrystMatrix<float> const &cm)
    {
        std::vector<int> dims(2);
        dims[0] = cm.rows();
        dims[1] = cm.cols();
        return makeNdArray((float *) cm.data(), dims);
    }

};

template <typename T1, typename T2>
struct std_pair_to_tuple
{
static PyObject* convert(std::pair<T1, T2> const& p)
{
  return boost::python::incref(
    boost::python::make_tuple(p.first, p.second).ptr());
}
static PyTypeObject const *get_pytype () {return &PyTuple_Type; }
};

// Helper for convenience.
template <typename T1, typename T2>
struct std_pair_to_python_converter
{
std_pair_to_python_converter()
{
  boost::python::to_python_converter<
    std::pair<T1, T2>,
    std_pair_to_tuple<T1, T2>
    >();
}
};

#endif
