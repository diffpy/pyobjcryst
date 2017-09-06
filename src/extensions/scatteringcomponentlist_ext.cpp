/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::ScatteringComponentList.
*
* Changes from ObjCryst::ScatteringComponentList
* - Wrapped as a to-python converter only (no constructor)
* - Added python list-like access
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/slice.hpp>

#include <ObjCryst/ObjCryst/ScatteringPower.h>

#include "helpers.hpp"


using namespace boost::python;
using namespace ObjCryst;

namespace
{

const ScatteringComponent&
getItem(const ScatteringComponentList& scl, long idx)
{
    long n = scl.GetNbComponent();
    if(idx < 0) idx += n;
    if(idx < 0 || idx >= n)
    {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw_error_already_set();
    }
    return scl(idx);
}

bool contains(const ScatteringComponentList& scl,
        const ScatteringComponent& sc)
{
    for(long i=0; i < scl.GetNbComponent(); ++i)
    {
        if( scl(i) == sc ) return true;
    }
    return false;
}

// Get slices directly from the boost python object
bp::object getSCSlice(bp::object & scl, bp::slice& s)
{
    bp::list l(scl);
    return l[s];
}


}

void wrap_scatteringcomponentlist()
{

    class_<ScatteringComponentList>
        ("ScatteringComponentList", no_init)
        //("ScatteringComponentList", init<const long>())
        //.def(init<const ScatteringComponentList &>())
        .def("Reset", &ScatteringComponentList::Reset)
        .def("GetNbComponent", &ScatteringComponentList::GetNbComponent)
        .def("Print", &ScatteringComponentList::Print)
        .def(self == self)
        .def(self += self)
        .def(self += ScatteringComponent())
        .def("__str__", &__str__<ScatteringComponentList>)
        // Container-type things
        .def("__len__", &ScatteringComponentList::GetNbComponent)
        .def("__getitem__", &getItem, return_internal_reference<>())
        .def("__getitem__", &getSCSlice,
                with_custodian_and_ward_postcall<1,0>())
        .def("__contains__", &contains)
        ;
}
