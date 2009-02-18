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
* boost::python bindings to ObjCryst::RigidGroup.  
* 
* Changes from ObjCryst++
* - RigidGroup is wrapped to have python-set methods rather than stl::set
*   methods.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/Molecule.h"

#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>

#include <set>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

void rgAdd(RigidGroup& rg, MolAtom* a) 
{ 
    rg.insert(a); 
}

void rgUpdate(RigidGroup& rg, const RigidGroup& other)
{
    rg.insert(other.begin(), other.end());
}

bool rgContains(const RigidGroup& rg, MolAtom* a)
{
    return rg.find(a) != rg.end();
}

MolAtom* rgGetItem(const RigidGroup& rg, size_t i)
{
    // Look for size violation
    if (i >= rg.size()) 
    {        
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw_error_already_set();
    }

    RigidGroup::const_iterator p = rg.begin();
    while (i > 0) { p++; i--; }
    return *p;
}

void rgDiscard(RigidGroup& rg,  MolAtom* a)
{
    if( rgContains(rg, a) )
    {
        rg.erase(a);
    }
}

void rgRemove(RigidGroup& rg,  MolAtom* a)
{
    if( rgContains(rg, a) )
    {
        rg.erase(a);
    }
    else
    {
        PyErr_SetString(PyExc_KeyError, "KeyError");
        throw_error_already_set();
    }
}


} // namespace


BOOST_PYTHON_MODULE(_rigidgroup)
{

    class_<RigidGroup>("RigidGroup")
        .def(init<const RigidGroup&>())
        .def("GetName", &RigidGroup::GetName)
        .def("add", &rgAdd, with_custodian_and_ward<1,2>())
        .def("clear", &RigidGroup::clear)
        .def("discard", &rgDiscard)
        .def("remove", &rgRemove)
        .def("update", &rgUpdate, with_custodian_and_ward<1,2>())
        .def("__contains__", &rgContains)
        .def("__getitem__", &rgGetItem, return_internal_reference<>())
        .def("__len__", &RigidGroup::size)
        ;
}
