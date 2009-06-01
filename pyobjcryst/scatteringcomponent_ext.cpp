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
* boost::python bindings to ObjCryst::ScatteringComponent.
*
* Changes from ObjCryst++
* - Added accessors X, Y, Z, Occupancy to conform to MolAtom.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/ScatteringPower.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include "helpers.hpp"

using namespace boost::python;
using namespace ObjCryst;

namespace
{

const ScatteringPower* _getScatteringPower(ScatteringComponent& s)
{
    return s.mpScattPow;
}

}

BOOST_PYTHON_MODULE(_scatteringcomponent)
{

    class_<ScatteringComponent>("ScatteringComponent")
        .def("Print", &ScatteringComponent::Print)
        .def_readwrite("mX", &ScatteringComponent::mX)
        .def_readwrite("X", &ScatteringComponent::mX)
        .def_readwrite("mY", &ScatteringComponent::mY)
        .def_readwrite("Y", &ScatteringComponent::mY)
        .def_readwrite("mZ", &ScatteringComponent::mZ)
        .def_readwrite("Z", &ScatteringComponent::mZ)
        .def_readwrite("mOccupancy", &ScatteringComponent::mOccupancy)
        .def_readwrite("Occupancy", &ScatteringComponent::mOccupancy)
        .def_readwrite("mDynPopCorr", &ScatteringComponent::mDynPopCorr)
        // Workaround to give attribute access. Again, returning the object,
        // where it should be read-only.
        .add_property("mpScattPow", 
            make_function( &_getScatteringPower, 
            return_internal_reference<>()))
        .def("__str__", &__str__<ScatteringComponent>)
        ;
}
