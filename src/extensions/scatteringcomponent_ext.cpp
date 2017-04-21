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
* boost::python bindings to ObjCryst::ScatteringComponent.
*
* Changes from ObjCryst::ScatteringComponent
* - Added attributes X, Y, Z, Occupancy to conform to MolAtom.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <ObjCryst/ObjCryst/ScatteringPower.h>

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

void wrap_scatteringcomponent()
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
        .def_readonly("mDynPopCorr", &ScatteringComponent::mDynPopCorr)
        // Workaround to give attribute access. Again, returning the object,
        // where it should be read-only.
        .add_property("mpScattPow",
            make_function( &_getScatteringPower,
            return_internal_reference<>()))
        .def("__str__", &__str__<ScatteringComponent>)
        ;
}
