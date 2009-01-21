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
* $Id$
*
*****************************************************************************/

#include "ObjCryst/ScatteringPower.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include "helpers.h"

using namespace boost::python;
using namespace ObjCryst;

BOOST_PYTHON_MODULE(_scatteringcomponent)
{

    class_<ScatteringComponent>("ScatteringComponent")
        .def("Print", &ScatteringComponent::Print)
        .def_readwrite("mX", &ScatteringComponent::mX)
        .def_readwrite("mY", &ScatteringComponent::mY)
        .def_readwrite("mZ", &ScatteringComponent::mZ)
        .def_readwrite("mOccupancy", &ScatteringComponent::mOccupancy)
        .def_readonly("mpScattPow", &ScatteringComponent::mpScattPow)
        .def_readwrite("mDynPopCorr", &ScatteringComponent::mDynPopCorr)
        .def("__str__", &__str__<ScatteringComponent>)
        ;
}
