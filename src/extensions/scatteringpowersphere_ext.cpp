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
* boost::python bindings to ObjCryst::ScatteringPowerSphere.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>

#include <string>

#include <ObjCryst/ObjCryst/ScatteringPowerSphere.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


void wrap_scatteringpowersphere()
{

    typedef void (ScatteringPowerSphere::*SPSInitType)(
            const string&, const double, const double);
    SPSInitType theinit = &ScatteringPowerSphere::Init;

    class_<ScatteringPowerSphere, bases<ScatteringPower> >
        ("ScatteringPowerSphere")
        .def(init<const std::string&, const double,
                  bp::optional<const double> >())
        .def("Init", theinit,
                (boost::python::arg("name"),
                boost::python::arg("radius"),
                boost::python::arg("biso")=1.0
                ))
        .def("GetRadius", &ScatteringPowerSphere::GetRadius)
        ;
}
