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
* boost::python bindings to ObjCryst::AsymmetricUnit.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>

#include <ObjCryst/ObjCryst/SpaceGroup.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

void wrap_asymmetricunit()
{

    class_<AsymmetricUnit>("AsymmetricUnit")
        // Constructors
        .def(init<const SpaceGroup&>(bp::arg("spg")))
        // Methods
        .def("SetSpaceGroup", &AsymmetricUnit::SetSpaceGroup, bp::arg("spg"))
        .def("IsInAsymmetricUnit", &AsymmetricUnit::IsInAsymmetricUnit,
            (bp::arg("x"), bp::arg("y"), bp::arg("z")))
        .def("Xmin", &AsymmetricUnit::Xmin)
        .def("Xmax", &AsymmetricUnit::Xmax)
        .def("Ymin", &AsymmetricUnit::Ymin)
        .def("Ymax", &AsymmetricUnit::Ymax)
        .def("Zmin", &AsymmetricUnit::Zmin)
        .def("Zmax", &AsymmetricUnit::Zmax)
        ;
}
