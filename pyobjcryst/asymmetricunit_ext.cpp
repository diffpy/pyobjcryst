#include "ObjCryst/SpaceGroup.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

using namespace boost::python;
using namespace ObjCryst;

BOOST_PYTHON_MODULE(_asymmetricunit)
{

    class_<AsymmetricUnit> ("AsymmetricUnit", init<>() )
        // Constructors
        .def(init<const SpaceGroup&>())
        // Methods
        .def("SetSpaceGroup", &AsymmetricUnit::SetSpaceGroup)
        .def("IsInAsymmetricUnit", &AsymmetricUnit::IsInAsymmetricUnit)
        .def("Xmin", &AsymmetricUnit::Xmin)
        .def("Xmax", &AsymmetricUnit::Xmax)
        .def("Ymin", &AsymmetricUnit::Ymin)
        .def("Ymax", &AsymmetricUnit::Ymax)
        .def("Zmin", &AsymmetricUnit::Zmin)
        .def("Zmax", &AsymmetricUnit::Zmax)
        ;
}
