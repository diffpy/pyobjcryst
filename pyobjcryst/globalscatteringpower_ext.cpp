#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include "ObjCryst/ScatteringPower.h"
#include "ObjCryst/ZScatterer.h"
#include "RefinableObj/RefinableObj.h"
#include "CrystVector/CrystVector.h"

using namespace boost::python;
using namespace ObjCryst;

BOOST_PYTHON_MODULE(_globalscatteringpower)
{

    class_<GlobalScatteringPower, bases<ScatteringPower> > ("GlobalScatteringPower", 
        init<>())
        .def(init<const ZScatterer &>())
        .def(init<const GlobalScatteringPower&>())
        .def("Init", &GlobalScatteringPower::Init)
        .def("GetRadius", &GlobalScatteringPower::GetRadius)
        ;
}
