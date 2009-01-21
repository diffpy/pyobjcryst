#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include "ObjCryst/General.h"

using namespace boost::python;
using namespace ObjCryst;

BOOST_PYTHON_MODULE(_general)
{
    enum_<RadiationType>("RadiationType")
        .value("RAD_NEUTRON", RAD_NEUTRON)
        .value("RAD_XRAY", RAD_XRAY)
        .value("RAD_ELECTRON", RAD_ELECTRON)
        ;
}
