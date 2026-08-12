/*
 * pyobjcryst nanobind port — GlobalScatteringPower bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_globalscatteringpower(nb::module_& m)
{
    nb::class_<GlobalScatteringPower> cls(m, "GlobalScatteringPower", nb::is_final());
    cls
        .def(nb::init<>())
        .def(nb::init<const ZScatterer&>())
        .def(nb::init<const GlobalScatteringPower&>())
        .def("Init",      nb::overload_cast<const ZScatterer&>(&GlobalScatteringPower::Init))
        .def("GetRadius", &GlobalScatteringPower::GetRadius)
        ;

    // GlobalScatteringPower inherits ScatteringPower virtually (like
    // ScatteringPowerAtom) -- neither ScatteringPower's own method surface
    // nor RefinableObj's is reachable via nanobind inheritance, so forward
    // both directly. See nb_scatteringpoweratom.cpp / helpers_nb.hpp.
    bind_scatteringpower_forwarding(cls);
    bind_refinableobj_forwarding(cls);
}
