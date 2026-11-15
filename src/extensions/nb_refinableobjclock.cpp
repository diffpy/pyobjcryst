/*
 * pyobjcryst nanobind port — RefinableObjClock bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/operators.h>

#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_refinableobjclock(nb::module_& m)
{
    nb::class_<RefinableObjClock>(m, "RefinableObjClock",
        "Internal clock used by ObjCryst++ to track modification times.")
        .def(nb::init<>())
        .def("AddChild",   &RefinableObjClock::AddChild,   nb::keep_alive<1,2>())
        .def("AddParent",  &RefinableObjClock::AddParent,  nb::keep_alive<1,2>())
        .def("Click",      &RefinableObjClock::Click)
        .def("Print",      &RefinableObjClock::Print)
        .def("PrintStatic",&RefinableObjClock::PrintStatic)
        .def("RemoveChild",&RefinableObjClock::RemoveChild)
        .def("RemoveParent",&RefinableObjClock::RemoveParent)
        .def("Reset",      &RefinableObjClock::Reset)
        .def("SetEqual",   [](RefinableObjClock& c1, const RefinableObjClock& c2){ c1 = c2; })
        .def("__lt__",  [](const RefinableObjClock& a, const RefinableObjClock& b){ return a < b; })
        .def("__le__",  [](const RefinableObjClock& a, const RefinableObjClock& b){ return a <= b; })
        .def("__gt__",  [](const RefinableObjClock& a, const RefinableObjClock& b){ return a > b; })
        .def("__ge__",  [](const RefinableObjClock& a, const RefinableObjClock& b){ return a >= b; })
        .def("__str__", [](const RefinableObjClock& c){ return obj_str(c); })
        ;
}
