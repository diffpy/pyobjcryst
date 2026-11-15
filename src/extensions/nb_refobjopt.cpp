/*
 * pyobjcryst nanobind port — RefObjOpt bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/RefinableObj/RefinableObj.h>

namespace nb = nanobind;
using namespace ObjCryst;

struct PyRefObjOpt : RefObjOpt {
    NB_TRAMPOLINE(RefObjOpt, 1);
    void SetChoice(const int choice) override { NB_OVERRIDE(SetChoice, choice); }
};

void wrap_refobjopt(nb::module_& m)
{
    nb::class_<RefObjOpt, PyRefObjOpt>(m, "RefObjOpt")
        .def(nb::init<>())
        .def("Init",        &RefObjOpt::Init)
        .def("GetNbChoice", &RefObjOpt::GetNbChoice)
        .def("GetChoice",   &RefObjOpt::GetChoice)
        .def("SetChoice",
             nb::overload_cast<const int>(&RefObjOpt::SetChoice))
        .def("SetChoice",
             nb::overload_cast<const std::string&>(&RefObjOpt::SetChoice))
        .def("GetName",       &RefObjOpt::GetName)
        .def("GetClassName",  &RefObjOpt::GetClassName)
        .def("GetChoiceName", &RefObjOpt::GetChoiceName)
        .def("GetClock",      &RefObjOpt::GetClock, nb::rv_policy::reference_internal)
        .def("XMLOutput",     &RefObjOpt::XMLOutput)
        .def("XMLInput",      &RefObjOpt::XMLInput)
        ;
}
