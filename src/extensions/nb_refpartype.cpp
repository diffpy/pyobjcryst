/*
 * pyobjcryst nanobind port — RefParType bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/RefinableObj/RefinableObj.h>

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_refpartype(nb::module_& m)
{
    nb::class_<RefParType>(m, "RefParType")
        .def(nb::init<const std::string&>(), nb::arg("name"))
        .def(nb::init<const RefParType*, const std::string&>(),
             nb::arg("parent"), nb::arg("name"),
             nb::keep_alive<1, 2>())
        .def("IsDescendantFromOrSameAs", &RefParType::IsDescendantFromOrSameAs)
        .def("GetName", &RefParType::GetName)
        .def("__eq__", [](const RefParType* a, const RefParType* b){ return a == b; })
        ;

    // Expose the global root RefParType
    m.attr("gpRefParTypeObjCryst") = gpRefParTypeObjCryst;
}
