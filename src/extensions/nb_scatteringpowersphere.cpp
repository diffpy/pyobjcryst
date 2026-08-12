/*
 * pyobjcryst nanobind port — ScatteringPowerSphere bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ObjCryst/ObjCryst/ScatteringPowerSphere.h>

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_scatteringpowersphere(nb::module_& m)
{
    nb::class_<ScatteringPowerSphere, ScatteringPower>(m, "ScatteringPowerSphere", nb::is_final())
        .def(nb::init<>())
        .def(nb::init<const std::string&, const REAL, const REAL>(),
             nb::arg("name"), nb::arg("radius"), nb::arg("bIso") = 1.0)
        .def("Init",
             nb::overload_cast<const std::string&, const REAL, const REAL>(
                 &ScatteringPowerSphere::Init),
             nb::arg("name"), nb::arg("radius"), nb::arg("biso") = 1.0)
        .def("GetRadius", &ScatteringPowerSphere::GetRadius)
        ;
}
