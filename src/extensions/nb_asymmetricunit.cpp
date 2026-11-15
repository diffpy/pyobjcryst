/*
 * pyobjcryst nanobind port — AsymmetricUnit bindings
 */

#include <nanobind/nanobind.h>

#include <ObjCryst/ObjCryst/SpaceGroup.h>

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_asymmetricunit(nb::module_& m)
{
    nb::class_<AsymmetricUnit>(m, "AsymmetricUnit")
        .def(nb::init<>())
        .def(nb::init<const SpaceGroup&>(), nb::arg("spg"))
        .def("SetSpaceGroup", &AsymmetricUnit::SetSpaceGroup, nb::arg("spg"))
        .def("IsInAsymmetricUnit", &AsymmetricUnit::IsInAsymmetricUnit,
             nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def("Xmin", &AsymmetricUnit::Xmin)
        .def("Xmax", &AsymmetricUnit::Xmax)
        .def("Ymin", &AsymmetricUnit::Ymin)
        .def("Ymax", &AsymmetricUnit::Ymax)
        .def("Zmin", &AsymmetricUnit::Zmin)
        .def("Zmax", &AsymmetricUnit::Zmax)
        ;
}
