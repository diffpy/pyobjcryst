#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_zatom(nb::module_& m)
{
    nb::class_<ZAtom>(m, "ZAtom")
        .def("GetClassName", &ZAtom::GetClassName)
        .def("GetName", &ZAtom::GetName)
        .def("SetName", &ZAtom::SetName)
        .def("GetZScatterer", nb::overload_cast<>(&ZAtom::GetZScatterer),
             nb::rv_policy::reference_internal)
        .def("GetZBondAtom", &ZAtom::GetZBondAtom)
        .def("GetZAngleAtom", &ZAtom::GetZAngleAtom)
        .def("GetZDihedralAngleAtom", &ZAtom::GetZDihedralAngleAtom)
        .def("GetZBondLength", &ZAtom::GetZBondLength)
        .def("GetZAngle", &ZAtom::GetZAngle)
        .def("GetZDihedralAngle", &ZAtom::GetZDihedralAngle)
        .def("GetOccupancy", &ZAtom::GetOccupancy)
        .def("GetScatteringPower", &ZAtom::GetScatteringPower,
             nb::rv_policy::reference_internal)
        .def("SetZBondLength", &ZAtom::SetZBondLength)
        .def("SetZAngle", &ZAtom::SetZAngle)
        .def("SetZDihedralAngle", &ZAtom::SetZDihedralAngle)
        .def("SetOccupancy", &ZAtom::SetOccupancy)
        .def("SetScatteringPower", &ZAtom::SetScatteringPower,
             nb::keep_alive<1,2>())
        ;
}
