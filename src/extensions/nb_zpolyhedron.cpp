#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

void wrap_zpolyhedron(nb::module_& m)
{
    nb::enum_<RegularPolyhedraType>(m, "RegularPolyhedraType")
        .value("TETRAHEDRON",              TETRAHEDRON)
        .value("OCTAHEDRON",               OCTAHEDRON)
        .value("SQUARE_PLANE",             SQUARE_PLANE)
        .value("CUBE",                     CUBE)
        .value("ANTIPRISM_TETRAGONAL",     ANTIPRISM_TETRAGONAL)
        .value("PRISM_TETRAGONAL_MONOCAP", PRISM_TETRAGONAL_MONOCAP)
        .value("PRISM_TETRAGONAL_DICAP",   PRISM_TETRAGONAL_DICAP)
        .value("PRISM_TRIGONAL",           PRISM_TRIGONAL)
        .value("PRISM_TRIGONAL_TRICAPPED", PRISM_TRIGONAL_TRICAPPED)
        .value("ICOSAHEDRON",              ICOSAHEDRON)
        .value("TRIANGLE_PLANE",           TRIANGLE_PLANE)
        .export_values()
        ;

    nb::class_<ZPolyhedron, ZScatterer>(m, "ZPolyhedron")
        .def(nb::init<const ZPolyhedron&>())
        .def(nb::init<const RegularPolyhedraType, Crystal&, double, double, double,
                      const std::string&, const ScatteringPower*, const ScatteringPower*,
                      double, double, double, double, double>(),
             nb::arg("type"), nb::arg("cryst"), nb::arg("x"), nb::arg("y"), nb::arg("z"),
             nb::arg("name"), nb::arg("centralAtomPow"), nb::arg("periphAtomPow"),
             nb::arg("centralPeriphDist"), nb::arg("ligandPopu") = 1.,
             nb::arg("phi") = 0., nb::arg("chi") = 0., nb::arg("psi") = 0.,
             nb::keep_alive<1,3>(), nb::keep_alive<1,8>(), nb::keep_alive<1,9>())
        ;
}
