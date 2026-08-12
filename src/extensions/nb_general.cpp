/*
 * pyobjcryst nanobind port — general bindings
 * Wraps RadiationType, WavelengthType enums and libobjcryst version info
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

#include <ObjCryst/version.h>
#include <ObjCryst/ObjCryst/General.h>
#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

// getTestVector/getTestMatrix -- ported from the legacy Boost.Python
// registerconverters.cpp (`def("getTestVector", ...)` / `def("getTestMatrix",
// ...)`). Exercised only by tests/test_converters.py, which verifies the
// CrystVector<->ndarray / CrystMatrix<->ndarray conversion path via a known
// fixed result, independent of any real ObjCryst++ computation. Uses
// CrystVector_REAL/CrystMatrix_REAL (not a hardcoded <double>) and the shared
// crystvec_to_array/crystmat_to_array helpers, per this project's REAL-macro
// convention (see helpers_nb.hpp).
nb_array_1d getTestVector()
{
    // Should produce [0, 1, 2]
    CrystVector_REAL tv(3);
    for (int i = 0; i < 3; ++i) tv(i) = i;
    return crystvec_to_array(tv);
}

nb_array_2d getTestMatrix()
{
    // Should produce [[0, 1], [2, 3], [4, 5]]
    CrystMatrix_REAL tm(3, 2);
    int counter = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            tm(i, j) = counter++;
    return crystmat_to_array(tm);
}

nb::dict get_libobjcryst_version_info_dict()
{
    nb::dict rv;
    rv["version"]     = libobjcryst_version_info::version;
    rv["version_str"] = std::string(libobjcryst_version_info::version_str);
    rv["major"]       = libobjcryst_version_info::major;
    rv["minor"]       = libobjcryst_version_info::minor;
    rv["micro"]       = libobjcryst_version_info::micro;
    rv["date"]        = std::string(libobjcryst_version_info::date);
    rv["git_commit"]  = std::string(libobjcryst_version_info::git_commit);
    rv["patch"]       = libobjcryst_version_info::patch;
    return rv;
}

} // namespace

void wrap_general(nb::module_& m)
{
    nb::enum_<RadiationType>(m, "RadiationType")
        .value("RAD_NEUTRON",  RAD_NEUTRON)
        .value("RAD_XRAY",     RAD_XRAY)
        .value("RAD_ELECTRON", RAD_ELECTRON)
        .export_values()
        ;

    nb::enum_<WavelengthType>(m, "WavelengthType")
        .value("WAVELENGTH_MONOCHROMATIC", WAVELENGTH_MONOCHROMATIC)
        .value("WAVELENGTH_ALPHA12",       WAVELENGTH_ALPHA12)
        .value("WAVELENGTH_TOF",           WAVELENGTH_TOF)
        .export_values()
        ;

    m.def("_get_libobjcryst_version_info_dict", &get_libobjcryst_version_info_dict,
          "Return dictionary with version data for the loaded libobjcryst library.");

    // Diagnostic test for nb::object + None
    m.def("_test_handle_none", [](nb::object x) -> bool {
        return x.is_none();
    }, nb::arg("x") = nb::none());

    // Used only by tests/test_converters.py -- see the comment at the
    // definitions above.
    m.def("getTestVector", &getTestVector);
    m.def("getTestMatrix", &getTestMatrix);
}
