/*
 * pyobjcryst nanobind port — ReflectionProfile bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>

#include <sstream>

#undef B0
#include <ObjCryst/ObjCryst/ReflectionProfile.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

struct PyReflectionProfile : ReflectionProfile {
    NB_TRAMPOLINE(ReflectionProfile, 5);

    ReflectionProfile* CreateCopy() const override {
        NB_OVERRIDE_PURE(CreateCopy);
    }
    CrystVector_REAL GetProfile(const CrystVector_REAL& x, const REAL xcenter,
                                const REAL h, const REAL k, const REAL l) const override {
        NB_OVERRIDE_PURE(GetProfile, x, xcenter, h, k, l);
    }
    REAL GetFullProfileWidth(const REAL relInt, const REAL xcenter,
                             const REAL h, const REAL k, const REAL l) override {
        NB_OVERRIDE_PURE(GetFullProfileWidth, relInt, xcenter, h, k, l);
    }
    void XMLOutput(std::ostream& os, int indent) const override {
        NB_OVERRIDE_PURE(XMLOutput, os, indent);
    }
    void XMLInput(std::istream& is, const XMLCrystTag& tag) override {
        NB_OVERRIDE_PURE(XMLInput, is, tag);
    }
};

namespace {

// Ported forward from upstream main (#79): expose the remaining public
// ReflectionProfile methods as callable, not just overridable via the
// trampoline. Previously only CreateCopy was bound, so Python code (e.g.
// PowderPatternDiffraction.GetProfile()'s return value) could not actually
// call GetProfile/GetFullProfileWidth/XMLOutput/XMLInput on a concrete
// profile object.

// GetProfile: accepts a sequence/ndarray for `x` (via the existing
// assignCrystVector helper -- nanobind has no built-in caster for the raw
// CrystVector_REAL type, so binding ReflectionProfile::GetProfile directly
// isn't an option), and converts the CrystVector_REAL result back to a
// numpy array (crystvec_to_array), matching every other Get*() binding in
// this codebase (e.g. PowderPattern::GetPowderPatternCalc).
nb_array_1d _GetProfile(const ReflectionProfile& rp, nb::object x,
                         const REAL xcenter, const REAL h, const REAL k,
                         const REAL l)
{
    CrystVector_REAL cvx;
    assignCrystVector(cvx, x);
    return crystvec_to_array(rp.GetProfile(cvx, xcenter, h, k, l));
}

void _XMLOutput(const ReflectionProfile& rp, nb::object output, int indent = 0)
{
    std::ostringstream os;
    rp.XMLOutput(os, indent);
    output.attr("write")(os.str());
}

void _XMLInput(ReflectionProfile& rp, nb::object input, const XMLCrystTag& tag)
{
    std::string s = read_pyfile_to_string(input);
    std::istringstream is(s);
    rp.XMLInput(is, tag);
}

} // namespace

void wrap_reflectionprofile(nb::module_& m)
{
    nb::class_<ReflectionProfile, RefinableObj, PyReflectionProfile>(m, "ReflectionProfile")
        .def("CreateCopy", &ReflectionProfile::CreateCopy,
             nb::rv_policy::take_ownership)
        .def("GetProfile", &_GetProfile,
             nb::arg("x"), nb::arg("xcenter"), nb::arg("h"), nb::arg("k"), nb::arg("l"))
        .def("GetFullProfileWidth",
             nb::overload_cast<const REAL, const REAL, const REAL, const REAL, const REAL>(
                 &ReflectionProfile::GetFullProfileWidth),
             nb::arg("relativeIntensity"), nb::arg("xcenter"),
             nb::arg("h"), nb::arg("k"), nb::arg("l"))
        .def("XMLOutput", &_XMLOutput, nb::arg("file"), nb::arg("indent") = 0)
        .def("XMLInput",  &_XMLInput,  nb::arg("file"), nb::arg("tag"))
        ;
}
