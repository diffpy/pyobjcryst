/*****************************************************************************
 *
 * pyobjcryst
 *
 * File coded by:    Vincent Favre-Nicolin
 *
 * See AUTHORS.txt for a list of people who contributed.
 * See LICENSE.txt for license information.
 *
 ******************************************************************************
 *
 * boost::python bindings to ObjCryst::ReflectionProfile.
 *
 * Changes from ObjCryst::ReflectionProfile
 *
 * Other Changes
 *
 *****************************************************************************/

#define BOOST_PYTHON_MAX_ARITY 20

#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/pure_virtual.hpp>
#undef B0

#include "helpers.hpp" // assignCrystVector helper for numpy/sequence inputs

#include <iostream>

#include <ObjCryst/ObjCryst/ReflectionProfile.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace
{

    class ReflectionProfileWrap : public ReflectionProfile, public wrapper<ReflectionProfile>
    {
    public:
        // Pure virtual functions

        ReflectionProfile *CreateCopy() const
        {
            return this->get_override("CreateCopy")();
        }

        CrystVector_REAL GetProfile(
            const CrystVector_REAL &x, const REAL xcenter,
            const REAL h, const REAL k, const REAL l) const
        {
            bp::override f = this->get_override("GetProfile");
            return f(x, xcenter, h, k, l);
        }

        REAL GetFullProfileWidth(
            const REAL relativeIntensity, const REAL xcenter,
            const REAL h, const REAL k, const REAL l)
        {
            bp::override f = this->get_override("GetFullProfileWidth");
            return f(relativeIntensity, xcenter, h, k, l);
        }

        void XMLOutput(ostream &os, int indent) const
        {
            bp::override f = this->get_override("XMLOutput");
            f(os, indent);
        }

        void XMLInput(istream &is, const XMLCrystTag &tag)
        {
            bp::override f = this->get_override("XMLInput");
            f(is, tag);
        }
    };

    // Accept python sequences/ndarrays for x and forward to the C++ API.
    CrystVector_REAL _GetProfile(
        const ReflectionProfile &rp, bp::object x, const REAL xcenter,
        const REAL h, const REAL k, const REAL l)
    {
        CrystVector_REAL cvx;
        assignCrystVector(cvx, x);
        return rp.GetProfile(cvx, xcenter, h, k, l);
    }

} // namespace

void wrap_reflectionprofile()
{
    class_<ReflectionProfileWrap, bases<RefinableObj>, boost::noncopyable>(
        "ReflectionProfile")
        .def("CreateCopy",
             pure_virtual(&ReflectionProfile::CreateCopy),
             (return_value_policy<manage_new_object>()),
             "Return a new ReflectionProfile instance copied from this one.")
        // Two overloads for GetProfile:
        // - Native CrystVector signature (for C++ callers / already-converted vectors).
        // - Python-friendly wrapper that accepts sequences/ndarrays and converts them.
        .def(
            "GetProfile",
             pure_virtual((CrystVector_REAL (ReflectionProfile::*)(const CrystVector_REAL &, REAL, REAL, REAL, REAL) const) & ReflectionProfile::GetProfile),
            (bp::arg("x"), bp::arg("xcenter"), bp::arg("h"),
             bp::arg("k"), bp::arg("l")),
             "Compute the profile values at positions `x` for reflection (h, k, l) centered at `xcenter`.")
        .def(
            "GetProfile", &_GetProfile,
            (bp::arg("x"), bp::arg("xcenter"), bp::arg("h"), bp::arg("k"),
             bp::arg("l")),
             "Compute the profile values at positions `x` (sequence/ndarray accepted) for reflection (h, k, l) centered at `xcenter`.")
        .def("GetFullProfileWidth",
             pure_virtual((REAL (ReflectionProfile::*)(const REAL, const REAL, const REAL, const REAL, const REAL) const) & ReflectionProfile::GetFullProfileWidth),
             (bp::arg("relativeIntensity"), bp::arg("xcenter"),
              bp::arg("h"), bp::arg("k"), bp::arg("l")),
              "Return the full profile width at a given relative intensity for reflection (h, k, l) around `xcenter`.")
        .def("IsAnisotropic", &ReflectionProfile::IsAnisotropic,
             "Return whether the profile depends on the reflection indices.")
        .def("XMLOutput",
             pure_virtual((void (ReflectionProfile::*)(ostream &, int) const) & ReflectionProfile::XMLOutput),
             (bp::arg("os"), bp::arg("indent")),
             "Write this ReflectionProfile as XML to a file-like object. `indent` controls indentation depth.")
        .def("XMLInput",
             pure_virtual((void (ReflectionProfile::*)(istream &, const XMLCrystTag &))&ReflectionProfile::XMLInput),
             (bp::arg("is"), bp::arg("tag")),
             "Load ReflectionProfile parameters from an XML stream and tag.");

    class_<ReflectionProfilePseudoVoigt, bases<ReflectionProfile> >(
        "ReflectionProfilePseudoVoigt", init<>())
        .def(init<const ReflectionProfilePseudoVoigt &>(bp::arg("old")))
        .def("CreateCopy", &ReflectionProfilePseudoVoigt::CreateCopy,
             return_value_policy<manage_new_object>(),
             "Return an independent copy of this profile.")
        .def("SetProfilePar", &ReflectionProfilePseudoVoigt::SetProfilePar,
             (bp::arg("fwhmCagliotiW"), bp::arg("fwhmCagliotiU") = 0,
              bp::arg("fwhmCagliotiV") = 0, bp::arg("eta0") = 0.5,
              bp::arg("eta1") = 0),
             "Set the isotropic pseudo-Voigt profile parameters.");

    class_<ReflectionProfilePseudoVoigtTCH, bases<ReflectionProfile> >(
        "ReflectionProfilePseudoVoigtTCH", init<>())
        .def(init<const ReflectionProfilePseudoVoigtTCH &>(bp::arg("old")))
        .def("CreateCopy", &ReflectionProfilePseudoVoigtTCH::CreateCopy,
             return_value_policy<manage_new_object>(),
             "Return an independent copy of this profile.")
        .def("SetProfilePar",
             (void (ReflectionProfilePseudoVoigtTCH::*)(
                 const REAL, const REAL, const REAL, const REAL,
                 const REAL, const REAL, const REAL, const REAL))
                 &ReflectionProfilePseudoVoigtTCH::SetProfilePar,
             (bp::arg("fwhmCagliotiW"), bp::arg("fwhmCagliotiU") = 0,
              bp::arg("fwhmCagliotiV") = 0,
              bp::arg("fwhmLorentzX") = 0,
              bp::arg("fwhmLorentzY") = 0,
              bp::arg("fwhmLorentzZ") = 0,
              bp::arg("fwhmScherrerP") = 0,
              bp::arg("scherrerLGmix") = 1),
             "Set the TCH pseudo-Voigt U, V, W, X, Y, Z, P and LGmix parameters.");

    class_<ReflectionProfilePseudoVoigtAnisotropic,
           bases<ReflectionProfile> >(
        "ReflectionProfilePseudoVoigtAnisotropic", init<>())
        .def(init<const ReflectionProfilePseudoVoigtAnisotropic &>(
            bp::arg("old")))
        .def("CreateCopy",
             &ReflectionProfilePseudoVoigtAnisotropic::CreateCopy,
             return_value_policy<manage_new_object>(),
             "Return an independent copy of this profile.")
        .def(
            "SetProfilePar",
            &ReflectionProfilePseudoVoigtAnisotropic::SetProfilePar,
            (bp::arg("fwhmCagliotiW"), bp::arg("fwhmCagliotiU") = 0,
             bp::arg("fwhmCagliotiV") = 0, bp::arg("fwhmGaussP") = 0,
             bp::arg("fwhmLorentzX") = 0,
             bp::arg("fwhmLorentzY") = 0,
             bp::arg("fwhmLorentzGammaHH") = 0,
             bp::arg("fwhmLorentzGammaKK") = 0,
             bp::arg("fwhmLorentzGammaLL") = 0,
             bp::arg("fwhmLorentzGammaHK") = 0,
             bp::arg("fwhmLorentzGammaHL") = 0,
             bp::arg("fwhmLorentzGammaKL") = 0,
             bp::arg("pseudoVoigtEta0") = 0,
             bp::arg("pseudoVoigtEta1") = 0,
             bp::arg("asymA0") = 1, bp::arg("asymA1") = 0,
             bp::arg("asymA2") = 0),
            "Set the anisotropic pseudo-Voigt profile parameters.");
}
