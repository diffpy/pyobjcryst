/*
 * pyobjcryst nanobind port — PowderPattern, SPGScore, SpaceGroupExplorer bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>

#include <string>
#include <sstream>
#include <map>
#include <memory>

#undef B0
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/ObjCryst/CIF.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

struct PyPowderPattern : PowderPattern {
    NB_TRAMPOLINE(PowderPattern, 1);
    void UpdateDisplay() const override { NB_OVERRIDE(UpdateDisplay); }
};

namespace {

PowderPattern* _CreatePowderPatternFromCIF_file(nb::object input)
{
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;
    std::string s = read_pyfile_to_string(input);
    std::istringstream in(s);
    ObjCryst::CIF cif(in);
    int idx0 = gPowderPatternRegistry.GetNb();
    ObjCryst::PowderPattern* p = ObjCryst::CreatePowderPatternFromCIF(cif);
    gag.release(); muzzle.release();
    if (gPowderPatternRegistry.GetNb() == idx0)
        throw ObjCrystException("Cannot create powder pattern from CIF");
    return p;
}

PowderPattern* _CreatePowderPatternFromCIF_obj(nb::object input, PowderPattern& pp)
{
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;
    std::string s = read_pyfile_to_string(input);
    std::istringstream in(s);
    ObjCryst::CIF cif(in);
    bool import_ok = false;
    for (auto& pos : cif.mvData) {
        if (pos.second.mPowderPatternObs.size() > 10) {
            pp.ImportPowderPatternCIF(cif);
            import_ok = true;
            break;
        }
    }
    gag.release(); muzzle.release();
    if (!import_ok)
        throw ObjCrystException("Cannot create powder pattern from CIF");
    return &pp;
}

PowderPatternBackground& _addppbackground(PowderPattern& pp)
{
    PowderPatternBackground* ppc = new PowderPatternBackground();
    pp.AddPowderPatternComponent(*ppc);
    return *ppc;
}

// Ported forward from upstream main's fix (#94, src/extensions/powderpattern_ext.cpp
// commit 69e6239): guard against a crash when the new PowderPatternDiffraction has
// no reflections in the target 2theta window. Prepare against the target context
// before final registration so a no-reflections failure can't leave a broken
// partially-attached component behind, and roll back cleanly on any other
// Prepare() failure too.
PowderPatternDiffraction& _addppdiffraction(PowderPattern& pp, Crystal& crst)
{
    std::unique_ptr<PowderPatternDiffraction> ppc(new PowderPatternDiffraction());
    ppc->SetCrystal(crst);
    ppc->SetParentPowderPattern(pp);
    ppc->GenHKLFullSpace();
    if (ppc->GetNbReflBelowMaxSinThetaOvLambda() == 0)
    {
        throw ObjCrystException(
            "PowderPatternDiffraction::CalcSinThetaLambda(): there are no reflections!"
        );
    }
    pp.AddPowderPatternComponent(*ppc);
    try
    {
        pp.Prepare();
    }
    catch (...)
    {
        pp.RemovePowderPatternComponent(*ppc);
        throw;
    }
    return *ppc.release();
}

void _setX(PowderPattern& pp, nb::object x)
{
    CrystVector_REAL cvx;
    assignCrystVector(cvx, x);
    pp.SetPowderPatternX(cvx);
}

void _setObs(PowderPattern& pp, nb::object x)
{
    CrystVector_REAL cvx;
    assignCrystVector(cvx, x);
    MuteObjCrystUserInfo muzzle;
    pp.SetPowderPatternObs(cvx);
}

// Ported forward from upstream main (#84): GetPowderPatternObsSigma/
// SetPowderPatternObsSigma were added after this branch's original fork.
void _setObsSigma(PowderPattern& pp, nb::object x)
{
    CrystVector_REAL cvx;
    assignCrystVector(cvx, x);
    pp.SetPowderPatternObsSigma(cvx);
}

nb::object _FindPeaks(PowderPattern& pp, const float dmin, const float maxratio,
                      const unsigned int maxpeak, const bool verbose)
{
    CaptureStdOut gag;
    if (verbose) gag.release();
    // PowderPattern::FindPeaks() returns a plain ObjCryst::PeakList by value,
    // but nanobind only has a registered Python type for PeakListNB (see
    // nb_indexing.cpp) -- returning PeakList directly fails with "Unable to
    // convert function return value to a Python type!". wrap_peaklist()
    // (declared in helpers_nb.hpp, defined in nb_indexing.cpp) does the
    // PeakList -> PeakListNB -> Python conversion.
    return wrap_peaklist(pp.FindPeaks(dmin, maxratio, maxpeak));
}

std::string __str__SPGScore(SPGScore& s)
{
    std::ostringstream oss;
    if (s.ngof > 0.0001)
        oss << s.hm << " nGoF=" << s.ngof << " GoF=" << s.gof << " Rw=" << s.rw;
    else
        oss << s.hm << " GoF=" << s.gof << " Rw=" << s.rw;
    return oss.str();
}

nb::list _GetScores(const SpaceGroupExplorer& spgex)
{
    return containerToPyList(spgex.GetScores());
}

// PowderPattern::GetPowderPatternComponent() returns a generic
// PowderPatternComponent&, but the dynamic type is always one of the leaf
// classes (PowderPatternBackground, PowderPatternDiffraction, ...).
// PowderPatternComponent is itself a virtual base of RefinableObj, AND
// PowderPatternDiffraction inherits it virtually too -- nanobind's
// automatic polymorphic-return downcast relies on a base-offset table built
// from declared nb::class_<Derived, Base> relationships, which is exactly
// wrong for virtual inheritance (§1 in nanobind_migration_notes.md).
// Concretely: PowderPatternDiffraction deliberately does NOT declare
// PowderPatternComponent as its nanobind base (see nb_powderpatterndiffraction.cpp),
// so nanobind has no offset path from PowderPatternComponent to
// PowderPatternDiffraction at all -- returning the base reference directly
// produced a different (and, once accessed, crashing) Python object than
// the one originally created by AddPowderPatternDiffraction(). Fix: resolve
// the concrete type by hand with a real dynamic_cast (RTTI, virtual-base
// safe) before handing nanobind an exact-type pointer, exactly like
// Crystal::GetScatteringPower()'s fix in nb_crystal.cpp.
nb::object wrapPowderPatternComponent(PowderPatternComponent& c, nb::handle parent)
{
    if (auto* d = dynamic_cast<PowderPatternDiffraction*>(&c))
        return nb::cast(d, nb::rv_policy::reference_internal, parent);
    if (auto* b = dynamic_cast<PowderPatternBackground*>(&c))
        return nb::cast(b, nb::rv_policy::reference_internal, parent);
    return nb::cast(&c, nb::rv_policy::reference_internal, parent);
}

// Argument-direction mirror of the above, for GetScaleFactor(const
// PowderPatternComponent&) and friends: a Python PowderPatternDiffraction
// no longer declares PowderPatternComponent as a nanobind base (see above),
// so passing one where a plain PowderPatternComponent& is expected has no
// declared relationship for nanobind to use. Try the concrete leaf types
// first (exact match, no offset needed), falling back to the base.
PowderPatternComponent& extractPowderPatternComponentArg(nb::object obj)
{
    try { return nb::cast<PowderPatternDiffraction&>(obj); } catch (...) {}
    try { return nb::cast<PowderPatternBackground&>(obj); } catch (...) {}
    return nb::cast<PowderPatternComponent&>(obj);
}

} // namespace

void wrap_powderpattern(nb::module_& m)
{
    m.attr("gPowderPatternRegistry") = &gPowderPatternRegistry;

    nb::class_<PowderPattern, RefinableObj, PyPowderPattern>(m, "PowderPattern")
        .def(nb::init<>())
        .def("AddPowderPatternBackground", &_addppbackground,
             nb::rv_policy::reference_internal)
        .def("AddPowderPatternDiffraction", &_addppdiffraction,
             nb::keep_alive<1,2>(), nb::rv_policy::reference_internal)
        .def("GetNbPowderPatternComponent", &PowderPattern::GetNbPowderPatternComponent)
        .def("GetPowderPatternComponent",
             [](nb::handle self, const int i) -> nb::object {
                 PowderPattern& pp = nb::cast<PowderPattern&>(self);
                 return wrapPowderPatternComponent(pp.GetPowderPatternComponent(i), self);
             })
        .def("FindPeaks", &_FindPeaks,
             nb::arg("dmin") = 2.0f, nb::arg("maxratio") = 0.01f,
             nb::arg("maxpeak") = 100u, nb::arg("verbose") = false)
        .def("GetScaleFactor",
             nb::overload_cast<const int>(&PowderPattern::GetScaleFactor, nb::const_))
        .def("GetScaleFactor",
             [](PowderPattern& pp, nb::object comp) {
                 return pp.GetScaleFactor(extractPowderPatternComponentArg(comp));
             })
        .def("SetScaleFactor",
             nb::overload_cast<const int, REAL>(&PowderPattern::SetScaleFactor))
        .def("SetPowderPatternPar", &PowderPattern::SetPowderPatternPar,
             nb::arg("xmin"), nb::arg("xstep"), nb::arg("nbpoints"))
        .def("SetPowderPatternX",   &_setX, nb::arg("x"))
        .def("GetPowderPatternCalc",
             [](PowderPattern& p){ return crystvec_to_array(p.GetPowderPatternCalc()); })
        .def("GetPowderPatternObs",
             [](PowderPattern& p){ return crystvec_to_array(p.GetPowderPatternObs()); })
        .def("GetPowderPatternObsSigma",
             [](PowderPattern& p){ return crystvec_to_array(p.GetPowderPatternObsSigma()); })
        .def("GetPowderPatternX",
             [](PowderPattern& p){ return crystvec_to_array(p.GetPowderPatternX()); })
        .def("GetNbPoint",     &PowderPattern::GetNbPoint)
        .def("GetNbPointUsed", &PowderPattern::GetNbPoint)
        .def("GetRadiation",
             nb::overload_cast<>(&PowderPattern::GetRadiation),
             nb::rv_policy::reference_internal)
        .def("GetRadiationType", &PowderPattern::GetRadiationType)
        .def("SetRadiationType", &PowderPattern::SetRadiationType)
        .def("GetWavelength",    &PowderPattern::GetWavelength)
        .def("SetWavelength",
             nb::overload_cast<const REAL>(&PowderPattern::SetWavelength),
             nb::arg("wavelength"))
        .def("SetWavelength",
             nb::overload_cast<const std::string&, const REAL>(&PowderPattern::SetWavelength),
             nb::arg("XRayTubeElementName"), nb::arg("alpha2Alpha2ratio") = 0.5)
        .def("SetEnergy", [](PowderPattern& pp, REAL energy_keV) {
             // 12.398 angstrom*keV = hc
             pp.SetWavelength(static_cast<REAL>(12.398 / energy_keV));
         }, nb::arg("nrj_kev"))
        .def("ImportPowderPatternFullprof",      &PowderPattern::ImportPowderPatternFullprof, nb::arg("filename"))
        .def("ImportPowderPatternPSI_DMC",       &PowderPattern::ImportPowderPatternPSI_DMC, nb::arg("filename"))
        .def("ImportPowderPatternILL_D1A5",      &PowderPattern::ImportPowderPatternILL_D1A5, nb::arg("filename"))
        .def("ImportPowderPatternXdd",           &PowderPattern::ImportPowderPatternXdd, nb::arg("filename"))
        .def("ImportPowderPatternSietronicsCPI",  &PowderPattern::ImportPowderPatternSietronicsCPI, nb::arg("filename"))
        .def("ImportPowderPattern2ThetaObsSigma",&PowderPattern::ImportPowderPattern2ThetaObsSigma,
             nb::arg("filename"), nb::arg("nbSkip") = 0)
        .def("ImportPowderPatternFullprof4",     &PowderPattern::ImportPowderPatternFullprof4, nb::arg("filename"))
        .def("ImportPowderPatternMultiDetectorLLBG42",&PowderPattern::ImportPowderPatternMultiDetectorLLBG42, nb::arg("filename"))
        .def("ImportPowderPattern2ThetaObs",     &PowderPattern::ImportPowderPattern2ThetaObs,
             nb::arg("filename"), nb::arg("nbSkip") = 0)
        .def("ImportPowderPatternTOF_ISIS_XYSigma",&PowderPattern::ImportPowderPatternTOF_ISIS_XYSigma, nb::arg("filename"))
        .def("ImportPowderPatternGSAS",          &PowderPattern::ImportPowderPatternGSAS, nb::arg("filename"))
        .def("SetPowderPatternObs", &_setObs, nb::arg("obs"))
        .def("SetPowderPatternObsSigma", &_setObsSigma, nb::arg("sigma"))
        .def("FitScaleFactorForR",              &PowderPattern::FitScaleFactorForR)
        .def("FitScaleFactorForIntegratedR",    &PowderPattern::FitScaleFactorForIntegratedR)
        .def("FitScaleFactorForRw",             &PowderPattern::FitScaleFactorForRw)
        .def("FitScaleFactorForIntegratedRw",   &PowderPattern::FitScaleFactorForIntegratedRw)
        .def("SetMaxSinThetaOvLambda", &PowderPattern::SetMaxSinThetaOvLambda, nb::arg("max"))
        .def("GetMaxSinThetaOvLambda", &PowderPattern::GetMaxSinThetaOvLambda)
        .def("X2XCorr",     &PowderPattern::X2XCorr)
        .def("X2PixelCorr", &PowderPattern::X2PixelCorr)
        .def("X2Pixel",     &PowderPattern::X2Pixel)
        .def("STOL2X",      &PowderPattern::STOL2X)
        .def("X2STOL",      &PowderPattern::X2STOL)
        .def("STOL2Pixel",  &PowderPattern::STOL2Pixel)
        .def("UpdateDisplay",&PowderPattern::UpdateDisplay)
        .def("GetR",              &PowderPattern::GetR)
        .def("GetIntegratedR",    &PowderPattern::GetIntegratedR)
        .def("GetRw",             &PowderPattern::GetRw)
        .def("GetIntegratedRw",   &PowderPattern::GetIntegratedRw)
        .def("GetChi2",           &PowderPattern::GetChi2)
        .def("GetIntegratedChi2", &PowderPattern::GetIntegratedChi2)
        .def("GetLogLikelihood",  &PowderPattern::GetLogLikelihood)
        .def("Prepare",           &PowderPattern::Prepare)
        .def_prop_rw("mur",    &PowderPattern::GetMuR, &PowderPattern::SetMuR)
        .def_prop_ro("r",             &PowderPattern::GetR)
        .def_prop_ro("r_integrated",  &PowderPattern::GetIntegratedR)
        .def_prop_ro("rw",            &PowderPattern::GetRw)
        .def_prop_ro("rw_integrated", &PowderPattern::GetIntegratedRw)
        .def_prop_ro("chi2",          &PowderPattern::GetChi2)
        .def_prop_ro("chi2_integrated",&PowderPattern::GetIntegratedChi2)
        .def_prop_ro("llk",           &PowderPattern::GetLogLikelihood)
        .def_prop_rw("wavelength",
             &PowderPattern::GetWavelength,
             nb::overload_cast<const REAL>(&PowderPattern::SetWavelength))
        ;

    nb::class_<SPGScore>(m, "SPGScore")
        .def(nb::init<const std::string&, const REAL, const REAL, const unsigned int, const REAL>(),
             nb::arg("hermann_mauguin"), nb::arg("rw"), nb::arg("gof"),
             nb::arg("nbextinct"), nb::arg("ngof") = 0)
        .def_ro("hermann_mauguin", &SPGScore::hm)
        .def_ro("Rw",  &SPGScore::rw)
        .def_ro("GoF", &SPGScore::gof)
        .def_ro("nGoF",&SPGScore::ngof)
        .def("__str__",  &__str__SPGScore)
        .def("__repr__", &__str__SPGScore)
        ;

    nb::class_<SpaceGroupExplorer>(m, "SpaceGroupExplorer")
        .def(nb::init<PowderPatternDiffraction*>(),
             nb::arg("powdiff"), nb::keep_alive<1,2>())
        .def("Run",
             nb::overload_cast<const std::string&, const bool, const bool, const bool,
                               const bool, const REAL, const REAL>(
                 &SpaceGroupExplorer::Run),
             nb::arg("spg"), nb::arg("fitprofile") = false, nb::arg("verbose") = false,
             nb::arg("restore_orig") = false, nb::arg("update_display") = false,
             nb::arg("relative_length_tolerance") = (REAL)0.01,
             nb::arg("absolute_angle_tolerance_degree") = (REAL)0.5)
        .def("RunAll", &SpaceGroupExplorer::RunAll,
             nb::arg("fitprofile_all") = false, nb::arg("verbose") = true,
             nb::arg("keep_best") = true, nb::arg("update_display") = true,
             nb::arg("fitprofile_p1") = true,
             nb::arg("relative_length_tolerance") = 0.01,
             nb::arg("absolute_angle_tolerance_degree") = 1)
        .def("GetScores", &_GetScores)
        ;

    m.def("CreatePowderPatternFromCIF",
          [](nb::object input) { return _CreatePowderPatternFromCIF_file(input); },
          nb::arg("file"), nb::rv_policy::take_ownership);
    m.def("CreatePowderPatternFromCIF",
          [](nb::object input, PowderPattern& pp) { return _CreatePowderPatternFromCIF_obj(input, pp); },
          nb::arg("file"), nb::arg("powpat"), nb::rv_policy::take_ownership);
}
