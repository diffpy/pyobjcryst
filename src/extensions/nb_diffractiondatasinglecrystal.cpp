/*
 * pyobjcryst nanobind port — DiffractionDataSingleCrystal bindings
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#undef B0
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/ObjCryst/DiffractionDataSingleCrystal.h>
#include <ObjCryst/ObjCryst/CIF.h>

#include "helpers_nb.hpp"

namespace nb = nanobind;
using namespace ObjCryst;

namespace {

DiffractionDataSingleCrystal* _CreateSingleCrystalDataFromCIF(nb::object input, Crystal& cryst)
{
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;
    std::string s = read_pyfile_to_string(input);
    std::istringstream in(s);
    ObjCryst::CIF cif(in);
    int idx0 = gDiffractionDataSingleCrystalRegistry.GetNb();
    ObjCryst::DiffractionDataSingleCrystal* d =
        ObjCryst::CreateSingleCrystalDataFromCIF(cif, &cryst);
    gag.release(); muzzle.release();
    if (gDiffractionDataSingleCrystalRegistry.GetNb() == idx0)
        throw ObjCrystException("Cannot create single crystal diffraction data from CIF");
    return d;
}

void _setIobs(DiffractionDataSingleCrystal& diff, nb::object iobs)
{
    CrystVector_REAL ii;
    assignCrystVector(ii, iobs);
    if (ii.size() != diff.GetIobs().size())
        throw ObjCrystException("DiffractionDataSingleCrystal::SetIobs(): size mismatch");
    MuteObjCrystUserInfo muzzle;
    diff.SetIobs(ii);
}

void _setSigma(DiffractionDataSingleCrystal& diff, nb::object sigma)
{
    CrystVector_REAL ss;
    assignCrystVector(ss, sigma);
    if (ss.size() != diff.GetIobs().size())
        throw ObjCrystException("DiffractionDataSingleCrystal::SetSigma(): size mismatch");
    MuteObjCrystUserInfo muzzle;
    diff.SetSigma(ss);
}

void _setHklIobs(DiffractionDataSingleCrystal& diff,
                 nb::object h, nb::object k, nb::object l,
                 nb::object iobs, nb::object sigma)
{
    CrystVector_REAL hdbl, kdbl, ldbl;
    assignCrystVector(hdbl, h);
    assignCrystVector(kdbl, k);
    assignCrystVector(ldbl, l);
    CrystVector<long> hh(hdbl.size()), kk(kdbl.size()), ll(ldbl.size());
    for (long i = 0; i < hdbl.size(); ++i) hh(i) = lround(hdbl(i));
    for (long i = 0; i < kdbl.size(); ++i) kk(i) = lround(kdbl(i));
    for (long i = 0; i < ldbl.size(); ++i) ll(i) = lround(ldbl(i));
    CrystVector_REAL iiobs, ssigma;
    assignCrystVector(iiobs, iobs);
    assignCrystVector(ssigma, sigma);
    MuteObjCrystUserInfo muzzle;
    diff.SetHklIobs(hh, kk, ll, iiobs, ssigma);
}

} // namespace

void wrap_diffractiondatasinglecrystal(nb::module_& m)
{
    m.attr("gDiffractionDataSingleCrystalRegistry") = &gDiffractionDataSingleCrystalRegistry;

    nb::class_<DiffractionDataSingleCrystal, ScatteringData>(m, "DiffractionDataSingleCrystal")
        .def(nb::init<Crystal&, const bool>(),
             nb::arg("cryst"), nb::arg("regist") = true,
             nb::keep_alive<1,2>())
        .def("GetIcalc",  [](DiffractionDataSingleCrystal& d){ return crystvec_to_array(d.GetIcalc()); })
        .def("GetIobs",   [](DiffractionDataSingleCrystal& d){ return crystvec_to_array(d.GetIobs()); })
        .def("GetSigma",  [](DiffractionDataSingleCrystal& d){ return crystvec_to_array(d.GetSigma()); })
        .def("SetIobs",   &_setIobs,   nb::arg("iobs"))
        .def("SetSigma",  &_setSigma,  nb::arg("sigma"))
        .def("SetHklIobs",&_setHklIobs,
             nb::arg("h"), nb::arg("k"), nb::arg("l"), nb::arg("iobs"), nb::arg("sigma"))
        .def("SetIobsToIcalc", &DiffractionDataSingleCrystal::SetIobsToIcalc)
        .def("GetRw",  &DiffractionDataSingleCrystal::GetRw)
        .def("GetR",   &DiffractionDataSingleCrystal::GetR)
        .def("GetChi2",&DiffractionDataSingleCrystal::GetChi2)
        .def("FitScaleFactorForRw", &DiffractionDataSingleCrystal::FitScaleFactorForRw)
        .def("FitScaleFactorForR",  &DiffractionDataSingleCrystal::FitScaleFactorForR)
        .def("PrintObsData",       &DiffractionDataSingleCrystal::PrintObsData)
        .def("PrintObsCalcData",   &DiffractionDataSingleCrystal::PrintObsCalcData)
        .def("SetUseOnlyLowAngleData",&DiffractionDataSingleCrystal::SetUseOnlyLowAngleData)
        .def("SaveHKLIobsIcalc",    &DiffractionDataSingleCrystal::SaveHKLIobsIcalc)
        .def("GetLogLikelihood",    &DiffractionDataSingleCrystal::GetLogLikelihood)
        .def("ImportHklIobs", &DiffractionDataSingleCrystal::ImportHklIobs,
             nb::arg("fileName"), nb::arg("nbRefl"), nb::arg("skipLines") = 0)
        .def("ImportHklIobsSigma", &DiffractionDataSingleCrystal::ImportHklIobsSigma,
             nb::arg("fileName"), nb::arg("nbRefl"), nb::arg("skipLines") = 0)
        .def("ImportShelxHKLF4", &DiffractionDataSingleCrystal::ImportShelxHKLF4)
        .def("ImportCIF",        &DiffractionDataSingleCrystal::ImportCIF)
        .def("SetWavelength",
             nb::overload_cast<const REAL>(&DiffractionDataSingleCrystal::SetWavelength),
             nb::arg("wavelength"))
        .def("SetWavelength",
             nb::overload_cast<const std::string&, const REAL>(
                 &DiffractionDataSingleCrystal::SetWavelength),
             nb::arg("XRayTubeElementName"), nb::arg("alpha2Alpha2ratio") = 0.5)
        .def("SetEnergy", &DiffractionDataSingleCrystal::SetEnergy, nb::arg("nrj_kev"))
        ;

    m.def("CreateSingleCrystalDataFromCIF", &_CreateSingleCrystalDataFromCIF,
          nb::arg("file"), nb::arg("crystal"),
          nb::keep_alive<0,2>(),
          nb::rv_policy::take_ownership);
}
