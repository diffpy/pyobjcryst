/*****************************************************************************
*
* pyobjcryst        Complex Modeling Initiative
*                   (c) 2015 Brookhaven Science Associates
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Vincent Favre-Nicolin, Kevin Knox
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::DiffractionDataSingleCrystal.
* Changes from ObjCryst::DiffractionDataSingleCrystal
* - SetHklIobs takes float(64) H, K, L arrays rather than long integers - easier
* because passing numpy int array seesm complicated, and more practical anyway
* since GetH() GetK() GetL() functions from ScatteringData natively use floats.
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/stl_iterator.hpp>
#undef B0

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/ObjCryst/DiffractionDataSingleCrystal.h>
#include <ObjCryst/ObjCryst/CIF.h>

#include "python_streambuf.hpp"
#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

DiffractionDataSingleCrystal* _CreateSingleCrystalDataFromCIF(bp::object input, Crystal &cryst)
{
    // Reading a cif file creates some output via fpObjCrystInformUser.
    // Mute the output and restore it on return or exception.
    // Also mute any hardcoded output to cout.
    MuteObjCrystUserInfo muzzle;
    CaptureStdOut gag;

    boost_adaptbx::python::streambuf sbuf(input);
    boost_adaptbx::python::streambuf::istream in(sbuf);
    ObjCryst::CIF cif(in);

    int idx0 = gDiffractionDataSingleCrystalRegistry.GetNb();

    ObjCryst::DiffractionDataSingleCrystal* d =
      ObjCryst::CreateSingleCrystalDataFromCIF(cif, &cryst);

    gag.release();
    muzzle.release();

    int idx = gDiffractionDataSingleCrystalRegistry.GetNb();
    if(idx == idx0)
    {
        throw ObjCryst::ObjCrystException("Cannot create single crystal diffraction data from CIF");
    }

    return d;
}

void setdiffractiondatasinglecrystal_iobs(DiffractionDataSingleCrystal& diff, bp::object iobs)
{
    CrystVector_REAL iiobs;
    assignCrystVector(iiobs, iobs);
    if(iiobs.size() != diff.GetIobs().size())
      throw ObjCryst::ObjCrystException("DiffractionDataSingleCrystal::SetIobs(): "
                                        "number of elements does not match the previous Iobs list. "
                                        "Use SetHklIobs if you want to change the number of reflections.");
    MuteObjCrystUserInfo muzzle;
    diff.SetIobs(iiobs);
}

void setdiffractiondatasinglecrystal_sigma(DiffractionDataSingleCrystal& diff, bp::object sigma)
{
    CrystVector_REAL ssigma;
    assignCrystVector(ssigma, sigma);
    if(ssigma.size() != diff.GetIobs().size())
      throw ObjCryst::ObjCrystException("DiffractionDataSingleCrystal::SetSigma(): "
                                        "number of elements does not match the Iobs list. "
                                        "Use SetHklIobs if you want to change the number of reflections.");
    MuteObjCrystUserInfo muzzle;
    diff.SetSigma(ssigma);
}

// TODO: For SetHklIobs we should pass directly an integer array but that seems difficult-passed numpy arrays
//       are always interpreted as doubles (?). It's more practical this way.
void assignCrystVector(CrystVector<long>& cv, bp::object obj)
{
    bp::stl_input_iterator<double> begin(obj), end;
    std::list<double> values(begin, end);
    cv.resize(values.size());
    std::list<double>::const_iterator vv = values.begin();
    long* dst = cv.data();
    for (; vv != values.end(); ++vv, ++dst) *dst = lround(*vv);
}

void setdiffractiondatasinglecrystal_hkliobs(DiffractionDataSingleCrystal& diff,
                                             bp::object h,bp::object k, bp::object l,
                                             bp::object iobs, bp::object sigma)
{
    CrystVector<long> hh;
    assignCrystVector(hh, h);
    CrystVector<long> kk;
    assignCrystVector(kk, k);
    CrystVector<long> ll;
    assignCrystVector(ll, l);
    CrystVector_REAL iiobs;
    assignCrystVector(iiobs, iobs);
    CrystVector_REAL ssigma;
    assignCrystVector(ssigma, sigma);

    if(hh.size() != kk.size())
      throw ObjCryst::ObjCrystException("DiffractionDataSingleCrystal::SetHklIobs(): h and k array sizes differ");
    if(hh.size() != ll.size())
      throw ObjCryst::ObjCrystException("DiffractionDataSingleCrystal::SetHklIobs(): h and l array sizes differ");
    if(hh.size() != iiobs.size())
      throw ObjCryst::ObjCrystException("DiffractionDataSingleCrystal::SetHklIobs(): h and iobs array sizes differ");
    if(hh.size() != ssigma.size())
      throw ObjCryst::ObjCrystException("DiffractionDataSingleCrystal::SetHklIobs(): h and sigma array sizes differ");

    MuteObjCrystUserInfo muzzle;
    diff.SetHklIobs(hh, kk, ll, iiobs, ssigma);
}


}   // namespace

void wrap_diffractiondatasinglecrystal()
{
    // Global object registry
    scope().attr("gDiffractionDataSingleCrystalRegistry") = object(boost::cref(gDiffractionDataSingleCrystalRegistry));

    class_<DiffractionDataSingleCrystal, bases<ScatteringData> >(
            "DiffractionDataSingleCrystal",
            init<Crystal&, const bool>((bp::arg("cryst"), bp::arg("regist")=true))
            [with_custodian_and_ward<1,2>()])
        // FIXME ... add crystal-less constructor
        .def("GetIcalc", &DiffractionDataSingleCrystal::GetIcalc,
                return_value_policy<copy_const_reference>())
        .def("GetIobs", &DiffractionDataSingleCrystal::GetIobs,
                return_value_policy<copy_const_reference>())
        .def("GetSigma", &DiffractionDataSingleCrystal::GetSigma,
                return_value_policy<copy_const_reference>())
        .def("SetIobs",
                &setdiffractiondatasinglecrystal_iobs,
                bp::arg("iobs"))
        .def("SetSigma",
                &setdiffractiondatasinglecrystal_sigma,
                bp::arg("sigma"))
        .def("SetHklIobs",
                &setdiffractiondatasinglecrystal_hkliobs,
                (bp::arg("h"),bp::arg("k"),bp::arg("l"), bp::arg("iobs"), bp::arg("sigma")))
        .def("SetIobsToIcalc", &DiffractionDataSingleCrystal::SetIobsToIcalc)
        .def("GetRw", &DiffractionDataSingleCrystal::GetRw)
        .def("GetR", &DiffractionDataSingleCrystal::GetR)
        .def("GetChi2", &DiffractionDataSingleCrystal::GetChi2)
        .def("FitScaleFactorForRw", &DiffractionDataSingleCrystal::FitScaleFactorForRw)
        .def("FitScaleFactorForR", &DiffractionDataSingleCrystal::FitScaleFactorForR)
        // TODO: These functions should print a limited number of reflections - problems otherwise
        .def("PrintObsData", &DiffractionDataSingleCrystal::PrintObsData)
        .def("PrintObsCalcData", &DiffractionDataSingleCrystal::PrintObsCalcData)
        .def("SetUseOnlyLowAngleData", &DiffractionDataSingleCrystal::SetUseOnlyLowAngleData)
        .def("SaveHKLIobsIcalc", &DiffractionDataSingleCrystal::SaveHKLIobsIcalc)
        .def("GetLogLikelihood", &DiffractionDataSingleCrystal::GetLogLikelihood)
        .def("ImportHklIobs", &DiffractionDataSingleCrystal::ImportHklIobs,
                (bp::arg("fileName"), bp::arg("nbRefl"), bp::arg("skipLines")=0))
        .def("ImportHklIobsSigma", &DiffractionDataSingleCrystal::ImportHklIobsSigma,
                (bp::arg("fileName"), bp::arg("nbRefl"), bp::arg("skipLines")=0))
        .def("ImportShelxHKLF4", &DiffractionDataSingleCrystal::ImportShelxHKLF4)
        .def("ImportCIF", &DiffractionDataSingleCrystal::ImportCIF)
        .def("SetWavelength",
                (void(DiffractionDataSingleCrystal::*)(const REAL))
                &DiffractionDataSingleCrystal::SetWavelength,
                bp::arg("wavelength"))
        .def("SetWavelength",
                (void (DiffractionDataSingleCrystal::*)(const string&, const REAL))
                &DiffractionDataSingleCrystal::SetWavelength,
                (bp::arg("XRayTubeElementName"), bp::arg("alpha2Alpha2ratio")=0.5))
        .def("SetEnergy", &DiffractionDataSingleCrystal::SetEnergy,
                bp::arg("nrj_kev"))
        ;
    def("CreateSingleCrystalDataFromCIF",
            &_CreateSingleCrystalDataFromCIF, (bp::arg("file"), bp::arg("crystal")),
            with_custodian_and_ward_postcall<0,2,
            return_value_policy<manage_new_object> >());
}
