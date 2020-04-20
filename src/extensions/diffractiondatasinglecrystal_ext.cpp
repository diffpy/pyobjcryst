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
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/manage_new_object.hpp>
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
        // FIXME ... add SetIobs, SetSigma ....
        .def("SetIobsToIcalc", &DiffractionDataSingleCrystal::SetIobsToIcalc)
        .def("GetRw", &DiffractionDataSingleCrystal::GetRw)
        .def("GetR", &DiffractionDataSingleCrystal::GetR)
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
