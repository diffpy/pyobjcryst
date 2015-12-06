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
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/ScatteringData.h>
#include <ObjCryst/ObjCryst/DiffractionDataSingleCrystal.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

}   // namespace

void wrap_diffractiondatasinglecrystal()
{
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
}
