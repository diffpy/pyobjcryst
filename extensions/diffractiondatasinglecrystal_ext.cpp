/*****************************************************************************
*
* pyobjcryst
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::MonteCarloObj.
*
* Changes from ObjCryst::MonteCarloObj
*
* Other Changes
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/slice.hpp>

#include <string>
#include <map>

#include <ObjCryst/ObjCryst/General.h>
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
    class_<DiffractionDataSingleCrystal, bases<RefinableObj> >(
            "DiffractionDataSingleCrystal",
            init<Crystal&, const bool>((bp::arg("cryst"), bp::arg("regist")=true))
            [with_custodian_and_ward<1,2>()])
        .def("GetIcalc", &DiffractionDataSingleCrystal::GetIcalc,
                return_value_policy<copy_const_reference>())
        .def("GetIobs", &DiffractionDataSingleCrystal::GetIobs,
                return_value_policy<copy_const_reference>())
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
        // Functions from ScatteringData
        .def("GenHKLFullSpace", &ScatteringData::GenHKLFullSpace,
                (bp::arg("maxTheta"), bp::arg("unique")=false))
        .def("GetNbRefl", &ScatteringData::GetNbRefl)
        .def("GetH", &ScatteringData::GetH,
                return_value_policy<copy_const_reference>())
        .def("GetK", &ScatteringData::GetK,
                return_value_policy<copy_const_reference>())
        .def("GetL", &ScatteringData::GetL,
                return_value_policy<copy_const_reference>())
        .def("GetReflX", &ScatteringData::GetReflX,
                return_value_policy<copy_const_reference>())
        .def("GetReflY", &ScatteringData::GetReflY,
                return_value_policy<copy_const_reference>())
        .def("GetReflZ", &ScatteringData::GetReflZ,
                return_value_policy<copy_const_reference>())
        .def("GetSinThetaOverLambda", &ScatteringData::GetSinThetaOverLambda,
                return_value_policy<copy_const_reference>())
        .def("GetTheta", &ScatteringData::GetTheta,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcSq", &ScatteringData::GetFhklCalcSq,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcReal", &ScatteringData::GetFhklCalcReal,
                return_value_policy<copy_const_reference>())
        .def("GetFhklCalcImag", &ScatteringData::GetFhklCalcImag,
                return_value_policy<copy_const_reference>())
        .def("GetFhklObsSq", &ScatteringData::GetFhklObsSq,
                return_value_policy<copy_const_reference>())
        .def("GetWavelength", &ScatteringData::GetWavelength)
        ;
}
