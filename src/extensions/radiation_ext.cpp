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
* boost::python bindings to ObjCryst::Radiation.
*
* Changes from ObjCryst::Radiation
* - GetWavelength returns a scalar instead of a vector
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <iostream>

#include <ObjCryst/ObjCryst/ScatteringData.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace{

double _GetWavelength(Radiation& r)
{
    return r.GetWavelength()(0);
}

}   // anonymous namespace

void wrap_radiation()
{
    class_<Radiation, bases<RefinableObj>, boost::noncopyable>("Radiation")
        .def("SetRadiationType", &Radiation::SetRadiationType)
        .def("GetRadiationType", &Radiation::GetRadiationType)
        .def("SetWavelengthType", &Radiation::SetWavelengthType)
        .def("GetWavelengthType", &Radiation::GetWavelengthType)
        // Overloaded to return a single wavelength instead of a vector
        // .def("GetWavelength", &Radiation::GetWavelength, return_value_policy<copy_const_reference>())
        .def("GetWavelength", &_GetWavelength)
        .def("SetWavelength", (void (Radiation::*)(const REAL)) &Radiation::SetWavelength)
        .def("SetWavelength", (void (Radiation::*)(const std::string &,const REAL)) &Radiation::SetWavelength,
        (bp::arg("XRayTubeElementName"), bp::arg("alpha2Alpha2ratio")=0.5))
        .def("GetXRayTubeDeltaLambda", & Radiation::GetXRayTubeDeltaLambda)
        .def("GetXRayTubeAlpha2Alpha1Ratio", & Radiation::GetXRayTubeAlpha2Alpha1Ratio)
        ;
}
