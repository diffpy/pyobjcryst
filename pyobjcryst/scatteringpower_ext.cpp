/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::ScatteringPower.
*
* $Id$
*
*****************************************************************************/

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include "ObjCryst/ScatteringPower.h"
#include "RefinableObj/RefinableObj.h"
#include "CrystVector/CrystVector.h"

using namespace boost::python;
using namespace ObjCryst;

namespace {

class ScatteringPowerWrap : public ScatteringPower, 
                 public wrapper<ScatteringPower>
{

    public: 

    ScatteringPowerWrap() : ScatteringPower() {}
    ScatteringPowerWrap(const ScatteringPowerWrap &S) : ScatteringPower(S) {}

    // Pure Virtual functions

    CrystVector<float> GetScatteringFactor(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
        return this->get_override("GetScatteringFactor")();
    }

    float GetForwardScatteringFactor(const RadiationType) const
    {
        return this->get_override("GetForwardScatteringFactor")();
    }

    CrystVector<float> GetTemperatureFactor(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
        return this->get_override("GetTemperatureFactor")();
    }

    CrystMatrix<float> GetResonantScattFactReal(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
        return this->get_override("GetResonantScattFactReal")();
    }

    CrystMatrix<float> GetResonantScattFactImag(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
        return this->get_override("GetResonantScattFactImag")();
    }

    float GetRadius() const
    {
        return this->get_override("GetRadius")();
    }

    // Just plain virtual functions
    //
    bool default_IsScatteringFactorAnisotropic() const
    { return ScatteringPower::IsScatteringFactorAnisotropic(); }

    bool IsScatteringFactorAnisotropic() const
    {
        if (override IsScatteringFactorAnisotropic = this->get_override("IsScatteringFactorAnisotropic")) 
            return IsScatteringFactorAnisotropic();
        return default_IsScatteringFactorAnisotropic();
    }

    bool default_IsTemperatureFactorAnisotropic() const
    { return ScatteringPower::IsTemperatureFactorAnisotropic(); }

    bool IsTemperatureFactorAnisotropic() const
    {
        if (override IsTemperatureFactorAnisotropic = this->get_override("IsTemperatureFactorAnisotropic")) 
            return IsTemperatureFactorAnisotropic();
        return default_IsTemperatureFactorAnisotropic();
    }

    bool default_IsResonantScatteringAnisotropic() const
    { return ScatteringPower::IsResonantScatteringAnisotropic(); }

    bool IsResonantScatteringAnisotropic() const
    {
        if (override IsResonantScatteringAnisotropic = this->get_override("IsResonantScatteringAnisotropic")) 
            return IsResonantScatteringAnisotropic();
        return default_IsResonantScatteringAnisotropic();
    }

    const std::string& default_GetSymbol() const
    { return ScatteringPower::GetSymbol(); }

    const std::string& GetSymbol() const
    {
        if (override GetSymbol = this->get_override("GetSymbol")) 
            return GetSymbol();
        return default_GetSymbol();
    }

    void default_SetBiso(const float newB)
    { ScatteringPower::SetBiso(newB); }

    void SetBiso(const float newB)
    {
        if (override SetBiso = this->get_override("SetBiso")) 
            SetBiso(newB);
        default_SetBiso(newB);
    }

    float default_GetFormalCharge() const
    { return ScatteringPower::GetFormalCharge(); }

    float GetFormalCharge() const
    {
        if (override GetFormalCharge = this->get_override("GetFormalCharge")) 
            return GetFormalCharge();
        return default_GetFormalCharge();
    }

    void default_SetFormalCharge(const float charge)
    { ScatteringPower::SetFormalCharge(charge); }

    void SetFormalCharge(const float charge)
    {
        if (override SetFormalCharge = this->get_override("SetFormalCharge")) 
            SetFormalCharge(charge);
        default_SetFormalCharge(charge);
    }


    protected:

    void InitRefParList()
    {
        this->get_override("InitRefParList")();
    }


}; // ScatteringPowerWrap

} // anonymous namespace


BOOST_PYTHON_MODULE(_scatteringpower)
{

    class_<ScatteringPowerWrap, boost::noncopyable, bases<RefinableObj> >
        ("ScatteringPower", init<>())
        .def(init<const ScatteringPowerWrap&>())
        .def("GetScatteringFactor", 
            pure_virtual(&ScatteringPower::GetScatteringFactor),
            (boost::python::arg("data"),
            boost::python::arg("spgSymPosIndex")=-1))
        .def("GetForwardScatteringFactor", 
            pure_virtual(&ScatteringPower::GetForwardScatteringFactor))
        .def("GetTemperatureFactor", 
            pure_virtual(&ScatteringPower::GetTemperatureFactor),
            (boost::python::arg("data"),
            boost::python::arg("spgSymPosIndex")=-1))
        .def("GetResonantScattFactReal", 
            pure_virtual(&ScatteringPower::GetResonantScattFactReal),
            (boost::python::arg("data"),
            boost::python::arg("spgSymPosIndex")=-1))
        .def("GetResonantScattFactImag", 
            pure_virtual(&ScatteringPower::GetResonantScattFactImag),
            (boost::python::arg("data"),
            boost::python::arg("spgSymPosIndex")=-1))
        .def("IsScatteringFactorAnisotropic", 
            &ScatteringPower::IsScatteringFactorAnisotropic,
            &ScatteringPowerWrap::default_IsScatteringFactorAnisotropic)
        .def("IsTemperatureFactorAnisotropic", 
            &ScatteringPower::IsTemperatureFactorAnisotropic,
            &ScatteringPowerWrap::default_IsTemperatureFactorAnisotropic)
        .def("IsResonantScatteringAnisotropic", 
            &ScatteringPower::IsResonantScatteringAnisotropic,
            &ScatteringPowerWrap::default_IsResonantScatteringAnisotropic)
        .def("GetSymbol", 
            &ScatteringPower::GetSymbol,
            &ScatteringPowerWrap::default_GetSymbol,
            return_value_policy<copy_const_reference>())
        .def("GetBiso", 
                (float (ScatteringPower::*)()const) &ScatteringPower::GetBiso)
        .def("SetBiso", &ScatteringPower::SetBiso,
            &ScatteringPowerWrap::default_SetBiso)
        .def("IsIsotropic", &ScatteringPower::IsIsotropic)
        .def("GetDynPopCorrIndex", &ScatteringPower::GetDynPopCorrIndex)
        .def("GetNbScatteringPower", &ScatteringPower::GetNbScatteringPower)
        .def("GetLastChangeClock", &ScatteringPower::GetLastChangeClock,
                return_value_policy<copy_const_reference>())
        .def("GetRadius", 
            pure_virtual(&ScatteringPower::GetRadius))
        .def("GetMaximumLikelihoodPositionError", 
            pure_virtual(&ScatteringPower::GetMaximumLikelihoodPositionError))
        .def("SetMaximumLikelihoodPositionError", 
            pure_virtual(&ScatteringPower::SetMaximumLikelihoodPositionError))
        .def("GetMaximumLikelihoodNbGhostAtom", 
            pure_virtual(&ScatteringPower::GetMaximumLikelihoodNbGhostAtom))
        .def("SetMaximumLikelihoodNbGhostAtom", 
            pure_virtual(&ScatteringPower::SetMaximumLikelihoodNbGhostAtom))
        .def("GetMaximumLikelihoodParClock", 
                &ScatteringPower::GetMaximumLikelihoodParClock,
                return_value_policy<copy_const_reference>())
        .def("GetFormalCharge", 
            &ScatteringPower::GetFormalCharge,
            &ScatteringPowerWrap::default_GetFormalCharge)
        .def("SetFormalCharge", 
            &ScatteringPower::SetFormalCharge,
            &ScatteringPowerWrap::default_SetFormalCharge)
        ;
}
