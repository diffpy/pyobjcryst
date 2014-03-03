/*****************************************************************************
*
* pyobjcryst        by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 The Trustees of Columbia University
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
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include <ObjCryst/ObjCryst/ScatteringPower.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>

using namespace boost::python;
using namespace ObjCryst;

namespace {

class ScatteringPowerWrap : public ScatteringPower,
                 public wrapper<ScatteringPower>
{

    public:

    ScatteringPowerWrap() : ScatteringPower() {}
    ScatteringPowerWrap(const ScatteringPower &S) : ScatteringPower(S) {}
    ScatteringPowerWrap(const ScatteringPowerWrap &S) : ScatteringPower(S) {}

    // Pure Virtual functions

    CrystVector<double> GetScatteringFactor(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
#ifdef _MSC_VER
        return call<CrystVector<double> >(
                this->get_override("GetScatteringFactor").ptr(),
                data, spgSymPosIndex
                );
#else
        return this->get_override("GetScatteringFactor")(data, spgSymPosIndex);
#endif
    }

    double GetForwardScatteringFactor(const RadiationType type) const
    {
#ifdef _MSC_VER
        return call<double>(
                this->get_override("GetForwardScatteringFactor").ptr(),
                type
                );
#else
        return this->get_override("GetForwardScatteringFactor")(type);
#endif
    }

    CrystVector<double> GetTemperatureFactor(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
#ifdef _MSC_VER
        return call<CrystVector<double> >(
                this->get_override("GetTemperatureFactor").ptr(),
                data, spgSymPosIndex
                );
#else
        return this->get_override("GetTemperatureFactor")(data, spgSymPosIndex);
#endif
    }

    CrystMatrix<double> GetResonantScattFactReal(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
#ifdef _MSC_VER
        return call<CrystMatrix<double> >(
                this->get_override("GetResonantScattFactReal").ptr(),
                data, spgSymPosIndex
             );
#else
        return this->get_override("GetResonantScattFactReal")(data,
                spgSymPosIndex);
#endif
    }

    CrystMatrix<double> GetResonantScattFactImag(
            const ScatteringData &data, const int spgSymPosIndex) const
    {
#ifdef _MSC_VER
        return call<CrystMatrix<double> >(
                this->get_override("GetResonantScattFactImag").ptr(),
                data, spgSymPosIndex
                );
#else
        return this->get_override("GetResonantScattFactImag")(data,
                spgSymPosIndex);
#endif
    }

    double GetRadius() const
    {
#ifdef _MSC_VER
        return call<double>(this->get_override("GetRadius").ptr());
#else
        return this->get_override("GetRadius")();
#endif
    }

    // Just plain virtual functions
    //
    bool default_IsScatteringFactorAnisotropic() const
    { return ScatteringPower::IsScatteringFactorAnisotropic(); }

    bool IsScatteringFactorAnisotropic() const
    {
        if (override IsScatteringFactorAnisotropic = this->get_override("IsScatteringFactorAnisotropic"))
#ifdef _MSC_VER
            return call<bool>(
                    IsScatteringFactorAnisotropic.ptr()
                    );
#else
            return IsScatteringFactorAnisotropic();
#endif
        return default_IsScatteringFactorAnisotropic();
    }

    bool default_IsTemperatureFactorAnisotropic() const
    { return ScatteringPower::IsTemperatureFactorAnisotropic(); }

    bool IsTemperatureFactorAnisotropic() const
    {
        if (override IsTemperatureFactorAnisotropic = this->get_override("IsTemperatureFactorAnisotropic"))
#ifdef _MSC_VER
            return call<bool>(
                    IsTemperatureFactorAnisotropic.ptr()
                    );
#else
            return IsTemperatureFactorAnisotropic();
#endif
        return default_IsTemperatureFactorAnisotropic();
    }

    bool default_IsResonantScatteringAnisotropic() const
    { return ScatteringPower::IsResonantScatteringAnisotropic(); }

    bool IsResonantScatteringAnisotropic() const
    {
        if (override IsResonantScatteringAnisotropic = this->get_override("IsResonantScatteringAnisotropic"))
#ifdef _MSC_VER
            return call<bool>(
                    IsResonantScatteringAnisotropic.ptr()
                    );
#else
            return IsResonantScatteringAnisotropic();
#endif
        return default_IsResonantScatteringAnisotropic();
    }

    const std::string& default_GetSymbol() const
    { return ScatteringPower::GetSymbol(); }

    const std::string& GetSymbol() const
    {
        if (override GetSymbol = this->get_override("GetSymbol"))
#ifdef _MSC_VER
            return call<const std::string&>(
                    GetSymbol.ptr()
                    );
#else
            return GetSymbol();
#endif
        return default_GetSymbol();
    }

    void default_SetBiso(const double newB)
    { ScatteringPower::SetBiso(newB); }

    void SetBiso(const double newB)
    {
        if (override SetBiso = this->get_override("SetBiso"))
            SetBiso(newB);
        default_SetBiso(newB);
    }

    void default_SetBij(const size_t& i, const size_t& j, const double newB)
    { ScatteringPower::SetBij(i, j, newB); }

    void SetBij(const size_t& i, const size_t& j, const double newB)
    {
        if (override SetBij = this->get_override("SetBij"))
            SetBij(i, j, newB);
        default_SetBij(i, j, newB);
    }

    double default_GetFormalCharge() const
    { return ScatteringPower::GetFormalCharge(); }

    double GetFormalCharge() const
    {
        if (override GetFormalCharge = this->get_override("GetFormalCharge"))
#ifdef _MSC_VER
            return call<double>(
                    GetFormalCharge.ptr()
                    );
#else
            return GetFormalCharge();
#endif
        return default_GetFormalCharge();
    }

    void default_SetFormalCharge(const double charge)
    { ScatteringPower::SetFormalCharge(charge); }

    void SetFormalCharge(const double charge)
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


// Accessors for Bij parameters

template <size_t i, size_t j>
double _GetBij(ScatteringPower& sp)
{
    return sp.GetBij(i, j);
}

template <size_t i, size_t j>
void _SetBij(ScatteringPower& sp, const double newB)
{
    return sp.SetBij(i, j, newB);
}


} // anonymous namespace


void wrap_scatteringpower()
{

    // By making this non-copyable ScatteringPower can be passed from c++ when
    // copy_const_reference is uses, but they are turned into RefinableObj
    // instances.
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
                (double (ScatteringPower::*)()const) &ScatteringPower::GetBiso)
        .def("SetBiso", &ScatteringPower::SetBiso,
            &ScatteringPowerWrap::default_SetBiso)
        .def("GetBij",
                (double (ScatteringPower::*)(const size_t&, const size_t&) const)
                &ScatteringPower::GetBij)
        .def("SetBij", (void (ScatteringPower::*)
            (const size_t&, const size_t&, const double))
            &ScatteringPower::SetBij,
            &ScatteringPowerWrap::default_SetBij)
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
        .add_property("Biso", (double (ScatteringPower::*)()const)
                &ScatteringPower::GetBiso, &ScatteringPower::SetBiso)
        .add_property("B11", &_GetBij<1,1>, &_SetBij<1,1>)
        .add_property("B22", &_GetBij<2,2>, &_SetBij<2,2>)
        .add_property("B33", &_GetBij<3,3>, &_SetBij<3,3>)
        .add_property("B12", &_GetBij<1,2>, &_SetBij<1,2>)
        .add_property("B13", &_GetBij<1,3>, &_SetBij<1,3>)
        .add_property("B23", &_GetBij<2,3>, &_SetBij<2,3>)
        ;
}
