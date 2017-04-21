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
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::RefinablePar and
* ObjCryst::RefParDerivStepModel.
*
* Changes from ObjCryst::RefinablePar
* - The constructor has been changed to accept a double,
*   rather than a pointer to a double.
* - The copy and default constructors and Init are not wrapped in order to avoid
*   memory corruption. Since boost cannot implicitly handle double* object, a
*   wrapper class had to be created. However, this wrapper class cannot be used
*   to convert RefinablePar objected created in c++.  Thus,
*   ObjCryst::RefinablePar objects created in c++ are passed into python as
*   instances of _RefinablePar, which is a python wrapper around
*   ObjCryst::RefinablePar. The RefinablePar python class is a wrapper around
*   the C++ class PyRefinablePar, which manages its own double*.  These python
*   classes are interchangable once instantiated, so users should not notice.
* - XML input/output are not exposed.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

/* A little wrapper around the initializers since python doubles and double* don't
 * get along.
 */

class PyRefinablePar : public RefinablePar
{

    public:

    PyRefinablePar() : RefinablePar(),  pval(0) {};

    PyRefinablePar(const string& name, double value, const double min, const
        double max, const RefParType* type, RefParDerivStepModel
        derivMode=REFPAR_DERIV_STEP_RELATIVE, const bool hasLimits=true, const
        bool isFixed=false, const bool isUsed=true, const bool isPeriodic=false,
        const double humanScale=1., double period=1.) : RefinablePar()
    {
        pval = new double(value);
        RefinablePar::Init(name, pval, min, max, type, derivMode, hasLimits,
            isFixed, isUsed, isPeriodic, humanScale, period);
    }

    ~PyRefinablePar()
    {
        if( pval != NULL )
        {
            delete pval;
        }
    }

    private:

    double* pval;


};

} // anonymous namespace


void wrap_refinablepar()
{

    enum_<RefParDerivStepModel>("RefParDerivStepModel")
        .value("REFPAR_DERIV_STEP_ABSOLUTE", REFPAR_DERIV_STEP_ABSOLUTE)
        .value("REFPAR_DERIV_STEP_RELATIVE", REFPAR_DERIV_STEP_RELATIVE)
        ;

    // Class for holding RefinablePar instances created in c++. This should not
    // be exported to the pyobjcryst namespace
    class_<RefinablePar, bases<Restraint> > ("_RefinablePar", no_init)
        .def("GetValue", &RefinablePar::GetValue)
        .def("SetValue", &RefinablePar::SetValue)
        .def("GetHumanValue", &RefinablePar::GetHumanValue,
            return_value_policy<copy_const_reference>())
        .def("SetHumanValue", &RefinablePar::SetHumanValue)
        .def("Mutate", &RefinablePar::Mutate)
        .def("MutateTo", &RefinablePar::MutateTo)
        .def("GetSigma", &RefinablePar::GetSigma)
        .def("GetHumanSigma", &RefinablePar::GetHumanSigma)
        .def("SetSigma", &RefinablePar::SetSigma)
        .def("GetName", &RefinablePar::GetName)
        .def("SetName", &RefinablePar::SetName)
        .def("Print", &RefinablePar::Print)
        .def("IsFixed", &RefinablePar::IsFixed)
        .def("SetIsFixed", &RefinablePar::SetIsFixed)
        .def("IsLimited", &RefinablePar::IsLimited)
        .def("SetIsLimited", &RefinablePar::SetIsLimited)
        .def("IsUsed", &RefinablePar::IsUsed)
        .def("SetIsUsed", &RefinablePar::SetIsUsed)
        .def("IsPeriodic", &RefinablePar::IsPeriodic)
        .def("SetIsPeriodic", &RefinablePar::SetIsPeriodic)
        .def("GetHumanScale", &RefinablePar::GetHumanScale)
        .def("SetHumanScale", &RefinablePar::SetHumanScale)
        .def("GetMin", &RefinablePar::GetMin)
        .def("SetMin", &RefinablePar::SetMin)
        .def("GetHumanMin", &RefinablePar::GetHumanMin)
        .def("SetHumanMin", &RefinablePar::SetHumanMin)
        .def("GetMax", &RefinablePar::GetMax)
        .def("SetMax", &RefinablePar::SetMax)
        .def("GetHumanMax", &RefinablePar::GetHumanMax)
        .def("SetHumanMax", &RefinablePar::SetHumanMax)
        .def("GetPeriod", &RefinablePar::GetPeriod)
        .def("SetPeriod", &RefinablePar::SetPeriod)
        .def("GetDerivStep", &RefinablePar::GetDerivStep)
        .def("SetDerivStep", &RefinablePar::SetDerivStep)
        .def("GetGlobalOptimStep", &RefinablePar::GetGlobalOptimStep)
        .def("SetGlobalOptimStep", &RefinablePar::SetGlobalOptimStep)
        .def("AssignClock", &RefinablePar::AssignClock)
        .def("SetLimitsAbsolute", &RefinablePar::SetLimitsAbsolute)
        .def("SetLimitsRelative", &RefinablePar::SetLimitsRelative)
        .def("SetLimitsProportional", &RefinablePar::SetLimitsProportional)
        // Python-only attributes
        .def("__str__", &__str__<RefinablePar>)
        .add_property("value", &RefinablePar::GetValue, &RefinablePar::SetValue)
        ;

    // Class for creating new RefinablePar instances
    class_<PyRefinablePar, bases<RefinablePar> >
        ("RefinablePar", //...
        init<const string&, double, const double, const double, const RefParType*,
            RefParDerivStepModel, const bool, const bool, const bool, const
            bool, const double, double >((
                bp::arg("name"), bp::arg("value"), bp::arg("min"),
                bp::arg("max"), bp::arg("type"),
                bp::arg("derivMode")=REFPAR_DERIV_STEP_RELATIVE,
                bp::arg("hasLimits")=true, bp::arg("isFixed")=false,
                bp::arg("isUsed")=true, bp::arg("isPeriodic")=false,
                bp::arg("humanScale")=1., bp::arg("period")=1.))
            [with_custodian_and_ward<1,6>()])
        ;


}
