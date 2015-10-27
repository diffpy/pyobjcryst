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
* Changes from ObjCryst::Restraint
* - The default and copy constructors are not wrapped, nor is Init.
* - GetType returns a non-const reference to the RefParType.  This should be a
*   no-no, but RefParType has no mutating methods, so this should no lead to
*   trouble.
* - XML input/output are not exposed.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>

#include <ObjCryst/RefinableObj/RefinableObj.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

class RestraintWrap : public Restraint,
                      public wrapper<Restraint>
{
    public:

    RestraintWrap() : Restraint() {};
    RestraintWrap(const RefParType* type) : Restraint(type) {};

    const RefParType* default_GetType() const
    {
        return this->Restraint::GetType();
    }

    const RefParType* GetType() const
    {
        override f = this->get_override("GetType");
        if(f)  return f();
        return this->default_GetType();
    }

    void default_SetType(const RefParType* type)
    {
        this->Restraint::SetType(type);
        return;
    }

    void SetType(const RefParType* type)
    {
        override f = this->get_override("SetType");
        if(f)
        {
            f(type);
            return;
        }
        return this->default_SetType(type);
    }

    double default_GetLogLikelihood() const
    {
        return this->Restraint::GetLogLikelihood();
    }

    double GetLogLikelihood() const
    {
        override f = this->get_override("GetLogLikelihood");
        if(f)  return f();
        return this->default_GetLogLikelihood();
    }

};

} // anonymous namespace


void wrap_restraint()
{

    class_<RestraintWrap, boost::noncopyable>("Restraint")
        .def(init<const RefParType*>((bp::arg("type")))[
            with_custodian_and_ward<1,2>()])
        .def("GetType", &Restraint::GetType, &RestraintWrap::default_GetType,
           return_internal_reference<>())
        .def("SetType", &Restraint::SetType, &RestraintWrap::default_SetType,
            with_custodian_and_ward<1,2>())
        .def("GetLogLikelihood", &Restraint::GetLogLikelihood,
            &RestraintWrap::default_GetLogLikelihood)
        ;
}
