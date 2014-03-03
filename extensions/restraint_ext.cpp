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

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
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
        if( override GetType = this->get_override("GetType"))
#ifdef _MSC_VER
            return call<const RefParType*>(
                    GetType.ptr()
                    );
#else
            return GetType();
#endif
        return default_GetType();
    }

    void default_SetType(const RefParType* type)
    {
        this->Restraint::SetType(type);
        return;
    }

    void SetType(const RefParType* type)
    {
        if( override SetType = this->get_override("SetType"))
        {
            SetType(type);
            return;
        }
        default_SetType(type);
        return;
    }

    double default_GetLogLikelihood() const
    {
        return this->Restraint::GetLogLikelihood();
    }

    double GetLogLikelihood() const
    {
        if( override GetLogLikelihood = this->get_override("GetLogLikelihood"))
#ifdef _MSC_VER
            return call<double>(
                    GetLogLikelihood.ptr()
                    );
#else
            return GetLogLikelihood();
#endif
        return default_GetLogLikelihood();
    }

};

} // anonymous namespace


void wrap_restraint()
{

    class_<RestraintWrap, boost::noncopyable>("Restraint", init<>())
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
