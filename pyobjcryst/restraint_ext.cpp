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
* boost::python bindings to ObjCryst::RefinablePar and
* ObjCryst::RefParDerivStepModel.
* 
* Changes from ObjCryst++
* * The constructor has been changed to accept a float,
*   rather than a pointer to a float. 
* * The default and copy constructors are not wrapped, nor is Init.
* * Get type returns a non-const reference to the RefParType by using const_cast
*   in the bindings. This should be a no-no, but RefParType has no mutating
*   methods, so this should no lead to trouble.
* * XML input/output are on hold until a general stream adapter is developed.
*
* $Id$
*
*****************************************************************************/

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include "RefinableObj/RefinableObj.h"

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
        {
            return GetType();
        }
        return default_GetType();
    }

    // Overloaded so it can be wrapped. boost::python doesnt like const xxx*
    RefParType& GetTypeRef()
    {
        return *const_cast<RefParType*>(GetType());
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

    float default_GetLogLikelihood() const
    {
        return this->Restraint::GetLogLikelihood();
    }

    float GetLogLikelihood() const
    {
        if( override GetLogLikelihood = this->get_override("GetLogLikelihood"))
        {
            return GetLogLikelihood();
        }
        return default_GetLogLikelihood();
    }

};

} // anonymous namespace


BOOST_PYTHON_MODULE(_restraint)
{

    class_<RestraintWrap, boost::noncopyable>("Restraint", init<>())
        .def(init<const RefParType*>((bp::arg("type"))))
        .def("GetType", &RestraintWrap::GetTypeRef,
            return_internal_reference<>())
        .def("SetType", &Restraint::SetType, &RestraintWrap::default_SetType)
        .def("GetLogLikelihood", &Restraint::GetLogLikelihood, 
            &RestraintWrap::default_GetLogLikelihood)
        ;
}
