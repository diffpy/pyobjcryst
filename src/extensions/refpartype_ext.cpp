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
* boost::python bindings to ObjCryst::RefParType.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>

using namespace ObjCryst;
using namespace boost::python;

namespace {

bool __eq__(const RefParType* rpt1, const RefParType* rpt2)
{
    return rpt1 == rpt2;
}

} // anonymous namespace


void wrap_refpartype()
{

    class_<RefParType>("RefParType", init<const string&>())
        .def(init<const RefParType*, const string&>()
            [with_custodian_and_ward<1,2>()])
        /* Functions */
        .def("IsDescendantFromOrSameAs", &RefParType::IsDescendantFromOrSameAs)
        .def("__eq__", &__eq__)
        .def("GetName", &RefParType::GetName,
            return_value_policy<copy_const_reference>())
        ;
}
