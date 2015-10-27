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
* boost::python bindings to ObjCryst/RefinableObj/IO.h.
*
* Changes from ObjCryst::XMLCrystTag
* - The istream constructor of XMLCrystTag is not wrapped.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/RefinableObj/IO.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

void wrap_io()
{

    class_<XMLCrystTag>
        ("XMLCrystTag", init<const std::string&, const bool, const bool>
         ((bp::arg("tagName"),
           bp::arg("isEndTag")=false,
           bp::arg("isEmptyTag")=false)))
        .def("GetName", &XMLCrystTag::GetName,
            return_value_policy<copy_const_reference>())
        .def("GetClassName", &XMLCrystTag::GetClassName,
            return_value_policy<copy_const_reference>())
        .def("GetNbAttribute", &XMLCrystTag::GetNbAttribute)
        .def("AddAttribute", &XMLCrystTag::AddAttribute,
            (bp::arg("attName"), bp::arg("attValue")))
        .def("GetAttribute", &XMLCrystTag::GetAttribute,
            (bp::arg("attNum"), bp::arg("attName"), bp::arg("attValue")))
        .def("GetAttributeName", &XMLCrystTag::GetAttributeName,
            return_value_policy<copy_const_reference>())
        .def("GetAttributeValue", &XMLCrystTag::GetAttributeValue,
            return_value_policy<copy_const_reference>())
        .def("SetIsEndTag", &XMLCrystTag::SetIsEndTag)
        .def("IsEndTag", &XMLCrystTag::IsEndTag)
        .def("SetIsEmptyTag", &XMLCrystTag::SetIsEmptyTag)
        .def("IsEmptyTag", &XMLCrystTag::IsEmptyTag)
        .def("Print", &XMLCrystTag::Print)
        .def("__str__", &__str__<XMLCrystTag>)
        // python-only
        ;
}
