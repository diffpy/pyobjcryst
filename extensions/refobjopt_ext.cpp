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
* boost::python bindings to ObjCryst::RefObjOpt.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>

using namespace boost::python;
using namespace ObjCryst;

namespace {

class RefObjOptWrap : public RefObjOpt,
                 public wrapper<RefObjOpt>
{

    public:

    void default_SetChoice(const int choice)
    {
        RefObjOpt::SetChoice(choice);
    }

    void SetChoice(const int choice)
    {
        override f = this->get_override("SetChoice");
        if (f)  f(choice);
        else  default_SetChoice(choice);
    }

}; // AtomWrap

} // anonymous namespace


void wrap_refobjopt()
{

    class_<RefObjOptWrap, boost::noncopyable>("RefObjOpt")
        .def("Init", &RefObjOpt::Init)
        .def("GetNbChoice", &RefObjOpt::GetNbChoice)
        .def("GetChoice", &RefObjOpt::GetChoice)
        .def("SetChoice", (void (RefObjOpt::*)(const int)) &RefObjOpt::SetChoice,
                    &RefObjOptWrap::default_SetChoice)
        .def("SetChoice", (void (RefObjOpt::*)(const std::string&)) &RefObjOpt::SetChoice)
        .def("GetName", &RefObjOpt::GetName,
            return_value_policy<copy_const_reference>())
        .def("GetClassName", &RefObjOpt::GetClassName,
            return_value_policy<copy_const_reference>())
        .def("GetChoiceName", &RefObjOpt::GetChoiceName,
            return_value_policy<copy_const_reference>())
        .def("GetClock", &RefObjOpt::GetClock,
                return_value_policy<copy_const_reference>())
        .def("XMLOutput", &RefObjOpt::XMLOutput)
        .def("XMLInput", &RefObjOpt::XMLInput)
        ;
}
