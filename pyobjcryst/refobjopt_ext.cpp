#include <string>
#include <iostream>

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include "RefinableObj/RefinableObj.h"

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
        if (override SetChoice = 
                this->get_override("SetChoice")) 
            SetChoice(choice);
        default_SetChoice(choice);
    }

}; // AtomWrap

} // anonymous namespace


BOOST_PYTHON_MODULE(_refobjopt)
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
