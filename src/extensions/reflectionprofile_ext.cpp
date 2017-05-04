/*****************************************************************************
*
* pyobjcryst
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::ReflectionProfile.
*
* Changes from ObjCryst::ReflectionProfile
*
* Other Changes
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/pure_virtual.hpp>
#undef B0

#include <iostream>

#include <ObjCryst/ObjCryst/ReflectionProfile.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

class ReflectionProfileWrap :
    public ReflectionProfile, public wrapper<ReflectionProfile>
{
    public:

        // Pure virtual functions

        ReflectionProfile* CreateCopy() const
        {
            return this->get_override("CreateCopy")();
        }

        CrystVector_REAL GetProfile(
                const CrystVector_REAL& x, const REAL xcenter,
                const REAL h, const REAL k, const REAL l) const
        {
            bp::override f = this->get_override("GetProfile");
            return f(x, xcenter, h, k, l);
        }

        REAL GetFullProfileWidth(
                const REAL relativeIntensity, const REAL xcenter,
                const REAL h, const REAL k, const REAL l)
        {
            bp::override f = this->get_override("GetFullProfileWidth");
            return f(relativeIntensity, xcenter, h, k, l);
        }

        void XMLOutput(ostream& os, int indent) const
        {
            bp::override f = this->get_override("XMLOutput");
            f(os, indent);
        }

        void XMLInput(istream& is, const XMLCrystTag& tag)
        {
            bp::override f = this->get_override("GetProfile");
            f(is, tag);
        }
};

}   // namespace


void wrap_reflectionprofile()
{
    class_<ReflectionProfileWrap, bases<RefinableObj>, boost::noncopyable>(
            "ReflectionProfile")
        // TODO add pure_virtual bindings to the remaining public methods
        .def("CreateCopy",
                pure_virtual(&ReflectionProfile::CreateCopy),
                return_value_policy<manage_new_object>())
        ;
}
