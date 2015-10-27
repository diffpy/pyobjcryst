/*****************************************************************************
*
* pyobjcryst
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
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

#include <iostream>

#include <ObjCryst/ObjCryst/ReflectionProfile.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

class ReflectionProfileWrap :
    public ReflectionProfile, public wrapper<ReflectionProfile>
{
    //:TODO: :KLUDGE: Dummy override of pure virtual functions
    public:
        ReflectionProfileWrap() : ReflectionProfile() {}
        virtual ReflectionProfile* CreateCopy() const {}
        virtual CrystVector_REAL GetProfile(
                const CrystVector_REAL& x, const REAL xcenter,
                const REAL h, const REAL k, const REAL l) const
        {}
        virtual REAL GetFullProfileWidth(
                const REAL relativeIntensity, const REAL xcenter,
                const REAL h, const REAL k, const REAL l)
        {}
        virtual void XMLOutput(ostream& os, int indent) const
        {}
        virtual void XMLInput(istream& is, const XMLCrystTag& tag)
        {}
};

}   // namespace


void wrap_reflectionprofile()
{
    class_<ReflectionProfileWrap, bases<RefinableObj>, boost::noncopyable>(
            "ReflectionProfile")
        ;
}
