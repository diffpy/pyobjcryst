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
* boost::python bindings to ObjCryst::PowderPatternComponent.
*
* Changes from ObjCryst::PowderPatternComponent
*
* Other Changes
*
*****************************************************************************/

#include <boost/python/class.hpp>
#undef B0

#include <ObjCryst/ObjCryst/PowderPattern.h>

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;


void wrap_powderpatterncomponent()
{
    class_<PowderPatternComponent, bases<RefinableObj>, boost::noncopyable>
        ("PowderPatternComponent", no_init)
        .def("GetParentPowderPattern",
                (PowderPattern& (PowderPatternComponent::*)())
                &PowderPatternComponent::GetParentPowderPattern,
                return_internal_reference<>())
        ;
}
