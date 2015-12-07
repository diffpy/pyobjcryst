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
* boost::python bindings to ObjCryst::RefinableObjClock.
*
* Changes from ObjCryst::RefinableObjClock
* - operator= is wrapped as the SetEqual method
*   a.SetEqual(b) -> a = b
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/operators.hpp>

#include <ObjCryst/RefinableObj/RefinableObj.h>

#include "helpers.hpp"

using namespace boost::python;
using ObjCryst::RefinableObjClock;

namespace {

const char* classdoc =
"We need to record exactly when refinable objects\n\
have been modified for the last time (to avoid re-computation),\n\
and to do that we need a precise time. Since the clock() function is not\n\
precise enough (and is architecture-dependant), we use a custom time,\n\
which records the number of events in the program which uses the library.\n\
This is purely internal, so don't worry about it...\n\
\n\
The clock values have nothing to do with 'time' as any normal person undertands it.";

const char* addchilddoc =
"Add a 'child' clock. Whenever a child clock is clicked, it will also click its parent.\n\
This function takes care of adding itself to the list of parent in the children clock.";

const char* addparentdoc =
"Add a 'parent' clock. Whenever a clock is clicked, all parent clocks\n\
also are.";

const char* clickdoc =
"Record an event for this clock (generally, the 'time'\n\
an object has been modified, or some computation has been made)";

const char * printdoc =
"Print clock value. Only for debugging purposes.";

const char * printstaticdoc =
"Print current general clock value. Only for debugging purposes.";

const char *  removechilddoc =
"remove a child clock. This also tells the child clock to remove the parent.";

const char *  removeparentdoc =
"remove a parent clock";

const char * resetdoc =
"Reset a Clock to 0, to force an update";

// set clock1 equal to clock2 (operator=)
void SetEqual(RefinableObjClock& c1, const RefinableObjClock& c2)
{
    c1 = c2;
}


} // anonymous namespace

void wrap_refinableobjclock()
{

    class_<RefinableObjClock>
        ("RefinableObjClock", classdoc)
        .def("AddChild", &RefinableObjClock::AddChild, addchilddoc,
                with_custodian_and_ward<1,2>())
        .def("AddParent", &RefinableObjClock::AddParent, addparentdoc,
                with_custodian_and_ward<1,2>())
        .def("Click", &RefinableObjClock::Click, clickdoc)
        .def("Print", &RefinableObjClock::Print, printdoc)
        .def("PrintStatic", &RefinableObjClock::PrintStatic, printstaticdoc)
        .def("RemoveChild", &RefinableObjClock::RemoveChild, removechilddoc)
        .def("RemoveParent", &RefinableObjClock::RemoveParent, removeparentdoc)
        .def("Reset", &RefinableObjClock::Reset, resetdoc)
        .def("SetEqual", &SetEqual)
        .def(self < self)
        .def(self <= self)
        .def(self > self)
        .def(self >= self)
        .def("__str__", &__str__<RefinableObjClock>)
        ;
}
