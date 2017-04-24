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
* boost::python bindings to ObjCryst::Scatterer. This is a virtual class that
* can be derived from in python. These bindings are used by ObjCryst objects
* that inherit from Scatterer (see for example atom_ext.cpp).
*
* Changes from ObjCryst::Scatterer
* - C++ methods that can return const or non-const objects return non-const
*   objects in python.
* - Operator string() is not exposed.
* - Internal use only methods have not been exposed.
* - InitRefParList is not exposed, as it is not used inside of Scatterer.
* - GetClockScattCompList is exposed using a workaround, because it is not
*   implemented in the library.
* - Methods related to visualization are not exposed.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/python/copy_const_reference.hpp>

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/Scatterer.h>

#include "helpers.hpp"

using namespace ObjCryst;
using namespace boost::python;

namespace {

class ScattererWrap : public Scatterer,
                      public wrapper<Scatterer>
{

    public:

    ScattererWrap() : Scatterer() {}

    ScattererWrap(const ScattererWrap& S) : Scatterer(S) {}

    void default_SetX(const double x)
    { this->Scatterer::SetX(x);}

    void SetX(const double x)
    {
        override f = this->get_override("SetX");
        if (f)  f(x);
        else  default_SetX(x);
    }

    void default_SetY(const double y)
    { this->Scatterer::SetY(y);}

    void SetY(const double y)
    {
        override f = this->get_override("SetY");
        if (f)  f(y);
        else  default_SetY(y);
    }

    void default_SetZ(const double z)
    { this->Scatterer::SetZ(z);}

    void SetZ(const double z)
    {
        override f = this->get_override("SetZ");
        if (f)  f(z);
        else  default_SetZ(z);
    }

    void default_SetOccupancy(const double occ)
    { this->Scatterer::SetOccupancy(occ);}

    void SetOccupancy(const double occ)
    {
        override f = this->get_override("SetOccupancy");
        if (f)  f(occ);
        else  default_SetOccupancy(occ);
    }

    // Pure virtual

    Scatterer* CreateCopy() const
    {
        return this->get_override("CreateCopy")();
    }

    int GetNbComponent() const
    {
        return this->get_override("GetNbComponent")();
    }

    const ScatteringComponentList& GetScatteringComponentList() const
    {
        return this->get_override("GetScatteringComponentList")();
    }

    std::string GetComponentName(const int i) const
    {
        return this->get_override("GetComponentName")(i);
    }

    void Print() const
    {
        this->get_override("Print")();
    }

    std::ostream& POVRayDescription(std::ostream& os,
            const CrystalPOVRayOptions& options) const
    {
        return this->get_override("POVRayDescription")(os, options);
    }

    // interface prior to Fox version 2016.2
    void GLInitDisplayList(const bool noSymmetrics,
            const REAL xMin, const REAL xMax,
            const REAL yMin, const REAL yMax,
            const REAL zMin, const REAL zMax,
            const bool displayEnantiomer,
            const bool displayNames,
            const bool hideHydrogens) const
    {
        // no operation
    }

    void GLInitDisplayList(const bool noSymmetrics,
            const REAL xMin, const REAL xMax,
            const REAL yMin, const REAL yMax,
            const REAL zMin, const REAL zMax,
            const bool displayEnantiomer,
            const bool displayNames,
            const bool hideHydrogens,
            const REAL fadeDistance=0) const
    {
        // no operation
    }

    const RefinableObjClock& _GetClockScattCompList() const
    {
        return mClockScattCompList;
    }

    protected:

    // Needed for compilation
    void InitRefParList() {};

}; // ScattererWrap

// We want to turn a ScatteringComponentList into an actual list
bp::list _GetScatteringComponentList(Scatterer& s)
{
    const ScatteringComponentList& scl = s.GetScatteringComponentList();
    bp::list l;
    for(int i = 0; i < scl.GetNbComponent(); ++i)
    {
        l.append(scl(i));
    }

    return l;
}


} // anonymous namespace


void wrap_scatterer()
{

    class_<ScattererWrap, boost::noncopyable, bases<RefinableObj> >
        ("Scatterer")
        /* Constructors */
        .def(init<const ScattererWrap&>())
        /* Methods */
        .def("GetX", &Scatterer::GetX)
        .def("GetY", &Scatterer::GetY)
        .def("GetZ", &Scatterer::GetZ)
        .def("GetOccupancy", &Scatterer::GetOccupancy)
        // virtual methods
        .def("SetX", &Scatterer::SetX, &ScattererWrap::default_SetX)
        .def("SetY", &Scatterer::SetY, &ScattererWrap::default_SetY)
        .def("SetZ", &Scatterer::SetZ, &ScattererWrap::default_SetZ)
        .def("SetOccupancy", &ObjCryst::Scatterer::SetOccupancy,
            &ScattererWrap::default_SetOccupancy)
        .def("GetClockScatterer",
            (RefinableObjClock & (Scatterer::*)())
            &Scatterer::GetClockScatterer,
            return_internal_reference<>())
        .def("SetCrystal", &Scatterer::SetCrystal,
            with_custodian_and_ward<1,2>())
        .def("GetCrystal", (Crystal &(Scatterer::*)())
            &Scatterer::GetCrystal,
            return_internal_reference<>())
        // pure virtual methods
        .def("GetNbComponent", pure_virtual(&Scatterer::GetNbComponent))
        .def("GetComponentName", pure_virtual(&Scatterer::GetComponentName))
        //.def("GetScatteringComponentList",
        //    pure_virtual(&Scatterer::GetScatteringComponentList),
        //    return_value_policy<copy_const_reference>())
        .def("GetScatteringComponentList", &_GetScatteringComponentList,
            with_custodian_and_ward_postcall<1,0>())
        .def("Print", pure_virtual(&Scatterer::Print))
        .def("__str__", &__str__<Scatterer>)
        // protected methods
        .def("GetClockScattCompList",
            &ScattererWrap::_GetClockScattCompList,
            return_value_policy<copy_const_reference>())
        // Properties - to be compatible with MolAtom
        .add_property("X", &Scatterer::GetX, &Scatterer::SetX)
        .add_property("Y", &Scatterer::GetY, &Scatterer::SetY)
        .add_property("Z", &Scatterer::GetZ, &Scatterer::SetZ)
        .add_property("Occupancy", &Scatterer::GetOccupancy,
                &Scatterer::SetOccupancy)
        ;

}
