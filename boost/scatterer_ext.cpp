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
* boost::python bindings to ObjCryst::Scatterer. This is a virtual class that
* can be derived from in python. These bindings are used by ObjCryst objects
* that inherit from Scatterer (see for example atom_ext.cpp).
*
* Changes from ObjCryst++
*
* - C++ methods that can return const or non-const objects return non-const
*   objects in python.
* - Operator string() is not exposed.
* - The output of Print() can be accessed from __str__(). This means that 
*   'print scatterer' is equivalent to scatterer.Print().
* - Internal use only methods have not been exposed.
* - InitRefParList is not exposed, as it is not used inside of Scatterer.
* - GetClockScattCompList is exposed using a workaround, because it is not
*   implemented in the library.
* - Methods related to visualization are not exposed.
*
*
* $Id$
*
*****************************************************************************/

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

#include "RefinableObj/RefinableObj.h"
#include "ObjCryst/General.h"
#include "ObjCryst/Scatterer.h"

#include "helpers.hpp"

using namespace ObjCryst;
using namespace boost::python;

namespace {

const char* scattererdoc = "Generic type of scatterer: can be an atom, or a more complex assembly of atoms.  A Scatterer is able to give its position (in fractionnal coordinates) in the unit cell, and more generally the position of all point scattering centers (ScatteringComponent), along with the ScatteringPower associated with each position.  For simple atoms, there is only one scattering position with the associated scattering power (scattering factor, anomalous, thermic). For complex scatterers (molecules: ZScatterer) there are as many positions as atoms.  A scatterer also has a few functions to display itself in 3D. This is an abstract base class.";

class ScattererWrap : public Scatterer, 
                      public wrapper<Scatterer>
{

    public: 

    ScattererWrap() : Scatterer() {}

    ScattererWrap(const ScattererWrap& S) : Scatterer(S) {}

    void default_SetX(const float x) 
    { this->Scatterer::SetX(x);}

    void SetX(const float x)
    {
        if (override SetX = this->get_override("SetX")) 
        {
            SetX(x);
            return;
        }
        default_SetX(x);
    }

    void default_SetY(const float y) 
    { this->Scatterer::SetY(y);}

    void SetY(const float y)
    {
        if (override SetY = this->get_override("SetY")) 
        {
            SetY(y);
            return;
        }
        default_SetY(y);
    }

    void default_SetZ(const float z) 
    { this->Scatterer::SetZ(z);}

    void SetZ(const float z)
    {
        if (override SetZ = this->get_override("SetZ")) 
        {
            SetZ(z);
            return;
        }
        default_SetZ(z);
    }

    void default_SetOccupancy(const float occ) 
    { this->Scatterer::SetOccupancy(occ);}

    void SetOccupancy(const float occ)
    {
        if (override SetOccupancy = this->get_override("SetOccupancy")) 
        {
            SetOccupancy(occ);
            return;
        }
        default_SetOccupancy(occ);
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
        return this->get_override("GetComponentName")();
    }

    void Print() const
    {
        this->get_override("Print")();
    }

    std::ostream& POVRayDescription(std::ostream& os, 
            const CrystalPOVRayOptions& options) const
    {
        return this->get_override("POVRayDescription")();
    }

    void GLInitDisplayList(const bool noSymmetrics,
            const float xMin, const float xMax,
            const float yMin, const float yMax,
            const float zMin, const float zMax,
            const bool displayEnantiomer,
            const bool displayNames) const
    {
        this->get_override("GLInitDisplayList")();
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
bp::list _GetScatteringComponentList(Scatterer &s)
{
    const ScatteringComponentList& scl = s.GetScatteringComponentList();
    bp::list l;
    for(size_t i = 0; i < scl.GetNbComponent(); ++i)
    {
        l.append(scl(i));
    }

    return l;
}


} // anonymous namespace


BOOST_PYTHON_MODULE(_scatterer)
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
