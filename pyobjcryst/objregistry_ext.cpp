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
* boost::python bindings to ObjCryst::ObjRegistry template class.
*
* Changes from ObjCryst++
* - DeleteAll not wrapped
* - C++ methods that can return const or non-const objects return non-const
*   objects in python.
*
* $Id$
*
*****************************************************************************/

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/to_python_converter.hpp>

#include <string>

#include "RefinableObj/RefinableObj.h"
#include "ObjCryst/Scatterer.h"
#include "ObjCryst/DiffractionDataSingleCrystal.h"

#include "helpers.h"

using namespace ObjCryst;
using namespace boost::python;

// Unfortunately there's no better way to do this
//typedef ObjRegistry<Scatterer> ORScatt;

namespace {

/* Wrap all the class methods for the template class */
template <class T>
void
wrapClass(class_<ObjRegistry<T> > & c)
{
    /* Functions */
    c.def("Register", &ObjRegistry<T>::Register)
    .def("DeRegister", (void (ObjRegistry<T>::*)(T&)) 
        &ObjRegistry<T>::DeRegister)
    .def("DeRegister", 
        (void (ObjRegistry<T>::*)(const string&)) 
        &ObjRegistry<T>::DeRegister)
    .def("DeRegisterAll", &ObjRegistry<T>::DeRegisterAll)
    // Dangerous and not wrapped
    //.def("DeleteAll", &ObjRegistry<T>::DeleteAll)
    .def("GetObj", (T& (ObjRegistry<T>::*)(const unsigned int)) 
        &ObjRegistry<T>::GetObj,
        return_internal_reference<>())
    .def("GetObj", 
        (T& (ObjRegistry<T>::*)(const string&)) &ObjRegistry<T>::GetObj,
        return_internal_reference<>())
    .def("GetObj", 
        (T& (ObjRegistry<T>::*)(const string&, const string&)) 
        &ObjRegistry<T>::GetObj,
        return_internal_reference<>())
    .def("GetNb", &ObjRegistry<T>::GetNb)
    .def("Print", &ObjRegistry<T>::Print)
    .def("SetName", &ObjRegistry<T>::SetName)
    .def("GetName", &ObjRegistry<T>::GetName,
        return_value_policy<copy_const_reference>())
    .def("Find", (long (ObjRegistry<T>::*)(const string&) const) 
        &ObjRegistry<T>::Find)
    .def("Find", 
        (long (ObjRegistry<T>::*)
        (const string&, const string&, const bool) const) 
        &ObjRegistry<T>::Find)
    .def("Find", (long (ObjRegistry<T>::*)(const T&) const) 
        &ObjRegistry<T>::Find)
    .def("GetRegistryClock", &ObjRegistry<T>::GetRegistryClock,
        return_value_policy<copy_const_reference>())
    // Python-only methods
    .def("__str__", &__str__< ObjRegistry<T> >)
    ;
}

} // end namespace


BOOST_PYTHON_MODULE(_objregistry)
{

    // ObjRegistry<RefinableObj>
    class_< ObjRegistry<RefinableObj> >
        RefinableObjRegistry("RefinableObjRegistry", 
        init<const string&>());
    wrapClass<RefinableObj>(RefinableObjRegistry);

    // ObjRegistry<Scatterer>
    class_< ObjRegistry<Scatterer> >
        ScattererRegistry("ScattererRegistry", 
        init<const string&>());
    wrapClass<Scatterer>(ScattererRegistry);

    // ObjRegistry<ScatteringPower>
    class_< ObjRegistry<ScatteringPower> >
        ScatteringPowerRegistry("ScatteringPowerRegistry", 
        init<const string&>());
    wrapClass<ScatteringPower>(ScatteringPowerRegistry);
}
