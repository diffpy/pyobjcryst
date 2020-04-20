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
* boost::python bindings to ObjCryst::ObjRegistry template class.
*
* Changes from ObjCryst::ObjRegistry
* - DeleteAll not wrapped
* - C++ methods that can return const or non-const objects return non-const
*   objects in python.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/iterator.hpp>
#include <boost/python/slice.hpp>
#undef B0

#include <string>

#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/ObjCryst/Atom.h>
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/DiffractionDataSingleCrystal.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>
#include <ObjCryst/ObjCryst/Scatterer.h>
#include <ObjCryst/ObjCryst/ZScatterer.h>

#include "helpers.hpp"

using namespace ObjCryst;
using namespace boost::python;

namespace {

// Get objects by slice
template <class T> bp::object getObjSlice(ObjRegistry<T>& o, bp::slice& s)
{
    bp::list l;

    for(typename std::vector<T*>::const_iterator it = o.begin(); it != o.end(); ++it)
    {
        l.append(bp::ptr(*it));
    }
    return l[s];
}



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
    .def("__len__", &ObjRegistry<T>::GetNb)
    .def("__getitem__", &getObjSlice<T>,
            with_custodian_and_ward_postcall<1,0>())
    .def("__getitem__", (T& (ObjRegistry<T>::*)(const unsigned int))
        &ObjRegistry<T>::GetObj,
        return_internal_reference<>())
    // Note to indexing robots: It took me a while to understand how this worked,
    // so that the object would be returned instead of the pointer !
    // The definition of NextPolicies use in boost.python range is quite obscure
    //
    // Unrelated note: this can be dangerous, as the registry is susceptible to
    // change while being iterated. Example
    //    c = pyobjcryst.crystal.Crystal(...)
    //    for c in pyobjcryst.crystal.gCrystalRegistry: print(c.GetName())
    // => this will erase the first crystal 'c' when looping other the registry,
    // which will effectively invalidate the iterator...
    .def("__iter__", range<return_value_policy<reference_existing_object> >
                       (&ObjRegistry<T>::list_begin, &ObjRegistry<T>::list_end))
    ;
}

} // end namespace


void wrap_objregistry()
{
    // Template instantiation

    // ObjRegistry<Crystal>
    class_< ObjRegistry<Crystal> >
        CrystalRegistry("CrystalRegistry",
        init<const string&>());
    wrapClass<Crystal>(CrystalRegistry);

    // ObjRegistry<PowderPattern>
    class_< ObjRegistry<PowderPattern> >
        PowderPatternRegistry("PowderPatternRegistry",
        init<const string&>());
    wrapClass<PowderPattern>(PowderPatternRegistry);

    // ObjRegistry<DiffractionDataSingleCrystal>
    class_< ObjRegistry<DiffractionDataSingleCrystal> >
        DiffractionDataSingleCrystalRegistry("DiffractionDataSingleCrystalRegistry",
        init<const string&>());
    wrapClass<DiffractionDataSingleCrystal>(DiffractionDataSingleCrystalRegistry);

    // ObjRegistry<OptimizationObj>
    class_< ObjRegistry<OptimizationObj> >
        OptimizationObjRegistry("OptimizationObjRegistry",
        init<const string&>());
    wrapClass<OptimizationObj>(OptimizationObjRegistry);

    // ObjRegistry<RefinableObj>
    class_< ObjRegistry<RefinableObj> >
        RefinableObjRegistry("RefinableObjRegistry",
        init<const string&>());
    wrapClass<RefinableObj>(RefinableObjRegistry);

    // ObjRegistry<RefObjOpt>
    class_< ObjRegistry<RefObjOpt> >
        RefObjOptRegistry("RefObjOpt",
        init<const string&>());
    wrapClass<RefObjOpt>(RefObjOptRegistry);

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

    // ObjRegistry<ScatteringPowerAtom>
    class_< ObjRegistry<ScatteringPowerAtom> >
        ScatteringPowerAtomRegistry("ScatteringPowerAtomRegistry",
        init<const string&>());
    wrapClass<ScatteringPowerAtom>(ScatteringPowerAtomRegistry);

    // ObjRegistry<ZAtom>
    class_< ObjRegistry<ZAtom> >
        ZAtomRegistry("ZAtomRegistry",
        init<const string&>());
    wrapClass<ZAtom>(ZAtomRegistry);

}
