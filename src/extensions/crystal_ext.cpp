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
* boost::python bindings to ObjCryst::Crystal.
*
* Changes from ObjCryst::Crystal
* - CIFOutput accepts a python file-like object
* - CIFOutput has default mindist = 0, rather than 0.5
* - CalcDynPopCorr is not enabled, as the API states that this is for internal
*   use only.
* - GetScatteringComponentList returns an actual list.
*
* Other Changes
* - CreateCrystalFromCIF is placed here instead of in a seperate CIF module. This
*   method accepts a python file rather than a CIF object.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/manage_new_object.hpp>

#include <string>
#include <map>

#undef B0
#include <ObjCryst/ObjCryst/Crystal.h>
#include <ObjCryst/ObjCryst/CIF.h>
#include <ObjCryst/ObjCryst/UnitCell.h>

#include "python_streambuf.hpp"
#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Overloaded so that AddScatterer does not add NULL
void _AddScatterer(Crystal& crystal, Scatterer* scatt)
{
    if(NULL == scatt)
    {
        PyErr_SetString(PyExc_ValueError,
                "Cannot add nonexistant Scatterer");
        throw_error_already_set();
    }
    crystal.AddScatterer(scatt);
}

// Overloaded so that RemoveScatterer cannot delete the passed scatterer
void _RemoveScatterer(Crystal& crystal, Scatterer* scatt)
{
    if(NULL == scatt)
    {
        PyErr_SetString(PyExc_ValueError,
                "Cannot remove nonexistant Scatterer");
        throw_error_already_set();
    }
    crystal.RemoveScatterer(scatt, false);
}

// Overloaded so that AddScatteringPower does not add NULL
void _AddScatteringPower(Crystal& crystal, ScatteringPower* scattpow)
{
    if(NULL == scattpow)
    {
        PyErr_SetString(PyExc_ValueError,
                "Cannot add nonexistant ScatteringPower");
        throw_error_already_set();
    }
    crystal.AddScatteringPower(scattpow);
}


// Overloaded so that RemoveScatteringPower cannot delete the passed
// scatteringpower
void _RemoveScatteringPower(Crystal& crystal, ScatteringPower* scattpow)
{
    if(NULL == scattpow)
    {
        PyErr_SetString(PyExc_ValueError,
                "Cannot remove nonexistant ScatteringPower");
        throw_error_already_set();
    }
    crystal.RemoveScatteringPower(scattpow, false);
}

void _PrintMinDistanceTable(const Crystal& crystal,
        const double minDistance = 0.1)
{

    crystal.PrintMinDistanceTable(minDistance);
}

// We want to turn a ScatteringComponentList into an actual list
bp::list _GetScatteringComponentList(Crystal& c)
{
    const ScatteringComponentList& scl = c.GetScatteringComponentList();
    bp::list l;
    for(int i = 0; i < scl.GetNbComponent(); ++i)
    {
        l.append(scl(i));
    }

    return l;
}


void _CIFOutput(Crystal& c, bp::object output, double mindist)
{
    boost_adaptbx::python::ostream os(output);
    c.CIFOutput(os, mindist);
    os.flush();
}

// wrap the virtual functions that need it
class CrystalWrap : public Crystal, public wrapper<Crystal>
{

    public:

    CrystalWrap() : Crystal()
    {
        SetDeleteSubObjInDestructor(false);
        SetDeleteRefParInDestructor(false);
    }

    CrystalWrap(const Crystal& c) : Crystal(c)
    {
        SetDeleteSubObjInDestructor(false);
        SetDeleteRefParInDestructor(false);
    }

    CrystalWrap(const double a, const double b, const double c ,
            const std::string& sg)
        : Crystal(a, b, c, sg)
    {
        SetDeleteSubObjInDestructor(false);
        SetDeleteRefParInDestructor(false);
    }

    CrystalWrap(const double a, const double b, const double c ,
            const double alpha, const double beta, const double gamma,
            const std::string& sg)
        : Crystal(a, b, c, alpha, beta, gamma, sg)
    {
        SetDeleteSubObjInDestructor(false);
        SetDeleteRefParInDestructor(false);
    }

    const ScatteringComponentList& default_GetScatteringComponentList() const
    { return this->Crystal::GetScatteringComponentList(); }

    const ScatteringComponentList& GetScatteringComponentList() const
    {
        override f = this->get_override("GetScatteringComponentList");
        if (f)  return f();
        return default_GetScatteringComponentList();
    }

};

// Easier than exposing all the CIF classes
// Also allow oneScatteringPowerPerElement and connectAtoms

Crystal*
_CreateCrystalFromCIF(bp::object input,
        const bool oneScatteringPowerPerElement=false,
        const bool connectAtoms=false)
{
    // Reading a cif file creates some output. Let's redirect stdout to a junk
    // stream and then throw it away.
    ostringstream junk;
    swapstdout(junk);

    boost_adaptbx::python::streambuf sbuf(input);
    boost_adaptbx::python::streambuf::istream in(sbuf);
    ObjCryst::CIF cif(in);

    int idx0 = gCrystalRegistry.GetNb();

    const bool verbose = false;
    const bool checkSymAsXYZ = true;
    ObjCryst::CreateCrystalFromCIF(cif, verbose, checkSymAsXYZ,
            oneScatteringPowerPerElement, connectAtoms);

    int idx = gCrystalRegistry.GetNb();

    // Switch the stream buffer back
    swapstdout(junk);

    if(idx == idx0)
    {
        throw ObjCryst::ObjCrystException("Cannot create crystal from CIF");
    }
    idx--;

    ObjCryst::Crystal* c = &gCrystalRegistry.GetObj( idx );
    c->SetDeleteSubObjInDestructor(false);
    c->SetDeleteRefParInDestructor(false);

    return c;
}


} // namespace


void wrap_crystal()
{

    class_<CrystalWrap, bases<UnitCell>, boost::noncopyable>("Crystal")
        /* Constructors */
        .def(init<const double, const double, const double, const std::string&>(
            (bp::arg("a"), bp::arg("b"), bp::arg("c"),
            bp::arg("SpaceGroupId"))))
        .def(init<const double, const double, const double,
            const double, const double, const double,
            const std::string&>(
            (bp::arg("a"), bp::arg("b"), bp::arg("c"),
            bp::arg("alpha"), bp::arg("beta"), bp::arg("gamma"),
            bp::arg("SpaceGroupId"))))
        .def(init<const Crystal&>(bp::arg("oldCryst")))
        /* Methods */
        .def("AddScatterer", &_AddScatterer,
            with_custodian_and_ward<1,2,with_custodian_and_ward<2,1> >())
        .def("RemoveScatterer", &_RemoveScatterer)
        .def("GetNbScatterer", &Crystal::GetNbScatterer)
        .def("GetScatt",
            (Scatterer& (Crystal::*)(const std::string&)) &Crystal::GetScatt,
            return_internal_reference<>())
        .def("GetScatt",
            (Scatterer& (Crystal::*)(const long)) &Crystal::GetScatt,
            return_internal_reference<>())
        .def("GetScatterer",
            (Scatterer& (Crystal::*)(const std::string&)) &Crystal::GetScatt,
            return_internal_reference<>())
        .def("GetScatterer",
            (Scatterer& (Crystal::*)(const long)) &Crystal::GetScatt,
            return_internal_reference<>())
        .def("GetScattererRegistry", ( ObjRegistry<Scatterer>&
            (Crystal::*) ()) &Crystal::GetScattererRegistry,
            return_internal_reference<>())
        .def("GetScatteringPowerRegistry", ( ObjRegistry<ScatteringPower>&
            (Crystal::*) ()) &Crystal::GetScatteringPowerRegistry,
            return_internal_reference<>())
        .def("AddScatteringPower", &_AddScatteringPower,
            with_custodian_and_ward<1,2,with_custodian_and_ward<2,1> >())
        .def("RemoveScatteringPower", &_RemoveScatteringPower)
        .def("GetScatteringPower",
            (ScatteringPower& (Crystal::*)(const std::string&))
            &Crystal::GetScatteringPower,
            return_internal_reference<>())
        .def("GetMasterClockScatteringPower",
            &Crystal::GetMasterClockScatteringPower,
            return_value_policy<copy_const_reference>())
        .def("GetScatteringComponentList", &_GetScatteringComponentList)
        .def("GetClockScattCompList", &Crystal::GetClockScattCompList,
                return_value_policy<copy_const_reference>())
        .def("GetMinDistanceTable", &Crystal::GetMinDistanceTable,
                (bp::arg("minDistance")=1.0))
        .def("PrintMinDistanceTable", &_PrintMinDistanceTable,
                (bp::arg("minDistance")=1.0))
        //.def("CalcDynPopCorr", &Crystal::CalcDynPopCorr,
        //        (bp::arg("overlapDist")=1.0,
        //         bp::arg("mergeDist")=0.0))
        .def("ResetDynPopCorr", &Crystal::ResetDynPopCorr)
        .def("SetUseDynPopCorr", &Crystal::SetUseDynPopCorr)
        .def("GetUseDynPopCorr", &Crystal::GetUseDynPopCorr)
        .def("GetBumpMergeCost", &Crystal::GetBumpMergeCost)
        .def("SetBumpMergeDistance",
            (void (Crystal::*)(const ScatteringPower&, const ScatteringPower&, const double))
             &Crystal::SetBumpMergeDistance,
             (bp::arg("scatt1"),
              bp::arg("scatt2"),
              bp::arg("dist")=1.5))
        .def("SetBumpMergeDistance",
            (void (Crystal::*)
            (const ScatteringPower&, const ScatteringPower&, const double, const
             bool))
             &Crystal::SetBumpMergeDistance,
             (bp::arg("scatt1"),
              bp::arg("scatt2"),
              bp::arg("dist"),
              bp::arg("allowMerge")))
        .def("RemoveBumpMergeDistance", &Crystal::RemoveBumpMergeDistance)
        .def("GetBumpMergeParList", (Crystal::VBumpMergePar& (Crystal::*)())
            &Crystal::GetBumpMergeParList, return_internal_reference<>())
        .def("GetClockScattererList", &Crystal::GetClockScattererList,
                return_value_policy<copy_const_reference>())
        .def("CIFOutput", &_CIFOutput, (bp::arg("file"), bp::arg("mindist")=0))
        .def("AddBondValenceRo", &Crystal::AddBondValenceRo)
        .def("RemoveBondValenceRo", &Crystal::AddBondValenceRo)
        .def("GetBondValenceCost", &Crystal::GetBondValenceCost)
        .def("GetBondValenceRoList",
            (std::map< pair< const ScatteringPower *, const ScatteringPower * >, double > &
            (Crystal::*)()) &Crystal::GetBondValenceRoList,
            return_internal_reference<>())
        .def("ConnectAtoms", &Crystal::ConnectAtoms,
             (bp::arg("min_relat_dist")=0.4, bp::arg("max_relat_dist")=1.3,
              bp::arg("warnuser_fail")=false))
        ;


    class_<Crystal::BumpMergePar>("BumpMergePar", no_init)
        .def_readwrite("mDist2", &Crystal::BumpMergePar::mDist2)
        .def_readwrite("mCanOverlap", &Crystal::BumpMergePar::mCanOverlap)
        ;

    def("CreateCrystalFromCIF", &_CreateCrystalFromCIF,
            (bp::arg("file"), bp::arg("oneScatteringPowerPerElement")=false,
             bp::arg("connectAtoms")=false),
            return_value_policy<manage_new_object>());
}
