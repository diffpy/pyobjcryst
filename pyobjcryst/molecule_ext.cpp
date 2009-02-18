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
* boost::python bindings to ObjCryst::Molecule.  
* 
* Changes from ObjCryst++
* - RemoveAtom returns None
* - RemoveBond returns None
* - RemoveBondAngle returns None
* - RemoveDihedralAngle returns None
* - RemoveRigidGroup returns None
* - FindBond returns the bond if found, None otherwise
* - FindBondAngle returns the bond angle if found, None otherwise
* - FindDihedralAngle returns the dihedral angle if found, None otherwise
* - FindAtom is not wrapped, as it would be wrapped to behave as GetAtom.
*
* $Id$
*
*****************************************************************************/

#include "RefinableObj/RefinableObj.h"
#include "ObjCryst/Molecule.h"
#include "ObjCryst/Crystal.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <vector>
#include <list>
#include <set>
#include <map>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Overloaded for void return type
void _RemoveAtom(Molecule& m, MolAtom& ma)
{
    m.RemoveAtom(ma);
}

// Overloaded for void return type
void _RemoveBond(Molecule& m, const MolBond& mb)
{
    m.RemoveBond(mb);
}

// Overloaded for void return type
void _RemoveBondAngle(Molecule& m, MolBondAngle& mba)
{
    m.RemoveBondAngle(mba);
}

// Overloaded for void return type
void _RemoveDihedralAngle(Molecule& m, MolDihedralAngle& mda)
{
    m.RemoveDihedralAngle(mda);
}

// Overloaded for void return type
void _RemoveRigidGroup(Molecule& m, RigidGroup& mda, const bool ud=true)
{
    m.RemoveRigidGroup(mda, ud);
}

PyObject* _FindBond(const Molecule& m, const MolAtom& ma1, const MolAtom& ma2)
{
    std::vector<MolBond*>::const_iterator mbi;
    mbi = m.FindBond(ma1, ma2);
    const std::vector<MolBond*> bondlist = m.GetBondList();
    PyObject *retval;
    if(bondlist.end() == mbi)
    {
        // return None
        retval = Py_None;
    }
    else
    {
        reference_existing_object::apply<MolBond*>::type converter;
        retval = converter(*mbi);
        
    }
    bp::incref(retval);
    return retval;
}

PyObject* _FindBondAngle(const Molecule& m, const MolAtom& ma1, const MolAtom&
    ma2, const MolAtom& ma3)
{
    std::vector<MolBondAngle*>::const_iterator mbai;
    mbai = m.FindBondAngle(ma1, ma2, ma3);
    const std::vector<MolBondAngle*> bondanglelist = m.GetBondAngleList();
    PyObject *retval;
    if(bondanglelist.end() == mbai)
    {
        // return None
        retval = Py_None;
    }
    else
    {
        reference_existing_object::apply<MolBondAngle*>::type converter;
        retval = converter(*mbai);
    }
    bp::incref(retval);
    return retval;
}

PyObject* _FindDihedralAngle(const Molecule& m, const MolAtom& ma1, 
    const MolAtom& ma2, const MolAtom& ma3, const MolAtom& ma4)
{
    std::vector<MolDihedralAngle*>::const_iterator mdai;
    mdai = m.FindDihedralAngle(ma1, ma2, ma3, ma4);
    const std::vector<MolDihedralAngle*> bondanglelist 
        = m.GetDihedralAngleList();
    PyObject *retval;
    if(bondanglelist.end() == mdai)
    {
        // return None
        retval = Py_None;
    }
    else
    {
        reference_existing_object::apply<MolDihedralAngle*>::type converter;
        retval = converter(*mdai);
    }
    bp::incref(retval);
    return retval;
}

// This could usually be done with an indexing suite, but I couldn't get it to
// work with a vector of pointers. This is just as well, as we don't want to
// allow the user to modify the internal vector.
bp::list _GetAtomList(const Molecule& m)
{
    bp::list l;

    std::vector<MolAtom*> v = m.GetAtomList();

    l = containerToPyList< std::vector<MolAtom*> >(v);

    return l;
}

bp::list _GetBondList(const Molecule& m)
{
    bp::list l;

    std::vector<MolBond*> v = m.GetBondList();

    l = containerToPyList< std::vector<MolBond*> >(v);

    return l;
}

bp::list _GetBondAngleList(const Molecule& m)
{
    bp::list l;

    std::vector<MolBondAngle*> v = m.GetBondAngleList();

    l = containerToPyList< std::vector<MolBondAngle*> >(v);

    return l;
}

bp::list _GetDihedralAngleList(const Molecule& m)
{
    bp::list l;

    std::vector<MolDihedralAngle*> v = m.GetDihedralAngleList();

    l = containerToPyList< std::vector<MolDihedralAngle*> >(v);

    return l;
}

bp::list _GetStretchModeBondLengthList(const Molecule& m)
{
    bp::list l;

    std::list<StretchModeBondLength> v = m.GetStretchModeBondLengthList();

    l = containerToPyList< std::list<StretchModeBondLength> >(v);

    return l;
}

bp::list _GetStretchModeBondAngleList(const Molecule& m)
{
    bp::list l;

    std::list<StretchModeBondAngle> v = m.GetStretchModeBondAngleList();

    l = containerToPyList< std::list<StretchModeBondAngle> >(v);

    return l;
}

bp::list _GetStretchModeTorsionList(const Molecule& m)
{
    bp::list l;

    std::list<StretchModeTorsion> v = m.GetStretchModeTorsionList();

    l = containerToPyList< std::list<StretchModeTorsion> >(v);

    return l;
}

bp::list _GetRigidGroupList(const Molecule& m)
{
    bp::list l;

    std::vector<RigidGroup*> v = m.GetRigidGroupList();

    l = containerToPyList< std::vector<RigidGroup*> >(v);

    for( size_t i = 0; i < v.size(); ++i)
    {
        l.append(bp::object(v[i]));
    }
    return l;
}

// Overloaded to accept a python list instead of a std::set
void _RotateAtomGroup(Molecule &m, const MolAtom& at1, const MolAtom& at2,
    const bp::list& atoms, const float angle, const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyListToSet<MolAtom*>(atoms);
    m.RotateAtomGroup(at1, at2, catoms, angle, keepCenter);
}

void _RotateAtomGroupVec(Molecule &m, const MolAtom& at1, const float vx,
    const float vy, const float vz, const bp::list& atoms, const float angle,
    const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyListToSet<MolAtom*>(atoms);
    m.RotateAtomGroup(at1, vx, vy, vz, catoms, angle, keepCenter);
}

void _TranslateAtomGroup(Molecule &m, const bp::list& atoms, const float dx,
    const float dy, const float dz, const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyListToSet<MolAtom*>(atoms);
    m.TranslateAtomGroup(catoms, dx, dy, dz, keepCenter);
}


bp::dict _GetConnectivityTable(Molecule &m)
{

    const std::map<MolAtom*, std::set<MolAtom*> >& ct 
        = m.GetConnectivityTable();

    std::map<MolAtom*, std::set<MolAtom*> >::const_iterator miter;

    bp::dict d;

    for(miter = ct.begin(); miter != ct.end(); ++miter)
    {
        bp::object key(miter->first);
        d[key] = containerToPyList< const std::set<MolAtom*> >(miter->second);
    }

    return d;
}


} // namespace


BOOST_PYTHON_MODULE(_molecule)
{

    class_<Molecule, bases<Scatterer> > ("Molecule", 
        init<Crystal&, const string&> ((bp::arg("cryst"), bp::arg("name")="")))
        //[with_custodian_and_ward<2,1>()])
        // The crystal is not used, so we don't need to manage it.
        /* Constructors */
        .def(init<const Molecule&>((bp::arg("old"))))
        /* Methods */
        .def("AddAtom", &Molecule::AddAtom, 
            (bp::arg("x"), bp::arg("y"), bp::arg("z"), bp::arg("pPow"),
             bp::arg("name"), bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,5>())
        .def("RemoveAtom", &_RemoveAtom)
        .def("AddBond", &Molecule::AddBond,
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("length"),
             bp::arg("sigma"), bp::arg("delta"), bp::arg("bondOrder")=1,
             bp::arg("updateDisplay")=true))
        .def("RemoveBond", &_RemoveBond)
        .def("FindBond", &_FindBond, 
            with_custodian_and_ward_postcall<1,0>())
        .def("AddBondAngle", &Molecule::AddBondAngle, 
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("angle"), bp::arg("sigma"), bp::arg("delta"),
             bp::arg("updateDisplay")=true))
        .def("RemoveBondAngle", &_RemoveBondAngle)
        .def("FindBondAngle", &_FindBondAngle,
            with_custodian_and_ward_postcall<1,0>())
        .def("AddDihedralAngle", &Molecule::AddDihedralAngle, 
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("atom4"), bp::arg("angle"), bp::arg("sigma"),
             bp::arg("delta"), bp::arg("updateDisplay")=true))
        .def("RemoveDihedralAngle", &_RemoveDihedralAngle)
        .def("FindDihedralAngle", &_FindDihedralAngle,
            with_custodian_and_ward_postcall<1,0>())
        // An internal copy of the group is made, so there is no need for
        // lifetime management.
        // FIXME - define RigidGroup
        .def("AddRigidGroup", &Molecule::AddRigidGroup)
        .def("RemoveRigidGroup", &_RemoveRigidGroup)
        .def("GetAtom", 
            (MolAtom& (Molecule::*)(unsigned int)) &Molecule::GetAtom, 
            return_internal_reference<>())
        .def("GetAtom", 
            (MolAtom& (Molecule::*)(const string&)) &Molecule::GetAtom, 
            return_internal_reference<>())
        .def("OptimizeConformation", &Molecule::OptimizeConformation,
            (bp::arg("nbTrial")=10000, bp::arg("stopCost")=0))
        .def("OptimizeConformationSteepestDescent", 
            &Molecule::OptimizeConformationSteepestDescent,
            (bp::arg("maxStep")=0.1, bp::arg("nbSteps")=1))
        // Doesn't work
        .def("GetAtomList", &_GetAtomList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondList", &_GetBondList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondAngleList", &_GetBondAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetDihedralAngleList", &_GetDihedralAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeBondLengthList", &_GetStretchModeBondLengthList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeBondAngleList", &_GetStretchModeBondAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeTorsionList", &_GetStretchModeTorsionList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetRigidGroupList", &_GetRigidGroupList,
            with_custodian_and_ward_postcall<1,0>())
        .def("RotateAtomGroup", &_RotateAtomGroup, 
            (bp::arg("at1"), bp::arg("at2"), bp::arg("atoms"), bp::arg("angle"),
             bp::arg("keepCenter")=true
             )
            )
        .def("RotateAtomGroup", &_RotateAtomGroupVec,
            (bp::arg("at1"), bp::arg("vx"), bp::arg("vy"), bp::arg("vz"),
             bp::arg("atoms"), bp::arg("angle"), bp::arg("keepCenter")=true
             )
            )
        .def("TranslateAtomGroup", &_TranslateAtomGroup,
            (bp::arg("atoms"), bp::arg("dx"), bp::arg("dy"), bp::arg("dz"),
             bp::arg("keepCenter")=true
             )
            )
        .def("GetConnectivityTable", &_GetConnectivityTable, 
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondListClock", (RefinableObjClock& (Molecule::*)())
            &Molecule::GetBondListClock,
            return_internal_reference<>())
        .def("GetAtomPositionClock", (RefinableObjClock& (Molecule::*)())
            &Molecule::GetAtomPositionClock,
            return_internal_reference<>())
        .def("GetRigidGroupClock", (RefinableObjClock& (Molecule::*)())
            &Molecule::GetRigidGroupClock,
            return_internal_reference<>())
        .def("RigidifyWithDihedralAngles", 
            &Molecule::RigidifyWithDihedralAngles)
        ;

}
