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
* Note that all indices are zero-based.
* 
* Changes from ObjCryst++
* - The public data are not wrapped. Accessors will be added as needed.
* - AddAtom returns the added MolAtom
* - AddBond returns the added MolBond
* - AddBondAngle returns the added MolBondAngle
* - AddDihedralAngle returns the added MolDihedralAngle
* - RemoveAtom returns None, has an indexed version
* - RemoveBond returns None, has an indexed version
* - RemoveBondAngle returns None, has an indexed version
* - RemoveDihedralAngle returns None, has an indexed version
* - RemoveRigidGroup returns None
* - Added GetNbAtoms
* - Added GetNbBonds
* - Added GetNbBondAngles
* - Added GetNbDihedralAngles
* - Added GetNbRigidGroups
* - Added GetBond
* - Added GetBondAngle
* - Added GetDihedralAngle
* - Added GetRigidGroup
* - FindBond returns the bond if found, None otherwise
* - FindBondAngle returns the bond angle if found, None otherwise
* - FindDihedralAngle returns the dihedral angle if found, None otherwise
* - FindAtom is identical to GetAtom.
*   FIXME
* - FlipAtomGroup is not wrapped.
* - FlipGroup, RotorGroup and StretchModeGroup are not wrapped.
* - StretchMode getters are not wrapped
*
* $Id$
*
*****************************************************************************/

#include "RefinableObj/RefinableObj.h"
#include "ObjCryst/Molecule.h"
#include "ObjCryst/Crystal.h"
#include "ObjCryst/ScatteringPower.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <set>
#include <map>
#include <algorithm>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// Overloaded to return added object
MolAtom& _AddAtom(Molecule& m, const float x, const float y, const float z,
        const ScatteringPower* pow, const std::string& name,
        const bool updateDisplay=true)
{
    m.AddAtom(x, y, z, pow, name, updateDisplay);
    std::vector<MolAtom*>& v = m.GetAtomList();
    return *v.back();
}

MolBond& _AddBond(Molecule& m, MolAtom& atom1, MolAtom& atom2, const float
        length, const float sigma, const float delta, const float bondOrder =
        1., const bool updateDisplay = true)
{
    m.AddBond(atom1, atom2, length, sigma, delta, bondOrder, updateDisplay);
    std::vector<MolBond*>& v = m.GetBondList();
    return *v.back();
}

MolBondAngle& _AddBondAngle(Molecule& m, MolAtom& atom1, MolAtom& atom2,
        MolAtom& atom3, const float angle, const float sigma, const float
        delta, const bool updateDisplay = true)
{
    m.AddBondAngle(atom1, atom2, atom3, angle, sigma, delta, updateDisplay);
    std::vector<MolBondAngle*>& v = m.GetBondAngleList();
    return *v.back();
}

MolDihedralAngle& _AddDihedralAngle(Molecule& m, MolAtom& atom1, MolAtom&
        atom2, MolAtom& atom3, MolAtom& atom4, const float angle, const float
        sigma, const float delta, const bool updateDisplay = true)
{
    m.AddDihedralAngle(atom1, atom2, atom3, atom4, angle, sigma, delta,
            updateDisplay);
    std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();
    return *v.back();
}

// New Functions
size_t _GetNbAtoms(Molecule& m)
{
    std::vector<MolAtom*>& v = m.GetAtomList();
    return v.size();
} 

size_t _GetNbBonds(Molecule& m)
{
    std::vector<MolBond*>& v = m.GetBondList();
    return v.size();
} 

size_t _GetNbBondAngles(Molecule& m)
{
    std::vector<MolBondAngle*>& v = m.GetBondAngleList();
    return v.size();
} 

size_t _GetNbDihedralAngles(Molecule& m)
{
    std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();
    return v.size();
} 

size_t _GetNbRigidGroups(Molecule& m)
{
    std::vector<RigidGroup*>& v = m.GetRigidGroupList();
    return v.size();
} 

// Overloaded for safety
MolAtom& _GetAtomIdx(Molecule& m, int idx)
{
    std::vector<MolAtom*>& v = m.GetAtomList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
} 

MolBond& _GetBondIdx(Molecule& m, int idx)
{
    std::vector<MolBond*>& v = m.GetBondList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
} 

MolBondAngle& _GetBondAngleIdx(Molecule& m, int idx)
{
    std::vector<MolBondAngle*>& v = m.GetBondAngleList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
} 

MolDihedralAngle& _GetDihedralAngleIdx(Molecule& m, int idx)
{
    std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();
    if(idx < 0) idx += v.size();
    if(0 == v.size() || idx < 0 || idx >= v.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw_error_already_set();
    }
    return *v[idx];
} 

// Overloaded for void return type and index access.
void _RemoveAtom(Molecule& m, MolAtom& ma)
{
    m.RemoveAtom(ma, false);
} 

void _RemoveAtomIdx(Molecule& m, int idx)
{
    m.RemoveAtom(_GetAtomIdx(m, idx), false);
} 

void _RemoveBond(Molecule& m, const MolBond& mb)
{
    m.RemoveBond(mb, false);
}

void _RemoveBondIdx(Molecule& m, int idx)
{
    m.RemoveBond(_GetBondIdx(m, idx), false);
}

void _RemoveBondAngle(Molecule& m, MolBondAngle& mba)
{
    m.RemoveBondAngle(mba, false);
}

void _RemoveBondAngleIdx(Molecule& m, int idx)
{
    m.RemoveBondAngle(_GetBondAngleIdx(m, idx), false);
}

void _RemoveDihedralAngle(Molecule& m, MolDihedralAngle& mda)
{
    m.RemoveDihedralAngle(mda, false);
}

void _RemoveDihedralAngleIdx(Molecule& m, int idx)
{
    m.RemoveDihedralAngle(_GetDihedralAngleIdx(m, idx), false);
}

// Overloaded for void return type
void _RemoveRigidGroup(Molecule& m, RigidGroup& mda, const bool ud=true)
{
    m.RemoveRigidGroup(mda, ud, false);
}

PyObject* _FindBond(const Molecule& m, const MolAtom& ma1, const MolAtom& ma2)
{
    std::vector<MolBond*>::const_iterator mbi;
    mbi = m.FindBond(ma1, ma2);
    const std::vector<MolBond*>& bondlist = m.GetBondList();
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
    const std::vector<MolBondAngle*>& bondanglelist = m.GetBondAngleList();
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
    const std::vector<MolDihedralAngle*>& dihedralanglelist 
        = m.GetDihedralAngleList();
    PyObject *retval;
    if(dihedralanglelist.end() == mdai)
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

// This could usually be done with an indexing suite, but it doesn't work with
// vectors of pointers.
bp::list _GetAtomList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolAtom*>& v = m.GetAtomList();

    l = ptrcontainerToPyList< const std::vector<MolAtom*> >(v);

    return l;
}

bp::list _GetBondList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolBond*>& v = m.GetBondList();

    l = ptrcontainerToPyList< const std::vector<MolBond*> >(v);

    return l;
}

bp::list _GetBondAngleList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolBondAngle*>& v = m.GetBondAngleList();

    l = ptrcontainerToPyList< const std::vector<MolBondAngle*> >(v);

    return l;
}

bp::list _GetDihedralAngleList(const Molecule& m)
{
    bp::list l;

    const std::vector<MolDihedralAngle*>& v = m.GetDihedralAngleList();

    l = ptrcontainerToPyList< const std::vector<MolDihedralAngle*> >(v);

    return l;
}

bp::list _GetStretchModeBondLengthList(const Molecule& m)
{
    bp::list l;

    const std::list<StretchModeBondLength>& v =
        m.GetStretchModeBondLengthList();

    l = containerToPyList< const std::list<StretchModeBondLength> >(v);

    return l;
}

bp::list _GetStretchModeBondAngleList(const Molecule& m)
{
    bp::list l;

    const std::list<StretchModeBondAngle>& v = m.GetStretchModeBondAngleList();

    l = containerToPyList< const std::list<StretchModeBondAngle> >(v);

    return l;
}

bp::list _GetStretchModeTorsionList(const Molecule& m)
{
    bp::list l;

    const std::list<StretchModeTorsion>& v = m.GetStretchModeTorsionList();

    l = containerToPyList< const std::list<StretchModeTorsion> >(v);

    return l;
}

bp::list _GetRigidGroupList(const Molecule& m)
{
    bp::list l;

    const std::vector<RigidGroup*>& v = m.GetRigidGroupList();

    l = ptrcontainerToPyList< const std::vector<RigidGroup*> >(v);

    for( size_t i = 0; i < v.size(); ++i)
    {
        l.append(bp::object(v[i]));
    }
    return l;
}

// Overloaded to accept a python iterable instead of a std::set. Again, could be
// done with converters, but there are issues with pointers. Perhaps another
// day...
void _RotateAtomGroup(Molecule &m, const MolAtom& at1, const MolAtom& at2,
    const bp::object& atoms, const float angle, const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyIterableToSet<MolAtom*>(atoms);
    m.RotateAtomGroup(at1, at2, catoms, angle, keepCenter);
}

void _RotateAtomGroupVec(Molecule &m, const MolAtom& at1, const float vx,
    const float vy, const float vz, const bp::object& atoms, const float angle,
    const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyIterableToSet<MolAtom*>(atoms);
    m.RotateAtomGroup(at1, vx, vy, vz, catoms, angle, keepCenter);
}

void _TranslateAtomGroup(Molecule &m, const bp::object& atoms, const float dx,
    const float dy, const float dz, const bool keepCenter=true)
{

    std::set<MolAtom*> catoms = pyIterableToSet<MolAtom*>(atoms);
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
        bp::object key(bp::ptr(miter->first));

        d[key] = ptrcontainerToPyList< const std::set<MolAtom*>
            >(miter->second);
    }

    return d;
}

bp::list _AsZMatrix(const Molecule& m, const bool keeporder)
{
    bp::list l;

    const std::vector<MolZAtom>& v = m.AsZMatrix(keeporder);

    l = containerToPyList< const std::vector<MolZAtom> >(v);

    return l;
}


} // namespace


BOOST_PYTHON_MODULE(_molecule)
{

    class_<Molecule, bases<Scatterer> > ("Molecule", 
        init<Crystal&, const string&> ((bp::arg("cryst"), bp::arg("name")=""))
        )
        //[with_custodian_and_ward<1,2>()])
        // The crystal is not used, so we don't need to manage it.
        /* Constructors */
        .def(init<const Molecule&>((bp::arg("old"))))
        /* Methods */
        .def("AddAtom", &_AddAtom,
            (bp::arg("x"), bp::arg("y"), bp::arg("z"), bp::arg("pPow"),
             bp::arg("name"), bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,5, return_internal_reference<> >())
        .def("RemoveAtom", &_RemoveAtom)
        .def("RemoveAtom", &_RemoveAtomIdx)
        .def("AddBond", &_AddBond,
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("length"),
             bp::arg("sigma"), bp::arg("delta"), bp::arg("bondOrder")=1,
             bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,2, 
            with_custodian_and_ward<1,3, 
            return_internal_reference<> > >())
        .def("RemoveBond", &_RemoveBond)
        .def("RemoveBond", &_RemoveBondIdx)
        .def("GetBond", &_GetBondIdx, return_internal_reference<>())
        .def("FindBond", &_FindBond, 
            with_custodian_and_ward_postcall<1,0>())
        .def("AddBondAngle", &_AddBondAngle, 
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("angle"), bp::arg("sigma"), bp::arg("delta"),
             bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,2, 
            with_custodian_and_ward<1,3, 
            with_custodian_and_ward<1,4, 
            return_internal_reference<> > > >())
        .def("RemoveBondAngle", &_RemoveBondAngle)
        .def("RemoveBondAngle", &_RemoveBondAngleIdx)
        .def("GetBondAngle", &_GetBondAngleIdx, return_internal_reference<>())
        .def("FindBondAngle", &_FindBondAngle,
            with_custodian_and_ward_postcall<1,0>())
        .def("AddDihedralAngle", &_AddDihedralAngle, 
            (bp::arg("atom1"), bp::arg("atom2"), bp::arg("atom3"),
             bp::arg("atom4"), bp::arg("angle"), bp::arg("sigma"),
             bp::arg("delta"), bp::arg("updateDisplay")=true),
            with_custodian_and_ward<1,2, 
            with_custodian_and_ward<1,3, 
            with_custodian_and_ward<1,4, 
            with_custodian_and_ward<1,5, 
            return_internal_reference<> > > > >())
        .def("RemoveDihedralAngle", &_RemoveDihedralAngle)
        .def("RemoveDihedralAngle", &_RemoveDihedralAngleIdx)
        .def("GetDihedralAngle", &_GetDihedralAngleIdx,
                return_internal_reference<>())
        .def("FindDihedralAngle", &_FindDihedralAngle,
            with_custodian_and_ward_postcall<1,0>())
        // An internal copy of the group is made, so there is no need for
        // lifetime management.
        .def("AddRigidGroup", &Molecule::AddRigidGroup)
        .def("RemoveRigidGroup", &_RemoveRigidGroup)
        .def("GetAtom", &_GetAtomIdx, return_internal_reference<>())
        .def("GetAtom", 
            (MolAtom& (Molecule::*)(const string&)) &Molecule::GetAtom, 
            return_internal_reference<>())
        .def("FindAtom", 
            (MolAtom& (Molecule::*)(const string&)) &Molecule::GetAtom, 
            return_internal_reference<>())
        .def("OptimizeConformation", &Molecule::OptimizeConformation,
            (bp::arg("nbTrial")=10000, bp::arg("stopCost")=0))
        .def("OptimizeConformationSteepestDescent", 
            &Molecule::OptimizeConformationSteepestDescent,
            (bp::arg("maxStep")=0.1, bp::arg("nbSteps")=1))
        .def("GetNbAtoms", &_GetNbAtoms)
        .def("GetNbBonds", &_GetNbBonds)
        .def("GetNbBondAngles", &_GetNbBondAngles)
        .def("GetNbDihedralAngles", &_GetNbDihedralAngles)
        .def("GetNbRigidGroups", &_GetNbRigidGroups)
        .def("GetAtomList", &_GetAtomList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondList", &_GetBondList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetBondAngleList", &_GetBondAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetDihedralAngleList", &_GetDihedralAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetRigidGroupList", &_GetRigidGroupList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeBondLengthList", &_GetStretchModeBondLengthList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeBondAngleList", &_GetStretchModeBondAngleList,
            with_custodian_and_ward_postcall<1,0>())
        .def("GetStretchModeTorsionList", &_GetStretchModeTorsionList,
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
        .def("BondLengthRandomChange", &Molecule::BondLengthRandomChange,
            (bp::arg("mode"), bp::arg("amplitude"),
             bp::arg("respectRestraint")=true)
            )
        .def("BondAngleRandomChange", &Molecule::BondAngleRandomChange,
            (bp::arg("mode"), bp::arg("amplitude"),
             bp::arg("respectRestraint")=true)
            )
        .def("DihedralAngleRandomChange", &Molecule::DihedralAngleRandomChange,
            (bp::arg("mode"), bp::arg("amplitude"),
             bp::arg("respectRestraint")=true)
            )
        .def("GetCenterAtom", &Molecule::GetCenterAtom,
            return_internal_reference<>())
        // Memory management shouldn't be necessary here, but there is the
        // possibility that a MolAtom that was created in another Molecule is
        // passed to this one. This could lead to memory corruption if the
        // original Molecule were to be deleted before this one, hence the
        // with_custodian_and_ward.
        .def("SetCenterAtom", &Molecule::SetCenterAtom,
            with_custodian_and_ward<1,2>())
        .def("AsZMatrix", &_AsZMatrix,
            with_custodian_and_ward_postcall<1,0>())
        .def("BuildRingList", &Molecule::BuildRingList)
        .def("BuildConnectivityTable", &Molecule::BuildConnectivityTable)
        .def("BuildRotorGroup", &Molecule::BuildRotorGroup)
        .def("TuneGlobalOptimRotationAmplitude", 
            &Molecule::TuneGlobalOptimRotationAmplitude)
        .def("BuildFlipGroup", &Molecule::BuildFlipGroup)
        .def("BuildStretchModeBondLength", 
            &Molecule::BuildStretchModeBondLength)
        .def("BuildStretchModeBondAngle", 
            &Molecule::BuildStretchModeBondAngle)
        .def("BuildStretchModeTorsion", 
            &Molecule::BuildStretchModeTorsion)
        .def("BuildStretchModeTwist", 
            &Molecule::BuildStretchModeTwist)
        .def("BuildStretchModeGroups", 
            &Molecule::BuildStretchModeGroups)
        .def("UpdateScattCompList", &Molecule::UpdateScattCompList)
        .def("InitOptions", &Molecule::InitOptions)
        ;
}

