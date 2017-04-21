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
* boost::python bindings to ObjCryst::SpaceGroup.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/tuple.hpp>

#include <ObjCryst/ObjCryst/SpaceGroup.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// This returns a list of translation operations
bp::list GetTranslationVectors(const SpaceGroup& sg)
{

    const std::vector<SpaceGroup::TRx>& tv = sg.GetTranslationVectors();

    bp::list outlist;
    std::vector<SpaceGroup::TRx>::const_iterator vec;
    for(vec = tv.begin(); vec != tv.end(); ++vec)
    {
        CrystVector<double> translation(3);
        for(int idx = 0; idx < 3; ++idx)
        {
            translation(idx) = vec->tr[idx];
        }
        outlist.append(translation);
    }
    return outlist;
}


// Returns a list of (translation vector, rotation) tuples
bp::list GetSymmetryOperations(const SpaceGroup& sg)
{

    const std::vector<SpaceGroup::SMx>& sv = sg.GetSymmetryOperations();

    bp::list outlist;
    int r, c;
    std::vector<SpaceGroup::SMx>::const_iterator tup;
    for(tup = sv.begin(); tup != sv.end(); ++tup)
    {
        CrystVector<double> translation(3);
        for(int idx = 0; idx < 3; ++idx)
        {
            translation(idx) = tup->tr[idx];
        }
        CrystMatrix<double> rotation(3,3);
        for(int idx = 0; idx < 9; ++idx)
        {
            r = idx/3;
            c = idx%3;
            rotation(r,c) = tup->mx[idx];
        }
        outlist.append(bp::make_tuple(translation, rotation));
    }
    return outlist;
}

}


void wrap_spacegroup()
{

    class_<SpaceGroup>("SpaceGroup")
        // Constructors
        .def(init<const std::string&>((bp::arg("spgId"))))
        // Methods
        .def("ChangeSpaceGroup", &SpaceGroup::ChangeSpaceGroup)
        .def("GetName", &SpaceGroup::GetName,
                return_value_policy<copy_const_reference>())
        .def("IsInAsymmetricUnit", &SpaceGroup::IsInAsymmetricUnit)
        .def("ChangeToAsymmetricUnit", &SpaceGroup::ChangeToAsymmetricUnit)
        .def("IsInAsymmetricUnit", &SpaceGroup::IsInAsymmetricUnit)
        .def("GetAsymUnit", &SpaceGroup::GetAsymUnit,
                return_internal_reference<>())
        .def("GetSpaceGroupNumber", &SpaceGroup::GetSpaceGroupNumber)
        .def("IsCentrosymmetric", &SpaceGroup::IsCentrosymmetric)
        .def("GetNbTranslationVectors", &SpaceGroup::GetNbTranslationVectors)
        .def("GetTranslationVectors", &GetTranslationVectors)
        .def("GetSymmetryOperations", &GetSymmetryOperations)
        .def("GetAllSymmetrics", &SpaceGroup::GetAllSymmetrics,
                (bp::arg("h"),
                 bp::arg("k"),
                 bp::arg("l"),
                 bp::arg("noCenter")=false,
                 bp::arg("noTransl")=false,
                 bp::arg("noIdentical")=false))
        .def("GetNbSymmetrics", &SpaceGroup::GetNbSymmetrics,
                 (bp::arg("noCenter")=false,
                 bp::arg("noTransl")=false))
        .def("GetInversionCenter", &SpaceGroup::GetInversionCenter)
        .def("Print", &SpaceGroup::Print)
        .def("HasInversionCenter", &SpaceGroup::HasInversionCenter)
        .def("IsInversionCenterAtOrigin",
                &SpaceGroup::IsInversionCenterAtOrigin)
        // Requires cctbx? Forward declaration doesn't work
        //.def("GetCCTbxSpg", &SpaceGroup::GetCCTbxSpg,
        //        return_value_policy<copy_const_reference>())
        .def("GetClockSpaceGroup", &SpaceGroup::GetClockSpaceGroup,
                return_value_policy<copy_const_reference>())
        .def("GetUniqueAxis", &SpaceGroup::GetUniqueAxis)
        .def("GetExtension", &SpaceGroup::GetExtension)
        .def("GetAllEquivRefl", &SpaceGroup::GetAllEquivRefl,
                (bp::arg("h"),
                 bp::arg("k"),
                 bp::arg("l"),
                 bp::arg("excludeFriedelMate")=false,
                 bp::arg("forceFriedelLaw")=false))
        .def("IsReflSystematicAbsent", &SpaceGroup::IsReflSystematicAbsent)
        .def("IsReflCentric", &SpaceGroup::IsReflCentric)
        .def("GetExpectedIntensityFactor", &SpaceGroup::GetExpectedIntensityFactor)
        .def("__str__", &__str__<SpaceGroup>)
        ;
}
