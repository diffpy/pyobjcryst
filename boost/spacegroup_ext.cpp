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
* boost::python bindings to ObjCryst::SpaceGroup.
*
* $Id$
*
*****************************************************************************/

#include "ObjCryst/SpaceGroup.h"

#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

// We'll use a CrystMatrix and let the converter we wrote take care of the rest.
CrystMatrix<float> GetTranslationVectors(const SpaceGroup& sg)
{

    std::vector<SpaceGroup::TRx> tv = sg.GetTranslationVectors();


    std::vector<int> dims(2);
    dims[0] = tv.size();
    dims[1] = 3;
    CrystMatrix<float> data(dims[0], dims[1]);
    for(int row = 0; row < dims[0]; ++row)
    {
        for(int col = 0; col < dims[1]; ++col)
        {
            data(row, col) = tv[row].tr[col];
        }
    }
    return data;
}

}


BOOST_PYTHON_MODULE(_spacegroup)
{

    class_<SpaceGroup> ("SpaceGroup", init<>() )
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
