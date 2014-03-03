/*****************************************************************************
*
* PyObjCryst        by DANSE Diffraction group
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
* boost::python bindings to general structures and objects defined in
* ObjCryst/ObjCryst/General.h
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/python/def.hpp>

#include <ObjCryst/ObjCryst/General.h>

using namespace boost::python;
using namespace ObjCryst;

void wrap_general()
{
    enum_<RadiationType>("RadiationType")
        .value("RAD_NEUTRON", RAD_NEUTRON)
        .value("RAD_XRAY", RAD_XRAY)
        .value("RAD_ELECTRON", RAD_ELECTRON)
        ;
}
