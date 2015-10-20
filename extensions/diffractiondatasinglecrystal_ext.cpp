/*****************************************************************************
*
* pyobjcryst        Complex Modeling Initiative
*                   (c) 2015 Brookhaven Science Associates
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Kevin Knox
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::DiffractionDataSingleCrystal.
*
*****************************************************************************/

#include <boost/python.hpp>

#include <ObjCryst/ObjCryst/DiffractionDataSingleCrystal.h>

using namespace boost::python;
using namespace ObjCryst;


void wrap_diffractiondatasinglecrystal()
{

    class_<DiffractionDataSingleCrystal, bases<ScatteringData> >
        ("DiffractionDataSingleCrystal")
        /* Methods */
        .def("SetWavelength",
                (void (DiffractionDataSingleCrystal::*)(const double))
                &DiffractionDataSingleCrystal::SetWavelength)
        ;
}
