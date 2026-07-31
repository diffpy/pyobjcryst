/*****************************************************************************
*
* pyobjcryst
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************/

#ifndef PYOBJCRYST_POWDERPATTERN_DIFFRACTION_SHIM_HPP
#define PYOBJCRYST_POWDERPATTERN_DIFFRACTION_SHIM_HPP

#include <ObjCryst/ObjCryst/PowderPattern.h>

class PowderPatternDiffractionShim : public ObjCryst::PowderPatternDiffraction
{
  public:
    using ObjCryst::PowderPatternDiffraction::Prepare;
};

#endif
