/*****************************************************************************
*
* pyobjcryst        
*
* File coded by:    Vincent Favre-Nicolin
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* boost::python bindings to ObjCryst::PowderPatternPowderPatternComponent.
*
* Changes from ObjCryst::MonteCarloObj
*
* Other Changes
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/slice.hpp>

#include <string>
#include <map>

#include <ObjCryst/ObjCryst/General.h>
#include <ObjCryst/RefinableObj/RefinableObj.h>
#include <ObjCryst/CrystVector/CrystVector.h>
#include <ObjCryst/ObjCryst/PowderPattern.h>

#include "python_file_stream.hpp"
#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {
	class PowderPatternComponentWrap : public PowderPatternComponent, public wrapper<PowderPatternComponent>
	{//:TODO: :KLUDGE: Dummy override of pure virtual functions
		public:
			PowderPatternComponentWrap():PowderPatternComponent(){}
			virtual void SetParentPowderPattern(PowderPattern&){}
      	    virtual const CrystVector_REAL& GetPowderPatternCalc()const{}
      		virtual pair<const CrystVector_REAL*,const RefinableObjClock*> GetPowderPatternIntegratedCalc()const{}
      		virtual const CrystVector_REAL& GetPowderPatternCalcVariance()const{}
      		virtual pair<const CrystVector_REAL*,const RefinableObjClock*> GetPowderPatternIntegratedCalcVariance() const{}
      		virtual bool HasPowderPatternCalcVariance()const{}
      		virtual void CalcPowderPattern() const{};
      		virtual const CrystVector_long& GetBraggLimits()const{}
      		virtual void SetMaxSinThetaOvLambda(const REAL max){}
      		virtual void Prepare(){}
	};
}   // namespace

void wrap_powderpatterncomponent()
{
    class_<PowderPatternComponentWrap, bases<RefinableObj> , boost::noncopyable>("PowderPatternComponent", init<>())
    	;
}

