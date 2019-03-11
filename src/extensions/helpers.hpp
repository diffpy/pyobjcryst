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
* This is home to converters and utility functions that are explicitly applied
* within the extensions, rather than registered in registerconverters.cpp.
*
*****************************************************************************/

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <boost/python/object.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>

#include <string>
#include <set>
#include <sstream>
#include <iostream>
#include <limits>

namespace bp = boost::python;

typedef std::numeric_limits<double> doublelim;

class MuteObjCrystUserInfo
{
    public:

        MuteObjCrystUserInfo();
        ~MuteObjCrystUserInfo();
        void release();

    private:

        // pointer to the previous info function
        void (*msave_info_func)(const std::string &);
};


// Switch stdout with another stream. To get things back the right way, just
// switch again with the same stream.
void swapstdout(std::ostream& buf);

template <class T>
std::string __str__(const T& obj)
{
    // Switch the stream buffer with std::cout, which is used by Print.
    std::ostringstream outbuf;
    swapstdout(outbuf);
    // Call Print()
    obj.Print();
    // Switch the stream buffer back
    swapstdout(outbuf);

    std::string outstr = outbuf.str();
    // Remove the trailing newline
    size_t idx = outstr.find_last_not_of("\n");
    if (idx != std::string::npos)
        outstr.erase(idx+1);

    return outstr;
}

template <class T>
std::set<T> pyIterableToSet(const bp::object& l)
{

    std::set<T> cl;
    T typeobj;

    for(int i=0; i < len(l); ++i)
    {
        typeobj = bp::extract<T>(l[i]);
        cl.insert(typeobj);
    }

    return cl;
}

// For turning vector-like containers into lists
// It is assumed that T contains non-pointers
template <class T>
bp::list containerToPyList(T& v)
{
    bp::list l;

    for(typename T::const_iterator it = v.begin(); it != v.end(); ++it)
    {
        l.append(*it);
    }
    return l;
}

// For turning vector-like containers into lists
// It is assumed that T contains pointers
template <class T>
bp::list ptrcontainerToPyList(T& v)
{
    bp::list l;

    for(typename T::const_iterator it = v.begin(); it != v.end(); ++it)
    {
        l.append(bp::ptr(*it));
    }
    return l;
}

template <class T>
bp::list setToPyList(std::set<T>& v)
{
    return containerToPyList< typename std::set<T> >(v);
}


// Extract CrystVector from a Python object
template <class T> class CrystVector;

void assignCrystVector(CrystVector<double>& cv, bp::object obj);

#endif
