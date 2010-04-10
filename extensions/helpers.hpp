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
* This is home to converters and utility functions that are explicitly applied
* within the extensions, rather than registered in registerconverters.cpp.
*
* $Id$
*
*****************************************************************************/

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <string>
#include <set>
#include <sstream>
#include <iostream>

#include <boost/python.hpp>

namespace bp = boost::python;

// Switch stdout with another stream. To get things back the right way, just
// switch again with the same stream.
void swapstdout(std::ostream& buf);

template <class T>
std::string __str__(const T &obj)
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
        l.append(ptr(*it));
    }
    return l;
}

template <class T>
bp::list setToPyList(std::set<T>& v)
{
    return containerToPyList< typename std::set<T> >(v);
}


#endif
