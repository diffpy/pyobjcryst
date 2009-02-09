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
* boost::python bindings to ObjCryst::AsymmetricUnit.
*
* $Id$
*
*****************************************************************************/

// This is home to converters that are explicitly applied within the extensions,
// rather than registered in registerconverters.cpp.
#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <string>
#include <sstream>
#include <iostream>

// This function will help convert a class Print statement into a __str__
// statement. See refinableobjclock_ext.cpp for an example.
template <class T>
std::string __str__(const T &obj)
{
    // Switch the stream buffer with std::cout, which is used by Print.
    ostringstream outbuf;
    streambuf* cout_strbuf(cout.rdbuf());
    std::cout.rdbuf(outbuf.rdbuf());
    // Call Print()
    obj.Print();
    // Switch the stream buffer back
    cout.rdbuf(cout_strbuf);

    string outstr = outbuf.str();
    // Remove the trailing newline
    size_t idx = outstr.find_last_not_of("\n");
    if (idx != string::npos)
        outstr.erase(idx+1);

    return outstr;
}

#endif
