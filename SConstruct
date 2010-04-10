# Top level targets that are defined in subsidiary SConscripts
#
# lib               -- build shared library object
# install           -- install everything under prefix directory
# install-include   -- install C++ header files
# install-lib       -- install shared library object

import os

# copy system environment variables related to compilation
extenv = {
        'CPPPATH' : os.environ.get('CPLUS_INCLUDE_PATH', '').split(os.pathsep),
        'LIBPATH' : os.environ.get('LIBRARY_PATH', '').split(os.pathsep)
                  + os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep),
        }
extenv['RPATH'] = extenv['LIBPATH']
DefaultEnvironment(**extenv)

# Create construction environment
env = DefaultEnvironment().Clone()

# Variables definitions below work only with 0.98 or later.
env.EnsureSConsVersion(0, 98)

# Customizable compile variables
vars = Variables('sconsvars.py')

vars.Add(EnumVariable('build',
    'compiler settings', 'debug',
    allowed_values=('debug', 'fast')))
vars.Add(BoolVariable('profile',
    'build with profiling information', False))
vars.Add('REAL', 'floating point type', 'double')
vars.Add(PathVariable('prefix',
    'installation prefix directory', '/usr/local'))
vars.Update(env)
vars.Add(PathVariable('libdir',
    'object code library directory [prefix/lib]',
    env['prefix'] + '/lib',
    PathVariable.PathIsDirCreate))
vars.Add(PathVariable('includedir',
    'installation directory for C++ header files [prefix/include]',
    env['prefix'] + '/include',
    PathVariable.PathIsDirCreate))
vars.Update(env)
env.Help(vars.GenerateHelpText(env))

builddir = env.Dir('build/' + env['build'])

Export('env')

SConscript(["src/SConscript"], variant_dir=builddir)

# vim: ft=python
