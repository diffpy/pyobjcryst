# This SConstruct is for faster parallel builds.
# Use "setup.py" for normal installation.
#
# module     -- build the shared library object _pyobjcryst.so
# develop    -- install _pyobjcryst.so under the pyobjcryst/ directory

import os
import re
import platform

def subdictionary(d, keyset):
    return dict([kv for kv in d.items() if kv[0] in keyset])

# copy system environment variables related to compilation
DefaultEnvironment(ENV=subdictionary(os.environ, [
    'PATH', 'PYTHONPATH',
    'CPATH', 'CPLUS_INCLUDE_PATH',
    'LD_LIBRARY_PATH', 'LIBRARY_PATH',
    ])
)

# Create construction environment
env = DefaultEnvironment().Clone()

# Variables definitions below work only with 0.98 or later.
env.EnsureSConsVersion(0, 98)

# Customizable compile variables
vars = Variables('sconsvars.py')

vars.Add(PathVariable('prefix',
    'installation prefix directory', None))
vars.Add(EnumVariable('build',
    'compiler settings', 'debug',
    allowed_values=('debug', 'fast')))
vars.Add(BoolVariable('profile',
    'build with profiling information', False))
vars.Update(env)
env.Help(vars.GenerateHelpText(env))

# Insert LIBRARY_PATH explicitly because some compilers
# ignore it in the system environemnt.
env.PrependUnique(LIBPATH=env['ENV'].get('LIBRARY_PATH', '').split(':'))

# Use Intel C++ compiler when it is available
icpc = env.WhereIs('icpc')
if icpc:
    env.Tool('intelc', topdir=icpc[:icpc.rfind('/bin')])

# Note: If we merge in libObjCryst scripts, this should apply
# to a separate environment for Python modules.
# Figure out compilation switches, filter away C-related items.
good_python_flags = lambda n : (
    not re.match(r'(-g|-Wstrict-prototypes|-O\d)$', n))
env.ParseConfig("python-config --cflags")
env.Replace(CCFLAGS=filter(good_python_flags, env['CCFLAGS']))
env.Replace(CPPDEFINES='')
# Add shared libraries.
# Note: ObjCryst and boost_python are added from SConscript.configure
env.ParseConfig("python-config --ldflags")

fast_linkflags = ['-s']

# Platform specific intricacies.
if env['PLATFORM'] == 'darwin':
    fast_linkflags[:] = []

# Compiler specific options
if icpc:
    # options for Intel C++ compiler on hpc dev-intel07
    env.AppendUnique(CCFLAGS=['-w1', '-fp-model', 'precise'])
    env.PrependUnique(LIBS=['imf'])
    fast_optimflags = ['-fast', '-no-ipo']
else:
    # g++ options
    env.AppendUnique(CCFLAGS=['-Wall'])
    fast_optimflags = ['-ffast-math']

# Configure build variants
if env['build'] == 'debug':
    env.AppendUnique(CCFLAGS='-g')
elif env['build'] == 'fast':
    env.AppendUnique(CCFLAGS=['-O3'] + fast_optimflags)
    env.AppendUnique(CPPDEFINES='NDEBUG')
    env.AppendUnique(LINKFLAGS=fast_linkflags)

if env['profile']:
    env.AppendUnique(CCFLAGS='-pg')
    env.AppendUnique(LINKFLAGS='-pg')

builddir = env.Dir('build/%s-%s' % (env['build'], platform.machine()))
Export('env')

if os.path.isfile('sconscript.local'):
    env.SConscript('sconscript.local')

env.SConscript('extensions/SConscript', variant_dir=builddir)

# vim: ft=python
