# This SConstruct is for faster parallel builds.
# Use "setup.py" for normal installation.

MY_SCONS_HELP = """\
SCons rules for compiling and installing pyobjcryst.
SCons build is much faster when run with parallel jobs (-j4).
Usage: scons [target] [var=value]

Targets:

module      build Python extension module _pyobjcryst.so [default]
install     install to default Python package location
develop     copy extension module to pyobjcryst/ directory
test        execute unit tests

Build configuration variables:
%s
Variables can be also assigned in a user script sconsvars.py.
SCons construction environment can be customized in sconscript.local script.
"""

import os
import re
import subprocess
import platform

def subdictionary(d, keyset):
    return dict([kv for kv in d.items() if kv[0] in keyset])

def getsyspaths(*names):
    s = os.pathsep.join(filter(None, map(os.environ.get, names)))
    return filter(os.path.exists, s.split(os.pathsep))

def pyconfigvar(name):
    cmd = [env['python'], '-c', '\n'.join((
            'from distutils.sysconfig import get_config_var',
            'print get_config_var(%r)' % name))]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    return out.rstrip()

# copy system environment variables related to compilation
DefaultEnvironment(ENV=subdictionary(os.environ, '''
    PATH PYTHONPATH
    LD_LIBRARY_PATH DYLD_LIBRARY_PATH LIBRARY_PATH
    '''.split())
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
    'compiler settings', 'fast',
    allowed_values=('debug', 'fast')))
vars.Add(BoolVariable('profile',
    'build with profiling information', False))
vars.Add('python',
    'Python executable to use for installation.', 'python')
vars.Update(env)
env.Help(MY_SCONS_HELP % vars.GenerateHelpText(env))

# Use Intel C++ compiler when it is available
icpc = env.WhereIs('icpc')
if icpc:
    env.Tool('intelc', topdir=icpc[:icpc.rfind('/bin')])

# Figure out compilation switches, filter away C-related items.
good_python_flags = lambda n : (
    not isinstance(n, basestring) or
    not re.match(r'(-g|-Wstrict-prototypes|-O\d)$', n))
env.ParseConfig("python-config --cflags")
env.Replace(CCFLAGS=filter(good_python_flags, env['CCFLAGS']))
env.Replace(CPPDEFINES='')
# the CPPPATH directories are checked by scons dependency scanner
cpppath = getsyspaths('CPLUS_INCLUDE_PATH', 'CPATH')
env.AppendUnique(CPPPATH=cpppath)
# Insert LIBRARY_PATH explicitly because some compilers
# ignore it in the system environment.
env.PrependUnique(LIBPATH=getsyspaths('LIBRARY_PATH'))
# Add shared libraries.
# Note: ObjCryst and boost_python are added from SConscript.configure

fast_linkflags = ['-s']
fast_shlinkflags = pyconfigvar('LDSHARED').split()[1:]

# Platform specific intricacies.
if env['PLATFORM'] == 'darwin':
    darwin_shlinkflags = [n for n in env['SHLINKFLAGS']
            if n != '-dynamiclib']
    env.Replace(SHLINKFLAGS=darwin_shlinkflags)
    env.AppendUnique(SHLINKFLAGS=['-bundle'])
    env.AppendUnique(SHLINKFLAGS=['-undefined', 'dynamic_lookup'])
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
    env.AppendUnique(SHLINKFLAGS=fast_shlinkflags)

if env['profile']:
    env.AppendUnique(CCFLAGS='-pg')
    env.AppendUnique(LINKFLAGS='-pg')

builddir = env.Dir('build/%s-%s' % (env['build'], platform.machine()))

Export('env', 'pyconfigvar')

if os.path.isfile('sconscript.local'):
    env.SConscript('sconscript.local')

env.SConscript('extensions/SConscript', variant_dir=builddir)

# vim: ft=python
