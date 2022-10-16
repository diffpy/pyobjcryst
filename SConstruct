# This SConstruct is for faster parallel builds.
# Use "setup.py" for normal installation.

MY_SCONS_HELP = """\
SCons rules for compiling and installing pyobjcryst.
SCons build is much faster when run with parallel jobs (-j4).
Usage: scons [target] [var=value]

Targets:

module      build Python extension module _pyobjcryst.so [default]
install     install to default Python package location
develop     copy extension module to src/pyobjcryst/ directory
test        execute unit tests

Build configuration variables:
%s
Variables can be also assigned in a user script sconsvars.py.
SCons construction environment can be customized in sconscript.local script.
"""

import os
from os.path import join as pjoin
import re
import subprocess
import platform


def subdictionary(d, keyset):
    return dict(kv for kv in d.items() if kv[0] in keyset)


def getsyspaths(*names):
    pall = sum((os.environ.get(n, '').split(os.pathsep) for n in names), [])
    rv = [p for p in pall if os.path.exists(p)]
    return rv


def pyoutput(cmd):
    proc = subprocess.Popen([env['python'], '-c', cmd],
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
    out = proc.communicate()[0]
    return out.rstrip()


def pyconfigvar(name):
    cmd = ('from distutils.sysconfig import get_config_var\n'
           'print(get_config_var(%r))\n') % name
    return pyoutput(cmd)


# copy system environment variables related to compilation
DefaultEnvironment(ENV=subdictionary(os.environ, '''
    PATH PYTHONPATH GIT_DIR
    CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_RUN_PATH
    LD_LIBRARY_PATH DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH
    MACOSX_DEPLOYMENT_TARGET LANG
    _PYTHON_SYSCONFIGDATA_NAME
    _CONDA_PYTHON_SYSCONFIGDATA_NAME
    '''.split())
                   )

# Create construction environment
env = DefaultEnvironment().Clone()

# Variables definitions below work only with 0.98 or later.
env.EnsureSConsVersion(0, 98)

# Customizable compile variables
vars = Variables('sconsvars.py')

# Set PREFIX for installation and linking
# TODO: also amend paths when VIRTUAL_ENV variable exists ?
if 'PREFIX' in os.environ:
    # building with a set prefix
    vars.Add(PathVariable(
        'prefix',
        'installation prefix directory',
        os.environ['PREFIX']))
elif 'CONDA_PREFIX' in os.environ:
    # building for a conda environment
    vars.Add(PathVariable(
        'prefix',
        'installation prefix directory',
        os.environ['CONDA_PREFIX']))
else:
    vars.Add(PathVariable('prefix',
                          'installation prefix directory', None))
vars.Update(env)

vars.Add(EnumVariable('build',
                      'compiler settings', 'fast',
                      allowed_values=('debug', 'fast')))
vars.Add(EnumVariable('tool',
                      'C++ compiler toolkit to be used', 'default',
                      allowed_values=('default', 'intelc', 'clang', 'clangxx')))
vars.Add(BoolVariable('profile',
                      'build with profiling information', False))
vars.Add('python',
         'Python executable to use for installation.', 'python')
vars.Update(env)
env.Help(MY_SCONS_HELP % vars.GenerateHelpText(env))

# Use Intel C++ compiler if requested by the user.
icpc = None
if env['tool'] == 'intelc':
    icpc = env.WhereIs('icpc')
    if not icpc:
        print("Cannot find the Intel C/C++ compiler 'icpc'.")
        Exit(1)
    env.Tool('intelc', topdir=icpc[:icpc.rfind('/bin')])

# Figure out compilation switches, filter away C-related items.
good_python_flag = lambda n: (
        not isinstance(n, str) or
        not re.match(r'(-g|-Wstrict-prototypes|-O\d|-fPIC)$', n))

# Determine python-config script name.
if 'PY_VER' in os.environ:
    pyversion = os.environ['PY_VER']
else:
    pyversion = pyoutput('import sys; print("%i.%i" % sys.version_info[:2])')

if 'CONDA_BUILD' in os.environ and 'PY_VER' in os.environ:
    # Messy: if CONDA_BUILD and PY_VER are in the path, we are building a conda package
    # using several environment. Make sure python3.X-config points to the destination
    # (host) environment
    pythonconfig = pjoin(os.environ['PREFIX'], 'bin', 'python%s-config' % os.environ['PY_VER'])
    print("Using $PREFIX and $PY_VER to determine python-config pth: %s" % pythonconfig)
    xpython = pjoin(os.environ['PREFIX'], 'bin', 'python')
    pyversion = os.environ['PY_VER']
else:
    pycfgname = 'python%s-config' % (pyversion if pyversion[0] == '3' else '')
    # realpath gets the real path if exec is a link (e.g. in a python environment)
    xpython = os.path.realpath(env.WhereIs(env['python']))
    pybindir = os.path.dirname(xpython)
    pythonconfig = pjoin(pybindir, pycfgname)

# for k in sorted(os.environ.keys()):
#     print("   ", k, os.environ[k])

if platform.system().lower() == "windows":
    # See https://scons.org/faq.html#Linking_on_Windows_gives_me_an_error
    env['ENV']['TMP'] = os.environ['TMP']
    # the CPPPATH directories are checked by scons dependency scanner
    cpppath = getsyspaths('CPLUS_INCLUDE_PATH', 'CPATH')
    env.AppendUnique(CPPPATH=cpppath)
    # Insert LIBRARY_PATH explicitly because some compilers
    # ignore it in the system environment.
    env.PrependUnique(LIBPATH=getsyspaths('LIBRARY_PATH'))
    if env['prefix'] is not None:
        env.Append(CPPPATH=[pjoin(env['prefix'], 'include')])
        env.Append(CPPPATH=[pjoin(env['prefix'], 'Library', 'include')])
        # Windows conda library paths are a MESS ('lib', 'libs', 'Library\lib'...)
        env.Append(LIBPATH=[pjoin(env['prefix'], 'Library', 'lib')])
        env.Append(LIBPATH=[pjoin(env['prefix'], 'libs')])
    # This disable automated versioned named e.g. libboost_date_time-vc142-mt-s-x64-1_73.lib
    # so we can use conda-installed libraries
    env.AppendUnique(CPPDEFINES='BOOST_ALL_NO_LIB')
    # Prevent the generation of an import lib (.lib) in addition to the dll
    # env.AppendUnique(no_import_lib=1)
    env.PrependUnique(CCFLAGS=['/Ox', '/EHsc', '/MD'])
    env.AppendUnique(CPPDEFINES={'NDEBUG': None})
else:
    if 'CONDA_BUILD' not in os.environ:
        # Verify python-config comes from the same path as the target python.
        xpythonconfig = env.WhereIs(pythonconfig)
        if os.path.dirname(xpython) != os.path.dirname(xpythonconfig):
            print("Inconsistent paths of %r and %r" % (xpython, xpythonconfig))
            Exit(1)
    # Process the python-config flags here.
    env.ParseConfig(pythonconfig + " --cflags")
    env.Replace(CCFLAGS=[f for f in env['CCFLAGS'] if good_python_flag(f)])
    env.Replace(CPPDEFINES='BOOST_ERROR_CODE_HEADER_ONLY')
    # the CPPPATH directories are checked by scons dependency scanner
    cpppath = getsyspaths('CPLUS_INCLUDE_PATH', 'CPATH')
    env.AppendUnique(CPPPATH=cpppath)
    # Insert LIBRARY_PATH explicitly because some compilers
    # ignore it in the system environment.
    env.PrependUnique(LIBPATH=getsyspaths('LIBRARY_PATH'))
    # Add shared libraries.
    # Note: ObjCryst and boost_python are added from SConscript.configure.

    fast_linkflags = ['-s']
    fast_shlinkflags = pyconfigvar('LDSHARED').split()[1:]

    # Specify minimum C++ standard.  Allow later standard from sconscript.local.
    # In case of multiple `-std` options the last option holds.
    env.PrependUnique(CXXFLAGS='-std=c++11', delete_existing=1)

    # Need this to avoid missing symbol with boost<1.66
    env.PrependUnique(CXXFLAGS=['-DBOOST_ERROR_CODE_HEADER_ONLY'])

    # Platform specific intricacies.
    if env['PLATFORM'] == 'darwin':
        darwin_shlinkflags = [n for n in env['SHLINKFLAGS'] if n != '-dynamiclib']
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
        env.AppendUnique(CCFLAGS=['-Wall', '-fno-strict-aliasing'])
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

    env.Append(CPPPATH=[pjoin(env['prefix'], 'include')])
    env.Append(LIBPATH=[pjoin(env['prefix'], 'lib')])

builddir = env.Dir('build/%s-%s' % (env['build'], platform.machine()))

Export('env', 'pyconfigvar', 'pyoutput', 'pyversion')

if os.path.isfile('sconscript.local'):
    env.SConscript('sconscript.local')

env.SConscript('src/extensions/SConscript', variant_dir=builddir)

# vim: ft=python
