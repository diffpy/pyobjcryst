# This SConstruct is for faster parallel builds.
# Use "setup.py" for normal installation.

MY_SCONS_HELP = """\
SCons rules for compiling and installing pyobjcryst.

Compile and install the pyobjcryst Python extension. 
For faster builds, run with parallel jobs, e.g.:
    scons -j4

Usage: 
    scons [target] [var=value]

Targets:
    (default)   Build the Python extension module `_pyobjcryst.so` (or `.pyd` on Windows)
    dev         install extension module into `src/pyobjcryst/` (development mode)
    test        Run pytest on the package with the installed extension

Build configuration variables:
%s
Variables can be also assigned in a user script sconsvars.py.
SCons construction environment can be customized in sconscript.local script.
"""

import os
from os.path import join as pjoin
import re
import sys


def subdictionary(d, keyset):
    return dict(kv for kv in d.items() if kv[0] in keyset)


def getsyspaths(*names):
    pall = sum((os.environ.get(n, '').split(os.pathsep) for n in names), [])
    rv = [p for p in pall if os.path.exists(p)]
    return rv

def ftpyflag(flags):
    # Figure out compilation switches, filter away fancy flags.
    pattern = re.compile(r'^(-g|-Wstrict-prototypes|-O\d|-fPIC)$')
    return [f for f in flags if not (isinstance(f, str) and pattern.match(f))]

# copy system environment variables related to compilation
DefaultEnvironment(ENV=subdictionary(os.environ, '''
    PATH PYTHONPATH GIT_DIR HOMEPATH HOMEDRIVE
    CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_RUN_PATH
    LD_LIBRARY_PATH DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH
    MACOSX_DEPLOYMENT_TARGET LANG
    _PYTHON_SYSCONFIGDATA_NAME
    _CONDA_PYTHON_SYSCONFIGDATA_NAME
    CONDA_PREFIX
    '''.split())
                   )

# Create construction environment
env = DefaultEnvironment().Clone()

# Variables definitions below work only with 0.98 or later.
env.EnsureSConsVersion(0, 98)

# Customizable compile variables
vars = Variables('sconsvars.py')

# Customizable build variables
vars.Add(EnumVariable(
    'build',
    'compiler settings',
    'fast', allowed_values=('debug', 'fast')))
vars.Add(EnumVariable(
    'tool',
    'C++ compiler toolkit to be used',
    'default', allowed_values=('default', 'intelc')))
vars.Add(BoolVariable(
    'profile',
    'build with profiling information', False))
vars.Update(env)

# Use C++ compiler specified by the 'tool' option.
if env['tool'] == 'intelc':
    icpc = env.WhereIs('icpc')
    if not icpc:
        print("Cannot find the Intel C/C++ compiler 'icpc'.")
        Exit(1)
    env.Tool('intelc', topdir=icpc[:icpc.rfind('/bin')])
    env=env.Clone()
# Default use scons auto found compiler

# Get prefixes, make sure current interpreter is in conda env so thus is the target.
if 'PREFIX' in os.environ:
    default_prefix = os.environ['PREFIX']
elif 'CONDA_PREFIX' in os.environ:
    default_prefix = os.environ['CONDA_PREFIX']
else:
    print("Environment variable PREFIX or CONDA_PREFIX must be set."
          " Activate conda environment.")
    Exit(1)

vars.Add(PathVariable(
    'prefix',
    'installation prefix directory',
    default_prefix))
vars.Update(env)

# Set paths
if env['PLATFORM'] == "win32":
    include_path = pjoin(env['prefix'], 'Library', 'include')
    lib_path = pjoin(env['prefix'], 'Library', 'lib')
    shared_path = pjoin(env['prefix'], 'Library', 'share')

    env.AppendUnique(CPPPATH=[ pjoin(env['prefix'], 'include') ]) # for python headers
    env.AppendUnique(LIBPATH=[ pjoin(env['prefix'], 'libs') ]) # for python libs

    env['ENV']['TMP'] = os.environ.get('TMP', env['ENV'].get('TMP', ''))
else:
    include_path = pjoin(env['prefix'], 'include')
    lib_path = pjoin(env['prefix'], 'lib')
    shared_path = pjoin(env['prefix'], 'share')

vars.Add(PathVariable(
    'includedir',
    'installation directory for C++ header files',
    include_path,
    PathVariable.PathAccept))
vars.Add(PathVariable(
    'libdir',
    'installation directory for compiled programs',
    lib_path,
    PathVariable.PathAccept))
vars.Add(PathVariable(
    'datadir',
    'installation directory for architecture independent data',
    shared_path,
    PathVariable.PathAccept))
vars.Update(env)

env.AppendUnique(CPPPATH=[include_path])
env.AppendUnique(LIBPATH=[lib_path])

env.Help(MY_SCONS_HELP % vars.GenerateHelpText(env))

# Determine python-config script name.
pyversion = os.environ.get('PY_VER') or f"{sys.version_info.major}.{sys.version_info.minor}"
if env['PLATFORM'] != 'win32':
    pythonconfig = pjoin(env['prefix'], 'bin', f'python{pyversion}-config')
    xpython = pjoin(env['prefix'], 'bin', 'python')
else:
    # use sysconfig on Windows
    pythonconfig = None
    xpython = pjoin(env['prefix'], 'python.exe')
print(f"Using python-config: {pythonconfig} from {xpython}")


common_cppdefs = ['REAL=double', 'BOOST_ERROR_CODE_HEADER_ONLY']
env.AppendUnique(CPPDEFINES=common_cppdefs)

if env['PLATFORM'] == 'win32':
    env.AppendUnique(CPPDEFINES=['BOOST_ALL_NO_LIB'])
    env.AppendUnique(CCFLAGS=['/EHsc', '/MD'])

    if env['build'] == 'debug':
        env.Append(CCFLAGS=['/Zi', '/Od', '/FS'])
        env.Append(LINKFLAGS=['/DEBUG'])

    elif env['build'] == 'fast':
        env.Append(CCFLAGS=['/Ox', '/GL'])
        env.Append(LINKFLAGS=['/LTCG', '/OPT:REF', '/OPT:ICF'])

else:
    # get python flags from python-config script
    # not using sysconfig here because of parsing issues
    env.ParseConfig(f"{pythonconfig} --cflags")
    env.Replace(CCFLAGS=ftpyflag(env['CCFLAGS']))

    env.PrependUnique(CCFLAGS=['-Wextra'])
    env.PrependUnique(CXXFLAGS=['-std=c++11'])

    if env['tool'] == 'intelc':
        # options for Intel C++ compiler on hpc dev-intel07
        env.AppendUnique(CCFLAGS=['-w1', '-fp-model', 'precise'])
        env.PrependUnique(LIBS=['imf'])
        fast_opts = ['-fast', '-no-ipo']
    else:
        env.AppendUnique(CCFLAGS=['-fno-strict-aliasing'])
        fast_opts = ['-ffast-math']

    if env['PLATFORM'] == 'darwin':
        # macOS bundle
        sh = [f for f in env['SHLINKFLAGS'] if f != '-dynamiclib']
        env.Replace(SHLINKFLAGS=sh + ['-bundle', '-undefined', 'dynamic_lookup'])
        fast_link = []  # no strip on macOS bundles
    else:
        fast_link = ['-s']

    if env['build'] == 'debug':
        # Python has NDEBUG defined in release builds.
        cppdefs = env.get('CPPDEFINES', [])
        env.Replace(CPPDEFINES=[d for d in cppdefs if d != 'NDEBUG'])

        env.Append(CCFLAGS=['-g', '-O0'])

    elif env['build'] == 'fast':
        env.Append(CCFLAGS=['-O3'] + fast_opts)
        env.Append(LINKFLAGS=fast_link)

builddir = env.Dir('build/%s-%s' % (env['build'], env['PLATFORM']))

Export('env', 'pyversion')

if os.path.isfile('sconscript.local'):
    env.SConscript('sconscript.local')

env.SConscript('src/extensions/SConscript', variant_dir=builddir)

# vim: ft=python
