import os

Import('env')

# Build environment configuration --------------------------------------------

# Use Intel C++ compiler when it is available
icpc = env.WhereIs('icpc')
if icpc:
    env.Tool('intelc', topdir=icpc[:icpc.rfind('/bin')])

# Compiler specific options
if icpc:
    # options for Intel C++ compiler on hpc dev-intel07
    env.AppendUnique(CCFLAGS=['-w1', '-fp-model', 'precise'])
    env.PrependUnique(LIBS=['imf'])
    fast_optimflags = ['-fast']
else:
    # g++ options
    env.AppendUnique(CCFLAGS=['-Wall'])
    fast_optimflags = ['-ffast-math']

# Configure build variants
if env['build'] == 'debug':
    env.Append(CCFLAGS='-g')
elif env['build'] == 'fast':
    env.AppendUnique(CCFLAGS=['-O3'] + fast_optimflags)
    env.AppendUnique(CPPDEFINES=['NDEBUG'])

if env['profile']:
    env.AppendUnique(CCFLAGS='-pg')
    env.AppendUnique(LINKFLAGS='-pg')

# Lists for storing built objects and header files
env['newmatobjs'] = []
env['cctbxobjs'] = []
env['objcrystobjs'] = []
env['lib_includes'] = []

# Subsidiary SConscripts -----------------------------------------------------

# These will create the built objects and header file lists.
SConscript(["SConscript.cctbx", "SConscript.newmat", "SConscript.objcryst"])

# Top Level Targets ----------------------------------------------------------

# This retrieves the intermediate objects
newmatobjs = env["newmatobjs"]
cctbxobjs = env["cctbxobjs"]
objcrystobjs = env["objcrystobjs"]

# This builds the shared library
libobjcryst = env.SharedLibrary("libObjCryst", 
        objcrystobjs + cctbxobjs + newmatobjs)
Alias('lib', libobjcryst)

# Installation targets.
prefix = env['prefix']

# install-lib
libdir = env.get('libdir', os.path.join(prefix, 'lib'))
Alias('install-lib', Install(libdir, libobjcryst))

# install-includes
includedir = env.get('includedir', os.path.join(env['prefix'], 'include'))
srcdir = Dir('.').path
def get_target_path(f):
    cutleft = len(srcdir) + 1
    frelsrc = f.path[cutleft:]
    rv = os.path.join(includedir, frelsrc)
    return rv
include_targets = map(get_target_path, env['lib_includes'])

Alias('install-include', InstallAs(include_targets, env['lib_includes']))

# install
Alias('install', ['install-include', 'install-lib'])
