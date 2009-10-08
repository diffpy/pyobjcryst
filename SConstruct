import os

# copy system environment variables related to compilation
env = DefaultEnvironment(ENV={
        'PATH' : os.environ['PATH'],
        'PYTHONPATH' : os.environ.get('PYTHONPATH', ''),
        'CPATH' : os.environ.get('CPATH', ''),
        'LIBRARY_PATH' : os.environ.get('LIBRARY_PATH', ''),
        'LD_LIBRARY_PATH' : os.environ.get('LD_LIBRARY_PATH', ''),
    }
)

# Use Intel C++ compiler when it is available
icpc = env.WhereIs('icpc')
if icpc:
    env.Tool('intelc', topdir=icpc[:icpc.rfind('/bin')])

# special configuration for specific compilers
if icpc:
    # options for Intel C++ compiler on hpc dev-intel07
    env.AppendUnique(CCFLAGS=['-w1', '-fp-model', 'precise'])
    env.PrependUnique(LIBS=['imf'])
    fast_optimflags = ['-fast']
else:
    # g++ options
    env.AppendUnique(CCFLAGS=['-Wall'])
    fast_optimflags = ['-ffast-math']

env['fast_optimflags'] = fast_optimflags

Export('env')

SConscript(["src/SConscript"])

# vim: ft=python
