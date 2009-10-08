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

Export('env')

SConscript(["src/SConscript"])

# vim: ft=python
