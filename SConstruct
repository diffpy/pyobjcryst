env = DefaultEnvironment()

# Environment variables
import os
CPPPATH = os.getenv("CPATH", "").split(os.path.pathsep)
LIBPATH = os.getenv("LIBRARY_PATH", "").split(os.path.pathsep)
env.AppendUnique(CPPPATH = CPPPATH)
env.AppendUnique(LIBPATH = LIBPATH)

Export('env')

SConscript(["src/SConscript"])
