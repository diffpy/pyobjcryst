env = DefaultEnvironment()

# Environment variables
import os
CPPPATH = os.getenv("INCLUDE", "").split(os.path.pathsep)
LIBPATH = os.getenv("LIB", "").split(os.path.pathsep)
env.AppendUnique(CPPPATH = CPPPATH)
env.AppendUnique(LIBPATH = LIBPATH)

# Get the compiler type
from distutils.ccompiler import new_compiler
env["compiler_type"] =  new_compiler().compiler_type

# Set some common flags
if env["compiler_type"] == "msvc":
    CCFLAGS = ["/MD", "/GR", "/EHsc", "/Gy", "/GF", "/GA"]
    env.AppendUnique(CFLAGS = CFLAGS)

Export('env')

SConscript(["src/SConscript"])
