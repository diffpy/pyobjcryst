import os.path

# Declare location of objcryst files
AddOption('--objcrystheaders',
          dest='objcrystheaders',
          type='string',
          nargs=1,
          action='store',
          metavar='DIR',
          default='',
          help='Location of Objcryst header files')

# Auxillary library location
AddOption('--auxlib',
          dest='auxlib',
          type='string',
          nargs=1,
          action='store',
          metavar='DIR',
          default='',
          help='Location of auxillary libraries')


env = DefaultEnvironment()

env.Append(CPPPATH = GetOption("objcrystheaders"))
env.Append(LIBPATH = GetOption("auxlib"))

env.ParseConfig("python-config --includes")
env.ParseConfig("python-config --ldflags")

Export('env')

SConscript("boost/SConscript")
