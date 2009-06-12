import os.path

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

env.Append(LIBPATH = GetOption("auxlib"))

Export('env')

SConscript(["src/SConscript", "boost/SConscript"])

# We should have
