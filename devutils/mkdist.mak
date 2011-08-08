# Makefile for creating ObjCryst bundle.  Only useful for the pyobjcryst
# developers.
#
# Instructions:
#
# (1) checkout https://objcryst.svn.sourceforge.net/svnroot/objcryst/trunk
#     repository to some convenient location
# (2) create a symbolic link srcdir that points to that directory
#
# Usage:
#
# make -f mkdist.mak archive	# create the ObjCryst-rNNNN.tar.gz archive
# make -f mkdist.mak upload	# upload the archive to dev.danse.us/
# make -f mkdist.mak all	# make and upload the archive

SRCDIR = srcdir
TARGET = login.cacr.caltech.edu

PKGROOT = /cacr/home/proj/danse/packages/dev_danse_us
REVISION := $(shell cd $(SRCDIR) && svn info | grep "^Revision:" | cut -b11-)
BUNDLE = ObjCryst-r$(REVISION).tar.gz
LBUNDLE = ObjCryst-latest.tar.gz

ifeq ($(REVISION),)
$(error Invalid SRCDIR.  Symlink the objcryst trunk as srcdir)
endif

.PHONY : archive
archive: $(BUNDLE)

all: archive upload

$(BUNDLE):
	tar -czf $(BUNDLE) -C $(SRCDIR) --exclude='*.o' --exclude='*.so' --exclude='.systemG.Desktop' --exclude='*.a' --exclude='*.exe' --exclude='*.out' --exclude='*.oxy' --exclude='Makefile' --exclude='*.mak' --exclude='profile.0.0.0' --exclude='ObjCryst/doc' --exclude-vcs --exclude='ObjCryst/example' --exclude='ObjCryst/wxCryst' --exclude="*.dep" cctbx newmat ObjCryst --dereference

.PHONY : upload
upload:
	rsync -ruv --progress $(BUNDLE) $(TARGET):$(PKGROOT)
	ssh $(TARGET) "ln -fs $(PKGROOT)/$(BUNDLE) $(PKGROOT)/$(LBUNDLE)"

clean:
	rm -f ObjCryst-r*.tar.gz
