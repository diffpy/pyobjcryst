# Makefile for creating ObjCryst bundle
SRCDIR = /home/chris/.local/src/ObjCryst
TARGET = farrowch@login.cacr.caltech.edu
PKGROOT = ~/dev_danse_us/
JUNK = $(shell svn up $(SRCDIR))
BUNDLE = ObjCryst-r$(shell svn info $(SRCDIR) | grep "^Revision:" | cut -b11-).tar.gz
LBUNDLE = ObjCryst-latest.tar.gz

all: archive upload

.PHONY : archive
archive:
	tar -czf $(BUNDLE) -C $(SRCDIR) --exclude='*.o' --exclude='*.so' --exclude='.systemG.Desktop' --exclude='*.a' --exclude='*.exe' --exclude='*.out' --exclude='*.oxy' --exclude='Makefile' --exclude='*.mak' --exclude='profile.0.0.0' --exclude='ObjCryst/doc' --exclude-vcs --exclude='ObjCryst/example' --exclude='ObjCryst/wxCryst' --exclude="*.dep" cctbx newmat ObjCryst --dereference

.PHONY : upload
upload:
	rsync -ruv --progress $(BUNDLE) $(TARGET):$(PKGROOT)
	ssh $(TARGET) "ln -fs $(PKGROOT)/$(BUNDLE) $(PKGROOT)/$(LBUNDLE)"
