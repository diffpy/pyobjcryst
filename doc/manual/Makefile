
all: doc upload

RELEASE = beta0
TARGET = farrowch@login.cacr.caltech.edu
DOCROOT = ~/docroot/diffraction/
PKGROOT = ~/dev_danse_us/

.PHONY : doc
doc:
	sphinx-build . pyobjcryst

.PHONY : upload
upload:
	rsync -ruvz --delete pyobjcryst $(TARGET):$(DOCROOT)
	ssh $(TARGET) "rm -rf $(DOCROOT)/pyobjcryst-$(RELEASE)"
	ssh $(TARGET) "cp -r $(DOCROOT)/pyobjcryst $(DOCROOT)/pyobjcryst-$(RELEASE)"

.PHONY : clean
clean:
	rm -rf pyobjcryst

