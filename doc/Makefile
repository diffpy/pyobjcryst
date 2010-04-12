
all: doc upload

RELEASE = beta0
TARGET = farrowch@login.cacr.caltech.edu
DOCROOT = ~/docroot/diffraction/
PKGROOT = ~/dev_danse_us/

.PHONY : doc
doc:
	epydoc pyobjcryst --html -vvv -o pyobjcrystapi -n pyobjcryst \
--include-log --introspect $@
	$(MAKE) -C manual $@

.PHONY : upload
upload:
	rsync -ruvz --delete pyobjcrystapi $(TARGET):$(DOCROOT)
	ssh $(TARGET) "rm -rf $(DOCROOT)/pyobjcrystapi-$(RELEASE)"
	ssh $(TARGET) "cp -r $(DOCROOT)/pyobjcrystapi $(DOCROOT)/pyobjcrystapi-$(RELEASE)"
	rsync -ruv srfit_examples.zip $(TARGET):$(PKGROOT)
	ssh $(TARGET) "rm -rf $(PKGROOT)/srfit_examples-$(RELEASE).zip"
	ssh $(TARGET) "cp -r $(PKGROOT)/srfit_examples.zip $(PKGROOT)/srfit_examples-$(RELEASE).zip"
	$(MAKE) -C devmanual $@

.PHONY : clean
clean:
	rm -rf pyobjcrystapi
	$(MAKE) -C devmanual $@

