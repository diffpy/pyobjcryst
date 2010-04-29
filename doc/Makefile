
all: doc upload

RELEASE = beta1
TARGET = farrowch@login.cacr.caltech.edu
DOCROOT = ~/docroot/diffraction/
PKGROOT = ~/dev_danse_us/

.PHONY : doc
doc:
	epydoc pyobjcryst --html -vvv -o pyobjcrystapi -n pyobjcryst \
--include-log --show-imports $@
	$(MAKE) -C manual $@

.PHONY : upload
upload:
	rsync -ruvz --delete pyobjcrystapi $(TARGET):$(DOCROOT)
	ssh $(TARGET) "rm -rf $(DOCROOT)/pyobjcrystapi-$(RELEASE)"
	ssh $(TARGET) "cp -r $(DOCROOT)/pyobjcrystapi $(DOCROOT)/pyobjcrystapi-$(RELEASE)"
	$(MAKE) -C manual $@

.PHONY : clean
clean:
	rm -rf pyobjcrystapi
	$(MAKE) -C manual $@

