.PHONY : all
all : ${subst _ext.cpp,,${wildcard *_ext.cpp}}

%:
	MODULE=$@ make -f module.make

.PHONY : clean
clean: 
	make -f module.make clean
