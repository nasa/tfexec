SHELL = /bin/sh
FORT = /cryo/sw/intel/17.4/compilers_and_libraries_2017.4.196/linux/bin/intel64/ifort
# CC = /cryo/sw/intel/17.4/compilers_and_libraries_2017.4.196/linux/bin/intel64/icc
# CXX = /cryo/sw/intel/17.4/compilers_and_libraries_2017.4.196/linux/bin/intel64/icpc
CC = /cryo/sw/gcc/8.1.0/bin/gcc
CXX = /cryo/sw/gcc/8.1.0/bin/g++

BUILDDIR = build

all: dir
	cd $(BUILDDIR); CC=$(CC) CXX=$(CXX) cmake3 ..; -DML_EXPORTS make
	$(FORT) -c src/ML_modules.f90 -o $(BUILDDIR)/ML_modules.o
	$(FORT) -c src/main.f90 -o $(BUILDDIR)/main.o
	$(FORT) build/*.o -cxxlib \
		-Lbuild/tfexec -I../../../include -ltfexec \
		-Ibuild/_deps/tensorflowcontent-src/include \
		-Lbuild/_deps/tensorflowcontent-src/lib \
		-ltensorflow -ltensorflow_framework -o $(BUILDDIR)/fortran_example
	

debug:
	cd $(BUILDDIR); CC=$(CC) CXX=$(CXX) cmake3 -DML_EXPORTS -DCMAKE_BUILD_TYPE=Debug ..; make
	ifort -c src/ML_modules.f90 -g -o $(BUILDDIR)/ML_modules.o
	ifort -c src/main.f90 -g -o $(BUILDDIR)/main.o
	ifort build/*.o -g -cxxlib \
		-Lbuild/tfexec -I../../../include -ltfexec \
		-Ibuild/_deps/tensorflowcontent-src/include \
		-Lbuild/_deps/tensorflowcontent-src/lib \
		-ltensorflow -ltensorflow_framework -o $(BUILDDIR)/fortran_example_debug
    
clean:
	cd $(BUILDDIR); make clean
	rm $(BUILDDIR)/main.o $(BUILDDIR)/ML_modules.o
	rm -f $(BUILDDIR)/fortran_example $(BUILDDIR)/fortran_example_debug
 
dir:
	mkdir -p $(BUILDDIR)