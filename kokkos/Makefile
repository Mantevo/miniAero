KOKKOS_PATH = ${HOME}/kokkos
KOKKOS_DEVICES = OpenMP
KOKKOS_OPTIONS = disable_deprecated_code

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = miniAero.cuda
KOKKOS_ARCH = "Maxwell52"
KOKKOS_CUDA_OPTIONS = enable_lambda
else
CXX = g++
EXE = miniAero.host
KOKKOS_ARCH = "SNB"
endif
CXXFLAGS = -O3 -g -DATOMICS_FLUX
LINK = ${CXX}
LINKFLAGS =  

DEPFLAGS = -M

SRC = $(wildcard *.C)
OBJ = $(SRC:.C=.o)
LIB =

CXXFLAGS += -Isrc
include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

test: $(EXE)
	cd tests; \
	./run_tests.sh ../$(EXE)

# Compilation rules

%.o:%.C $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
